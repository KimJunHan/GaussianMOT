from typing import Any, Dict
import contextlib  # ④ 옵션 B: prev trunk를 no_grad로 감싸기 위한 조건부 context
import os          # [진단 토글] GC_DENSIFY 환경변수 읽기

import numpy as np  # [A recipe] radar 팽창 splat 가우시안 가중치 계산용
import torch
import torch.nn as nn
import rootutils
from einops import rearrange  # 텐서 reshape/축 재배열을 가독성 있게 표현하는 유틸

# 프로젝트 루트(.project-root 기준)를 sys.path에 등록 — 절대 import가 동작하도록.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# GS 래스터라이저 래퍼: Gaussian → BEV feature 렌더.
from gaussianmot.render import GaussianRenderer


class GaussianMOT(nn.Module):
    # GaussMOT/GaussianMOT 메인 모델.
    # 전체 흐름: 멀티뷰 이미지·radar 포인트 → 각각 Gaussian으로 변환 → BEV로 splat(렌더)
    #            → camera/radar BEV 융합 → DPT 디코더로 multi-scale BEV → det/track head.
    # GaussMOT 추가 요소: ① augmented Gaussian(velocity+identity) ② temporal ego-warp
    #                     ③ tracking-aware densification (모두 zero-init/학습-only로 baseline 보존).
    def __init__(
        self,
        embed_dims: int,            # Gaussian feature / BEV feature 채널 수
        depth_num: int,             # depth bin 개수 (per-pixel depth 분포 이산화)
        depth_min: int,             # depth bin 최소 거리(m)
        depth_max: int,             # depth bin 최대 거리(m)
        error_tolerance: float,     # depth 불확실성 → Gaussian 공분산 스케일 변환 계수
        opacity_filter: float,      # 렌더 시 opacity 임계값
        x_min: float = -50,
        x_max: float = 50,
        y_min: float = -50,
        y_max: float = 50,
        bev_h: int = 200,
        bev_w: int = 200,
        image_encoder: nn.Module = None,   # PixelsToGaussians: 이미지 → per-pixel Gaussian
        radar_encoder: nn.Module = None,   # PointsToGaussians: radar 포인트 → Gaussian
        fuser: nn.Module = None,           # CMXFuser: camera/radar BEV 융합
        decoder: nn.Module = None,         # DPTHead: multi-scale BEV feature 디코딩
        head: nn.Module = None,            # detection head (center/offset/dim/angle/vel 등)
        track_head: nn.Module = None,      # tracking head (per-instance embedding)
        use_tracking: bool = False,        # track_head 실행 여부
        temporal_warp: bool = False,       # 이전 프레임 BEV ego-warp 사용 여부(temporal)
        # [multi-frame 8/21] temporal 융합에 쓸 이전 프레임 수 K (RCTrans식 다중프레임).
        #   data.data_config.num_prev_frames와 반드시 같은 값으로 둘 것. 1=기존 동작.
        max_prev_frames: int = 1,
        # [B③] 박스레벨 radar doppler 융합: radar 보정속도(vel_cols)를 BEV 맵(vx,vy,mask)으로
        #   splat해 head의 vel_head 입력에 직접 공급 (head.use_radar_vel_fusion과 세트).
        radar_vel_fusion: bool = False,
        radar_vel_cols: tuple = (15, 16),  # radar_points에서 보정속도(vx,vy) 컬럼 (DopplerLoss와 동일)
        radar_xy_cols: tuple = (0, 1),     # radar_points에서 (x,y) 좌표 컬럼
    ) -> None:
        super().__init__()

        # Parameters.
        self.embed_dims = embed_dims
        # BEV render extents (metric, augmented frame) — needed for the geometric
        # ego-warp grid (metric ↔ normalized grid_sample coords).
        self.bev_x_min, self.bev_x_max = float(x_min), float(x_max)
        self.bev_y_min, self.bev_y_max = float(y_min), float(y_max)
        self.depth_num = depth_num
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.error_tolerance = error_tolerance
        self.opacity_filter = opacity_filter

        # Modules.
        # camera/radar 인코더는 외부(Hydra)에서 주입된다.
        self.image_encoder = image_encoder
        self.radar_encoder = radar_encoder
        # camera 경로 전용 GS 렌더러: 카메라 Gaussian → BEV feature.
        self.gs_render_image = GaussianRenderer(
            embed_dims,
            opacity_filter,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bev_h=bev_h,
            bev_w=bev_w,
        )
        # radar 경로 전용 GS 렌더러: radar Gaussian → 동일 BEV 그리드로 렌더.
        self.gs_render_radar = GaussianRenderer(
            embed_dims,
            opacity_filter,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bev_h=bev_h,
            bev_w=bev_w,
        )
        self.fuser = fuser              # camera BEV + radar BEV 융합 모듈
        self.decoder = decoder          # 융합 BEV → multi-scale feature 디코더
        self.head = head                # detection head
        # [B③] radar 속도맵 융합 설정
        self.radar_vel_fusion = bool(radar_vel_fusion)
        self.radar_vel_cols = tuple(radar_vel_cols)
        self.radar_xy_cols = tuple(radar_xy_cols)
        self.bev_h_grid, self.bev_w_grid = int(bev_h), int(bev_w)
        self.track_head = track_head    # tracking head
        self.use_tracking = use_tracking

        # Augmented-Gaussian injection (GaussMOT ①): residual-inject the splatted
        # [velocity(2), identity(K)] BEV fields into the full-res fused BEV so the
        # motion/identity signal informs both detection (mAVE) and tracking (AMOTA).
        # Zero-init → training starts identical to the baseline, then learns to use it.
        # 한글: ① 각 Gaussian에 붙은 velocity(2)+identity(K)를 BEV로 splat한 뒤, 그 필드를
        #       1x1 conv로 embed_dims에 맞춰 융합 BEV에 잔차(residual)로 더한다.
        #       weight/bias를 0으로 초기화 → 학습 초기엔 baseline과 완전히 동일, 이후 학습으로 활용.
        self.identity_dim = 8                                                   # identity 임베딩 채널 수(K)
        self.aug_proj = nn.Conv2d(2 + self.identity_dim, embed_dims, kernel_size=1)  # [velocity+identity] → embed_dims
        nn.init.zeros_(self.aug_proj.weight)                                   # zero-init: 잔차 기여 0에서 시작
        nn.init.zeros_(self.aug_proj.bias)

        # Geometric ego-warp (temporal): resample the prev-frame fused BEV into the
        # current frame via `bev_warp` (4x4, from the data pipeline) and residual-inject.
        # Zero-init projection → training starts identical to the single-frame baseline,
        # then learns to exploit the temporal context (motion → mAVE, identity → AMOTA).
        # 한글: ② 이전 프레임의 융합 BEV를 데이터 파이프라인이 준 bev_warp(4x4 ego-motion)로
        #       현재 프레임 좌표계에 정렬(grid_sample)한 뒤 잔차로 더해 시간적 문맥을 주입.
        #       역시 zero-init proj → 처음엔 단일-프레임 baseline과 동일.
        self.temporal_warp = bool(temporal_warp)
        # [multi-frame 8/21] 사용할 이전 프레임 수 K (data.num_prev_frames와 일치시킬 것).
        #   K=1이면 기존과 완전 동일. K≥2면 프레임별 zero-init proj를 추가로 만든다.
        self.max_prev_frames = max(1, int(max_prev_frames))
        if self.temporal_warp:
            self.temporal_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)  # warp된 prev BEV → 잔차 투영
            nn.init.zeros_(self.temporal_proj.weight)                              # zero-init
            nn.init.zeros_(self.temporal_proj.bias)
            for k in range(2, self.max_prev_frames + 1):
                proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
                nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
                setattr(self, f"temporal_proj{k}", proj)

        # Tracking-aware adaptive densification (GaussMOT ③): TRAINING-ONLY clone/split
        # of high-opacity Gaussians near GT object centers, with identity inheritance.
        # Skipped at inference (self.training False) → zero inference cost.
        # 한글: ③ 학습 중에만, GT 객체 중심 근처의 opacity 높은 Gaussian을 복제(clone)/분할(split)해
        #       객체 영역의 Gaussian 밀도를 높인다(identity 상속). 추론 때는 건너뛰어 비용 0.
        #
        # [2026-06-25 비활성화 — GT 누출] 이 densify는 GT 박스 위치를 학습 입력 BEV에 직접
        # 주입(train-only)하는데 추론 때는 꺼져 train/test mismatch를 만든다. 그 결과 24ep eval
        # mAP가 baseline 0.32 → 0.173으로 회귀함이 eval로 확정됨([[project_densify_gt_leak]]).
        # 따라서 GT-누출 버전은 비활성화한다. contribution으로 되살리려면 GT-free(opacity-only
        # 또는 예측-center, train/test 동일)로 재구현 후 A/B로 검증할 것.
        # [2026-07-07 GT-free 재설계·활성화] opacity·크기 기반(3DGS식)으로 재구현 → GT 미사용이라
        #   train/test 동일 실행(추론에서도) → GT-leak 해소. 큰 Gaussian=split(공분산↓), 작은 것=clone.
        # [진단 토글] GC_DENSIFY=0 환경변수로 끌 수 있음 (densify ablation eval용).
        self.densify_train = os.environ.get("GC_DENSIFY", "1") != "0"
        self.densify_n_extra = 512      # fixed extra slots (zero-opacity padded)  # 추가 슬롯 고정 개수(미사용분은 opacity 0)
        self.densify_opacity_thr = 0.1                                              # densify 대상 opacity 하한(중요 Gaussian)
        self.densify_jitter = 0.2       # m: clone position jitter std              # clone 시 위치 흔들기 표준편차(m)

        # Geometry for PixelsToGaussians.
        # depth bin 중심값 버퍼 — per-pixel depth 분포로부터 3D 위치를 복원할 때 사용.
        bins = self._init_bin_centers()
        self.register_buffer("bins", bins, persistent=False)

    def _init_bin_centers(self):
        # depth_min~depth_max 구간을 depth_num개로 나눈 각 bin의 "중심" 거리값을 계산.
        depth_range = self.depth_max - self.depth_min      # 전체 depth 범위
        interval = depth_range / self.depth_num            # bin 간격(균등)
        interval = interval * torch.ones((self.depth_num+1))  # [depth_num+1] 간격 벡터
        interval[0] = self.depth_min                       # 첫 항을 시작 거리로 설정
        bin_edges = torch.cumsum(interval, 0)              # 누적합 → bin 경계(edge)들
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # 인접 edge 평균 → bin 중심
        return bin_centers
    
    @torch.no_grad()
    def _get_pixel_coords_3d(
        self,
        coords_d,        # depth bin 중심값 [D]
        depth,           # depth 예측 텐서(해상도 참조용) [..., H, W]
        lidar2img,       # lidar→이미지 투영 행렬 [B, N, 4, 4]
        img_h=224,
        img_w=480,
    ):
        # 각 픽셀 × 각 depth bin 위치를 lidar(3D) 좌표로 역투영한 그리드를 만든다.
        eps = 1e-5

        B, N = lidar2img.shape[:2]               # B=batch, N=카메라 뷰 수
        H, W = depth.shape[-2:]                  # depth feature 해상도
        # feature 격자 좌표를 0~1로 정규화 후 원본 이미지 픽셀 좌표로 스케일.
        coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
        coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w

        D = coords_d.shape[0]                    # depth bin 개수
        # (w, h, d) 3D 격자 생성 → [W, H, D, 3]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)   # 동차좌표화: [W,H,D,4]
        # 이미지→lidar 역투영 전, 픽셀 (u,v)에 depth z를 곱해 카메라 좌표계로 변환(원근 역투영).
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        dtype = lidar2img.dtype
        img2lidars = lidar2img.float().inverse().to(dtype) # b n 4 4         # lidar2img 역행렬 = img→lidar

        # 격자와 변환행렬을 [B, N, W, H, D, ...]로 broadcast하기 위한 차원 정렬/복제.
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)        # [B,N,W,H,D,4,1]
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)  # [B,N,W,H,D,4,4]
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3  # 역투영 후 xyz만

        return coords3d, coords_d                # 각 픽셀×bin의 3D 위치, depth bin 값

    def _pred_depth(self, lidar2img, depth, img_h, img_w, coords_3d=None):
        # per-pixel depth 분포로부터 각 픽셀 Gaussian의 3D 평균(mean)과 공분산(cov)을 추정.
        if coords_3d is None:
            # 픽셀×bin 3D 격자 생성 후 [B,N,W,H,D,3] -> [(B*N), D, H, W, 3]로 재배열.
            coords_3d, coords_d = self._get_pixel_coords_3d(self.bins, depth, lidar2img, img_h=img_h, img_w=img_w)
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')

        depth_prob = depth.softmax(1)                              # depth 채널(D) softmax → bin별 확률
        # 기대 위치: bin별 확률 가중 합 → 픽셀당 3D 평균 위치. [(BN), H, W, 3]
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)

        # 각 bin 위치 - 평균 위치 = 편차. 확률 가중 outer-product 합 → 3x3 공분산.
        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9                    # 허용오차 → 공분산 스케일 보정 계수
        cov = cov * scale

        return pred_coords_3d, cov                                 # Gaussian 중심(mean)과 공분산

    def forward_features_camera(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # 카메라 경로: 멀티뷰 이미지 → PixelsToGaussians → per-pixel Gaussian 파라미터.

        # Arrange the input images.
        B, N, _, H, W = batch["image"].shape          # B=batch, N=카메라 뷰 수
        images = batch["image"].flatten(0, 1).contiguous()   # [B, N, 3, H, W] -> [B*N, 3, H, W]
        lidar2img = batch["lidar2img"].contiguous()          # [B, N, 4, 4] 투영 행렬

        # Process images.
        feats = self.image_encoder(images)            # 인코더: depth/offsets/features/opacity/velocity/identity 예측
        # depth 분포 → 픽셀별 Gaussian 중심(mean)·공분산(cov).
        means_3d, covs_3d = self._pred_depth(lidar2img, feats["depth"], H, W)
        # 예측된 offset을 더해 중심 위치를 미세 보정. [(BN),d,H,W] -> [(BN),H,W,d]로 맞춰 더함.
        means_3d = means_3d + rearrange(feats["offsets"], "(b n) d h w -> (b n) h w d", b=B, n=N)
        covs_3d = covs_3d.flatten(-2, -1)             # 3x3 공분산 [.. ,3,3] -> [.., 9]
        # 대칭 공분산이라 상삼각 6개 성분만 추출: (00,01,02,11,12,22).
        covs_3d = torch.cat((covs_3d[..., 0:3], covs_3d[..., 4:6], covs_3d[..., 8:9]), dim=-1)

        # 모든 뷰의 픽셀을 한 batch의 Gaussian 집합으로 평탄화: [(BN),d,H,W] -> [B, N*H*W, d]
        feats_3d = rearrange(feats["features"], '(b n) d h w -> b (n h w) d', b=B, n=N)
        means_3d = rearrange(means_3d, '(b n) h w d-> b (n h w) d', b=B, n=N)        # [B, G, 3]
        covs_3d = rearrange(covs_3d, '(b n) h w d -> b (n h w) d',b=B, n=N)          # [B, G, 6]
        opacities_3d = rearrange(feats["opacity"], '(b n) d h w -> b (n h w) d', b=B, n=N)  # [B, G, 1]
        # Augmented Gaussian primitive (①): per-Gaussian velocity + identity.
        # ① Gaussian마다 속도(velocity)·정체성(identity) 속성을 함께 가져온다.
        velocities_3d = rearrange(feats["velocity"], '(b n) d h w -> b (n h w) d', b=B, n=N)  # [B, G, 2]
        identities_3d = rearrange(feats["identity"], '(b n) d h w -> b (n h w) d', b=B, n=N)  # [B, G, K]

        return {
            "features": feats_3d,        # Gaussian별 feature 벡터
            "centers": means_3d,         # Gaussian 중심(3D)
            "covariances": covs_3d,      # Gaussian 공분산(6)
            "opacities": opacities_3d,   # Gaussian opacity
            "velocities": velocities_3d, # ① 속도 속성
            "identities": identities_3d, # ① identity 속성
        }

    def forward_features_radar(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # radar 경로: 가변 길이 radar 포인트 리스트 → PointsToGaussians(PTv3) → Gaussian dict.
        radar_gaussians = self.radar_encoder(batch["radar_points"])
        return radar_gaussians

    def _render_augmented(self, gaussians: Dict[str, Any], renderer) -> Dict[str, torch.Tensor]:
        """Augmented Gaussian render (①): splat per-Gaussian velocity (2) + identity (K)
        through the SAME rasterizer by packing them (zero-padded to embed_dims) — no CUDA
        rebuild. Returns BEV velocity/identity maps. Returns empty dict if the encoder did
        not emit augmented attributes."""
        # 인코더가 ① 속성을 안 내놨으면 그냥 빈 dict 반환(=비활성).
        if "velocities" not in gaussians or "identities" not in gaussians:
            return {}
        vel = gaussians["velocities"]          # [b, G, 2]
        idt = gaussians["identities"]          # [b, G, K]
        b, G, _ = vel.shape
        k = idt.shape[-1]
        # 같은 래스터라이저(embed_dims 채널)를 재사용하려고, velocity+identity를
        # embed_dims 길이 벡터의 앞쪽 채널에 채우고 나머지는 0으로 패딩.
        packed = torch.zeros(b, G, self.embed_dims, device=vel.device, dtype=vel.dtype)
        packed[..., 0:2] = vel                 # 0~1 채널: 속도
        packed[..., 2:2 + k] = idt             # 2~2+k 채널: identity
        # feature 자리에 packed를 넣어 동일 Gaussian 기하로 BEV에 splat (CUDA 재빌드 불필요).
        aug_bev, _ = renderer(
            packed, gaussians["centers"], gaussians["covariances"], gaussians["opacities"]
        )                                       # [b, embed_dims, H, W]
        return {
            "velocity_bev": aug_bev[:, 0:2],    # BEV 속도 필드 [b,2,H,W]
            "identity_bev": aug_bev[:, 2:2 + k],# BEV identity 필드 [b,K,H,W]
        }

    def _densify_gaussians(self, gaussians: Dict[str, Any]) -> Dict[str, Any]:
        """GT-free adaptive densification (3DGS식). opacity 높은(중요) Gaussian을 **크기 기준**으로
        분할/복제한다: 큰 것 = split(공분산×0.5 → 더 작게), 작은 것 = clone(위치 jitter 복제 → 밀도↑).
        GT 미사용 → train/test 동일하게 실행(추론에서도) → 이전 GT-leak 문제 해소.
        고정 N 슬롯을 append(미사용은 opacity 0 → 렌더 opacity 필터가 자동 제거). ① 속성 없으면 no-op."""
        if "velocities" not in gaussians or "identities" not in gaussians:
            return gaussians
        keys = ["centers", "covariances", "features", "opacities", "velocities", "identities"]
        opac = gaussians["opacities"]            # [B, G, 1]
        cov = gaussians["covariances"]           # [B, G, 6] = [xx,xy,xz,yy,yz,zz]
        B = opac.shape[0]; device = opac.device
        N = self.densify_n_extra
        size = cov[:, :, 0] + cov[:, :, 3]       # [B, G] 2D 공간 크기(xx+yy)
        extra = {k: torch.zeros(B, N, gaussians[k].shape[-1], device=device, dtype=gaussians[k].dtype)
                 for k in keys}
        for b in range(B):
            # GT-free 선택: opacity 높은(중요) Gaussian만 densify 대상.
            imp = (opac[b, :, 0] > self.densify_opacity_thr).nonzero(as_tuple=False).squeeze(1)
            if imp.numel() == 0:
                continue
            if imp.numel() > N:                  # 슬롯 초과 시 opacity 상위 N개
                imp = imp[opac[b, imp, 0].topk(N).indices]
            n = imp.numel()
            block = {k: gaussians[k][b, imp].clone() for k in keys}   # 선택 Gaussian 복제 블록 [n, dim]
            sz = size[b, imp]
            large = sz > sz.median()             # split=큰 것 / clone=작은 것 (크기 기준, 3DGS식)
            small = ~large
            block["covariances"][large] = block["covariances"][large] * 0.5   # split: 공분산 축소
            if bool(small.any()):                                             # clone: 위치 jitter
                jit = torch.randn(int(small.sum()), 2, device=device) * self.densify_jitter
                block["centers"][small, :2] = block["centers"][small, :2] + jit
            for k in keys:
                extra[k][b, :n] = block[k]
        out = dict(gaussians)
        for k in keys:
            out[k] = torch.cat([gaussians[k], extra[k]], dim=1)   # 원본 + 추가 슬롯을 G축으로 concat
        return out

    def _warp_prev_bev(
        self, prev_bev: torch.Tensor, bev_warp: torch.Tensor
    ) -> torch.Tensor:
        """Resample a previous-frame BEV feature into the current frame.

        Args:
            prev_bev: [B, C, H, W] feature in the *prev* augmented BEV frame.
            bev_warp: [B, 4, 4] metric transform mapping current-augmented (x, y, z)
                → prev-augmented (x, y, z). `affine_grid`'s theta maps output (current)
                normalized coords → input (prev) normalized coords, which is exactly
                this direction.

        Returns:
            [B, C, H, W] prev feature resampled onto the current BEV grid.
        """
        B, C, H, W = prev_bev.shape
        dev, dt = prev_bev.device, prev_bev.dtype
        bw = bev_warp.to(device=dev, dtype=dt)

        # Planar (x, y) part of the 4x4 (BEV collapses z): linear R + translation t.
        # BEV는 z를 무시하므로 4x4 중 평면(x,y) 부분만 사용: 회전 R + 평행이동 t.
        R = bw[:, :2, :2]            # [B, 2, 2]  평면 회전/선형부
        t = bw[:, :2, 3]             # [B, 2]   (meters)  평면 평행이동(미터)

        # metric m = c + D g, where g ∈ [-1, 1]^2 is grid_sample's (x=col, y=row) coord.
        #   x = cx - sx * g_y ,  y = cy - sy * g_x   (BEV: +x→row up, +y→col left)
        # 한글: grid_sample 정규화 좌표 g(±1)와 metric 좌표 m 사이 변환을 정의.
        #       BEV 좌표 규약(+x→행 위, +y→열 왼쪽)을 반영해 D/c 행렬 구성.
        sx = (self.bev_x_max - self.bev_x_min) / 2.0   # x 반폭(미터)
        sy = (self.bev_y_max - self.bev_y_min) / 2.0   # y 반폭(미터)
        cx = (self.bev_x_max + self.bev_x_min) / 2.0   # x 중심(미터)
        cy = (self.bev_y_max + self.bev_y_min) / 2.0   # y 중심(미터)
        D = torch.tensor([[0.0, -sx], [-sy, 0.0]], device=dev, dtype=dt)        # g → metric 선형부
        Dinv = torch.tensor([[0.0, -1.0 / sy], [-1.0 / sx, 0.0]], device=dev, dtype=dt)  # 그 역변환
        c = torch.tensor([cx, cy], device=dev, dtype=dt)

        # g_prev = (Dinv R D) g_cur + Dinv (R c + t - c)
        # 현재 프레임 grid 좌표(g_cur)를 prev 프레임 grid 좌표(g_prev)로 보내는 affine theta 합성.
        theta_lin = torch.matmul(torch.matmul(Dinv, R), D)               # [B, 2, 2]  선형부
        rhs = torch.matmul(R, c) + t - c                                 # [B, 2]
        theta_tr = torch.matmul(Dinv, rhs.unsqueeze(-1))                 # [B, 2, 1]  평행이동부
        theta = torch.cat([theta_lin, theta_tr], dim=-1)                 # [B, 2, 3]  affine 행렬

        # theta로 샘플링 그리드를 만들고 prev BEV를 현재 좌표계로 bilinear 리샘플.
        grid = torch.nn.functional.affine_grid(
            theta, [B, C, H, W], align_corners=False
        )
        return torch.nn.functional.grid_sample(
            prev_bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    def forward_track_only(self, batch: Dict[str, Any], track_head_grad_only: bool = False) -> Dict[str, Any]:
        """Phase 3b: prev frame 처리용 경량 forward.
        full forward와 동일 경로(encoder → render → fuse → decode)를 거치되
        detection_head, aux_head, density 출력은 스킵하고 track_head만 실행해서
        prev frame의 메모리·연산 비용을 줄임. track_embed가 없으면 빈 dict.

        track_head_grad_only=True (④ 옵션 B): trunk(encoder~decode)는 no_grad로 돌려
        grad를 흘리지 않고 track_head에만 grad를 흘린다. → prev 프레임 contrastive grad가
        공유 trunk를 detection에서 끌어당기는 것(det loss 상승)을 막으면서, prev embedding의
        cross-frame positive 학습은 유지. 부수효과로 prev trunk activation 미보존 → 메모리·backward↓.
        """
        # encoder → render → fuse → decode (현재 forward와 동일 경로).
        # track_head_grad_only면 trunk 전체를 no_grad로 감싼다(아니면 nullcontext=주변 grad 상태 유지).
        trunk_ctx = torch.no_grad() if track_head_grad_only else contextlib.nullcontext()
        with trunk_ctx:
            camera_gaussians = self.forward_features_camera(batch)   # 이미지 → Gaussian
            radar_gaussians = self.forward_features_radar(batch)     # radar → Gaussian

            camera_bev, _ = self.gs_render_image(                    # 카메라 Gaussian → BEV
                camera_gaussians["features"], camera_gaussians["centers"],
                camera_gaussians["covariances"], camera_gaussians["opacities"],
            )
            radar_bev, _ = self.gs_render_radar(                     # radar Gaussian → BEV
                radar_gaussians["features"], radar_gaussians["centers"],
                radar_gaussians["covariances"], radar_gaussians["opacities"],
            )
            fused_bev = self.fuser(camera_bev, radar_bev)            # 융합
            decoded = self.decoder(fused_bev)                        # 디코딩

        out: Dict[str, Any] = {}
        # det/aux head는 생략하고 track head만 실행(경량). track_head는 위 with 밖이라
        # track_head_grad_only=True여도 track_head 파라미터에는 grad가 흐른다(입력 decoded/skip은 no_grad).
        if self.use_tracking and self.track_head is not None:
            # ⑤ skip: 현재 forward와 동일하게 융합 BEV(scale-0)를 함께 전달.
            skip_feat = fused_bev[0] if isinstance(fused_bev, (list, tuple)) else fused_bev
            out = dict(self.track_head(decoded, skip_feat))
        # Temporal ego-warp: expose the full-res fused BEV (cr1) so the current-frame
        # forward can warp it into place. Detached/no-grad upstream in the module.
        # 현재 프레임이 ego-warp 입력으로 쓸 수 있도록 full-res 융합 BEV를 그대로 노출.
        if self.temporal_warp:
            out["bev_feat"] = fused_bev[0] if isinstance(fused_bev, (list, tuple)) else fused_bev
        return out

    @torch.no_grad()   # 측정값 splat이므로 gradient 불필요 (DopplerLoss._build_target과 동일 규칙)
    def _build_radar_vel_map(self, radar_points, device, dtype):
        """[A recipe] radar 보정속도를 BEV 맵 [B,5,H,W] (vx, vy, occ, losx, losy)로 팽창 splat.

        진단(8/14)으로 확정된 설계 결함 2개를 고침:
        ① 단일셀 scatter → vel_head 수용영역(±1.25m)에 측정값이 안 보임
           ⇒ 반경 R셀 disk로 거리가중(gaussian) 팽창 splat.
        ② doppler는 radial 성분만 측정하는데 conv는 위치불변이라 시선방향을 모름
           ⇒ LOS 단위벡터(losx,losy) 채널 추가 → v_full=(s/ĥ·r̂)ĥ 재구성을 학습 가능하게.
        좌표 규칙은 DopplerLoss와 동일(col = W/2 - y/res, row = H/2 - x/res).
        """
        H, W = self.bev_h_grid, self.bev_w_grid
        res_row = (self.bev_x_max - self.bev_x_min) / H
        res_col = (self.bev_y_max - self.bev_y_min) / W
        B = len(radar_points)
        out = torch.zeros(B, 5, H, W, device=device, dtype=dtype)
        R = 3                                   # 팽창 반경 [셀] (=1.5m)
        sigma2 = 2.0 * (R / 2.0) ** 2
        offsets = [(di, dj) for di in range(-R, R + 1) for dj in range(-R, R + 1)
                   if di * di + dj * dj <= R * R]
        for b, pts in enumerate(radar_points):
            if pts is None or pts.numel() == 0:
                continue
            pts = pts.to(device).float()
            x = pts[:, self.radar_xy_cols[0]]
            y = pts[:, self.radar_xy_cols[1]]
            vx = pts[:, self.radar_vel_cols[0]]
            vy = pts[:, self.radar_vel_cols[1]]
            rn = (x ** 2 + y ** 2).sqrt().clamp(min=1e-3)
            losx, losy = x / rn, y / rn          # 점별 LOS(ego→점) 단위벡터
            col = (W / 2.0 - y / res_col).long()
            row = (H / 2.0 - x / res_row).long()
            wsum = torch.zeros(H * W, device=device)
            acc = torch.zeros(4, H * W, device=device)   # vx, vy, losx, losy 가중합
            for di, dj in offsets:
                rr, cc = row + di, col + dj
                valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                if valid.sum() == 0:
                    continue
                w = float(np.exp(-(di * di + dj * dj) / sigma2))
                flat = rr[valid] * W + cc[valid]
                ones_w = torch.full((int(valid.sum()),), w, device=device)
                wsum.index_add_(0, flat, ones_w)
                acc[0].index_add_(0, flat, w * vx[valid])
                acc[1].index_add_(0, flat, w * vy[valid])
                acc[2].index_add_(0, flat, w * losx[valid])
                acc[3].index_add_(0, flat, w * losy[valid])
            occ = wsum > 1e-6
            norm = wsum.clamp(min=1e-6)
            out[b, 0] = (acc[0] / norm).view(H, W).to(dtype)
            out[b, 1] = (acc[1] / norm).view(H, W).to(dtype)
            out[b, 2] = occ.float().view(H, W).to(dtype)
            out[b, 3] = (acc[2] / norm).view(H, W).to(dtype)
            out[b, 4] = (acc[3] / norm).view(H, W).to(dtype)
        return out

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # 메인 forward 흐름:
        #   batch(image/lidar2img/radar_points/[temporal]) 입력
        #   → ① camera/radar를 Gaussian으로 변환 → (③ 학습-only densify)
        #   → GS로 camera/radar BEV 렌더 → (① velocity/identity BEV splat)
        #   → fuser로 융합 → (① aug 잔차 주입, ② temporal warp 잔차 주입)
        #   → decoder → detection head (+ track head)
        #   → 반환: {"output": head 결과, gaussian 수, 중간 feature, aug BEV}

        # Convert pixels and points to Gaussians.
        camera_gaussians = self.forward_features_camera(batch)   # 이미지 → Gaussian dict
        radar_gaussians = self.forward_features_radar(batch)     # radar  → Gaussian dict

        # [GT-free 재설계] adaptive densification (3DGS식): opacity·크기 기반 clone/split.
        # GT 미사용 → train/test 동일하게 실행(추론 포함)해 이전 GT-leak(train/test mismatch) 해소.
        if self.densify_train:
            camera_gaussians = self._densify_gaussians(camera_gaussians)
            radar_gaussians = self._densify_gaussians(radar_gaussians)

        # Rasterize Gaussians to BEV features.
        # GS 핵심: camera Gaussian → BEV feature [B, C, H, W] (+ 평균 Gaussian 수).
        camera_bev, num_gaussians_cam = self.gs_render_image(
            camera_gaussians["features"],
            camera_gaussians["centers"],
            camera_gaussians["covariances"],
            camera_gaussians["opacities"]
        )
        # GS 핵심: radar Gaussian → 동일 BEV 그리드로 렌더.
        radar_bev, num_gaussians_radar = self.gs_render_radar(
            radar_gaussians["features"],
            radar_gaussians["centers"],
            radar_gaussians["covariances"],
            radar_gaussians["opacities"]
        )

        # Augmented Gaussian BEV (①): splat per-Gaussian velocity + identity. Camera and
        # radar contribute additively to a shared BEV velocity/identity field.
        # ① camera/radar 각각의 velocity·identity를 BEV로 splat 후, 둘을 더해 공유 필드 생성.
        aug_cam = self._render_augmented(camera_gaussians, self.gs_render_image)
        aug_radar = self._render_augmented(radar_gaussians, self.gs_render_radar)
        velocity_bev = identity_bev = None
        if aug_cam and aug_radar:                                  # 양쪽 모두 ① 속성을 낸 경우만
            velocity_bev = aug_cam["velocity_bev"] + aug_radar["velocity_bev"]  # BEV 속도 필드 합
            identity_bev = aug_cam["identity_bev"] + aug_radar["identity_bev"]  # BEV identity 필드 합

        # Fuse BEV features.
        # camera BEV + radar BEV → 융합 BEV(단일 또는 multi-scale 리스트).
        fused_bev = self.fuser(camera_bev, radar_bev)

        # Inject the Augmented-Gaussian velocity/identity fields into the full-res
        # fused BEV (residual, zero-init). Handles multi-scale fuser output by only
        # augmenting scale-0 and resizing the aug field to match.
        # ① 잔차 주입: velocity+identity BEV를 aug_proj(zero-init)로 변환해 융합 BEV에 더함.
        if velocity_bev is not None:
            aug = torch.cat([velocity_bev, identity_bev], dim=1)  # [B, 2+K, H, W]
            is_list = isinstance(fused_bev, (list, tuple))         # multi-scale 출력 여부
            f0 = fused_bev[0] if is_list else fused_bev            # scale-0(full-res)만 증강
            if aug.shape[-2:] != f0.shape[-2:]:                    # 해상도가 다르면 맞춰 보간
                aug = torch.nn.functional.interpolate(
                    aug, size=f0.shape[-2:], mode="bilinear", align_corners=False)
            f0 = f0 + self.aug_proj(aug)                           # 잔차 합(초기엔 0)
            if is_list:
                fused_bev = [f0, *list(fused_bev[1:])]             # scale-0만 교체
            else:
                fused_bev = f0

        # Geometric ego-warp (temporal): resample the prev-frame fused BEV into the
        # current frame using `bev_warp` (ego-motion ∘ augmentation, from the pipeline)
        # and residual-inject (zero-init proj). prev_bev_feat is populated by the module
        # from the no-grad prev-frame forward, so this adds temporal context without
        # back-propagating into the prev branch.
        # ② temporal ego-warp 잔차 주입: prev 융합 BEV를 현재 좌표계로 warp 후 더함.
        if self.temporal_warp:
            is_list = isinstance(fused_bev, (list, tuple))
            f0 = fused_bev[0] if is_list else fused_bev
            injected = False
            # [multi-frame 8/21] t-1 … t-K를 각각 현재 좌표계로 warp 후 잔차 주입.
            #   프레임마다 별도 zero-init proj(temporal_proj / temporal_proj{k})를 써서
            #   네트워크가 프레임별 기여도를 스스로 학습한다. K=1이면 기존과 완전히 동일.
            #   GS 인코딩은 프레임 단위라 이 경로와 무관 — core contribution 불변.
            for k in range(1, self.max_prev_frames + 1):
                pkey = "prev_bev_feat" if k == 1 else f"prev{k}_bev_feat"
                wkey = "bev_warp" if k == 1 else f"bev_warp{k}"
                proj = self.temporal_proj if k == 1 else getattr(self, f"temporal_proj{k}", None)
                prev_bev, bev_warp = batch.get(pkey), batch.get(wkey)
                if prev_bev is None or bev_warp is None or proj is None:
                    continue
                if prev_bev.shape[-2:] != f0.shape[-2:]:           # 해상도 정렬
                    prev_bev = torch.nn.functional.interpolate(
                        prev_bev, size=f0.shape[-2:], mode="bilinear", align_corners=False)
                warped = self._warp_prev_bev(prev_bev.to(f0.dtype), bev_warp)  # 현재 좌표계로 리샘플
                f0 = f0 + proj(warped)                            # 잔차 합(zero-init proj)
                injected = True
            if injected:
                fused_bev = [f0, *list(fused_bev[1:])] if is_list else f0

        # Decode the fused BEV features.
        # 융합 BEV → multi-scale BEV feature (DPTHead).
        decoded = self.decoder(fused_bev)

        # Query head: 학습 시 DN-DETR용 GT 전달 (head 내부에서 self.training 체크).
        # query 기반 head는 학습 중 denoising(DN)용 GT를 받아야 하므로 박스/라벨을 꺼낸다.
        gt_key_b = f"{self.head.key}_gt_boxes" if hasattr(self.head, 'key') else None
        gt_key_l = f"{self.head.key}_gt_labels" if hasattr(self.head, 'key') else None
        gt_boxes_list = batch.get(gt_key_b) if gt_key_b else None
        gt_labels_list = batch.get(gt_key_l) if gt_key_l else None
        if isinstance(gt_boxes_list, torch.Tensor):
            gt_boxes_list = [gt_boxes_list[i] for i in range(gt_boxes_list.shape[0])]  # 배치별 리스트화
        if isinstance(gt_labels_list, torch.Tensor):
            gt_labels_list = [gt_labels_list[i] for i in range(gt_labels_list.shape[0])]
        # [B③] radar 속도맵 (vx,vy,mask) — head vel_head 입력용
        radar_vel_map = None
        if self.radar_vel_fusion and batch.get("radar_points") is not None:
            f_ref = fused_bev[0] if isinstance(fused_bev, (list, tuple)) else fused_bev
            radar_vel_map = self._build_radar_vel_map(
                batch["radar_points"], f_ref.device, f_ref.dtype)

        if radar_vel_map is not None:
            # [B③] 융합 경로는 CenterHead 전용 — try/except 밖에서 호출해 내부 TypeError가
            #   fallback으로 삼켜져 radar 융합이 조용히 꺼지는 사고를 방지.
            out = self.head(decoded, gt_boxes_list, gt_labels_list, radar_vel_map=radar_vel_map)
        else:
            try:
                out = self.head(decoded, gt_boxes_list, gt_labels_list)   # GT 받는 head(query) 경로
            except TypeError:
                # head가 추가 인자를 안 받으면 fallback
                out = self.head(decoded)                                  # GT 안 받는 head(heatmap) 경로

        # Tracking head (track_embed 예측).
        # tracking 사용 시 per-instance embedding을 추가로 예측해 out에 병합.
        if self.use_tracking and self.track_head is not None:
            # ⑤ skip: decoder 이전의 융합 BEV(scale-0)를 track_head에 함께 전달.
            skip_feat = fused_bev[0] if isinstance(fused_bev, (list, tuple)) else fused_bev
            track_out = self.track_head(decoded, skip_feat)
            out = {**out, **track_out}

        key_prefix = getattr(self.head, "key", "vehicle") if self.head is not None else "vehicle"

        # Augmented Gaussian BEV maps for downstream losses/heads (②/④). Stored both on
        # `out` (loss access via outputs["output"]) and the top-level dict.
        # ① BEV 속도/identity 맵을 loss/후속 head가 쓸 수 있도록 out에도 기록.
        if velocity_bev is not None:
            out[f"{key_prefix}_velocity_bev"] = velocity_bev
            out[f"{key_prefix}_identity_bev"] = identity_bev

        # 최종 반환: head 출력 + 로깅용 Gaussian 수 + 중간 BEV feature + aug BEV.
        return {
            "output": out,                            # loss/metric이 보는 head 결과
            "num_gaussians_cam": num_gaussians_cam,   # 로깅용
            "num_gaussians_radar": num_gaussians_radar,
            "features_cam": camera_bev,               # 카메라 BEV(디버그/분석)
            "features_radar": radar_bev,              # radar BEV
            "features_fused": fused_bev,              # 융합 BEV
            "velocity_bev": velocity_bev,             # ① BEV 속도 필드
            "identity_bev": identity_bev,             # ① BEV identity 필드
        }
