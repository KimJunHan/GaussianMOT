"""CenterPoint-style dense detection head (CR3DT-inspired).

[파이프라인에서의 역할]
DetBEVEncoder가 만든 BEV feature를 받아 3D 객체를 "검출"하는 dense head.
BEV 격자의 각 셀(cell)마다 (1) 클래스별 heatmap(중심 확률)과 (2) 박스 회귀값
(offset/height/dim/rotation/velocity)을 예측한다. inference 시 heatmap에서
top-K 피크(peak)를 뽑아 그 위치의 회귀값으로 3D 박스를 디코딩한다.
추가로 검출마다 추적용 임베딩(track_embed)도 함께 낸다.

Simpler than TransFusionHead: no transformer decoder, no DETR queries.
Just dense per-cell heatmap + regression heads, top-K peak extraction at
inference. (트랜스포머 디코더/DETR 쿼리 없이, dense head + top-K 피크 추출로 단순)

Output keys (back-compatible with downstream BBoxDecoder/loss):
  - {key}_pred_logits:    [1, B, K, num_classes]   (single "layer" axis)
  - {key}_pred_boxes:     [1, B, K, 10] = (cx, cy, cz, w, l, h, sin, cos, vx, vy)
  - {key}_dense_heatmap:  [B, num_classes, H, W]
  - {key}_query_track_embed: [B, K, D] L2-normed per-detection embedding
  - {key}_anchor_cell, {key}_proposal_class: kept for API compat
"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SeparateHead(nn.Module):
    """A small 3x3 conv stack for one task head.

    하나의 task(heatmap/offset/dim 등)를 담당하는 작은 conv 헤드:
    3x3 conv → BN → ReLU → 3x3 conv(out_channels). 각 task가 독립 헤드를 가짐.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 64, init_bias: float = 0.0):
        super().__init__()
        # [①] 크기 사전: 학습에서 타깃에서 뺀 값을 디코드에서 되돌린다. 둘이 어긋나면
        #   크기 예측이 조용히 무너지므로 losses.CenterPointLoss와 동일 값을 써야 한다.
        if size_prior is not None:
            import torch as _t
            self.register_buffer("size_prior", _t.tensor([list(r) for r in size_prior], dtype=_t.float32))
        else:
            self.size_prior = None
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )
        # init_bias 지정 시 마지막 conv의 bias를 초기화(heatmap focal-prior 초기화에 사용)
        if init_bias != 0.0:
            nn.init.constant_(self.layers[-1].bias, init_bias)

    def forward(self, x):
        return self.layers(x)


class CenterHead(nn.Module):
    """CR3DT/CenterPoint-style dense head.

    Args:
        in_channels: BEV feature channel (from DetBEVEncoder, 256).
        hidden_channel: per-task hidden width.
        num_classes: detection classes (10 for nuScenes).
        num_proposals: at inference, top-K peaks across the heatmap.
        nms_kernel_size: max-pool size for peak NMS.
        nms_rescale_factors: per-class scale multipliers for NMS radius (CR3DT Scale-NMS).
        track_embed_dim: per-detection track embedding dim.
        bev_h, bev_w, x_min, x_max, y_min, y_max: BEV geometry.
        ped_cone_classes: classes with 1x1 NMS (small/dense objects).
        key: prefix for output dict keys.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channel: int = 128,
        num_classes: int = 10,
        num_proposals: int = 300,
        nms_kernel_size: int = 3,
        nms_rescale_factors: Optional[Sequence[float]] = (1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5, 3.5),
        track_embed_dim: int = 64,
        bev_h: int = 200,
        bev_w: int = 200,
        x_min: float = -50.0,
        x_max: float = 50.0,
        y_min: float = -50.0,
        y_max: float = 50.0,
        key: str = "vehicle",
        ped_cone_classes: Sequence[int] = (5, 8),
        # [B②] SECOND식 direction-bin 분류 head. yaw 180° flip(ped/moto/bicycle mAOE>1.1) 억제:
        #   sin/cos 회귀와 별개로 "cos(yaw)>0인가"를 2-class 분류하고, 추론 시 회귀 yaw의
        #   부호가 분류 결과와 어긋나면 π를 더해 뒤집는다.
        use_dir_classifier: bool = False,
        # [A-yaw] MultiBin yaw 파라미터화 (8/14 설계). sin/cos L1 회귀는 앞/뒤 다봉 분포에서
        #   모드 평균으로 붕괴(180° flip의 근본 원인) → 4-bin 분류(모드 선택) + bin별 (sin,cos)
        #   잔차 회귀(±45° 단봉)로 교체. 실패한 dir-bin(독립 회귀를 사후 뒤집기)과 달리
        #   파라미터화 자체를 바꾸므로 두 head가 싸우는 실패 모드가 없음.
        use_multibin_yaw: bool = False,
        # [B③] 박스레벨 radar doppler 융합. radar 보정속도(vx,vy,mask 3채널 BEV 맵)를
        #   vel_head 입력에 concat — 속도를 회귀만으로 배우지 않고 측정값을 직접 참조 (CR3DT 방식).
        use_radar_vel_fusion: bool = False,
        # [V-recon 8/19] Doppler→전체속도 해석적 재구성 층. 진단(8/19): 이동객체가 이미
        #   gradient의 92%를 받는데도 속도 과소예측(car/bus 비율 0.83, moto 0.67, ped 0.41)
        #   → 가중치 문제가 아니라 conv가 v=(s/ĥ·r̂)ĥ 비선형 연산을 학습 못하는 것.
        #   해법: 셀마다 해석적으로 계산해 vel_head 입력에 주입(학습 불필요, 예측 yaw 사용).
        #   오프라인 검증: GT-yaw면 이동객체 오차 1.87→1.09. 현재 car yaw 오차 ~10°로 근접.
        use_radar_vel_recon: bool = False,
        recon_min_cos: float = 0.3,       # |ĥ·r̂| 하한 (측면 통과 시 재구성 발산 방지)
        recon_max_speed: float = 30.0,    # 재구성 속도 크기 상한 [m/s]
        # [T-vel 8/20] temporal displacement → 속도. track_offset_head가 이미 회귀하는
        #   "현재→이전 프레임 변위[m]"를 v = -offset/dt 로 바꿔 vel_head 입력에 주입.
        #   A/B 진단(8/20): radar 재구성은 yaw 의존이라 mAOE 0.86(49°)에서 효과 2.8%뿐.
        #   temporal은 **yaw·radar 가시성 모두 무관** → 보행자(radar 안 잡힘)까지 커버.
        #   RCTrans가 4프레임 temporal fusion으로 mAVE 0.198을 얻는 것과 동일 원리.
        #   track_offset은 loss로만 쓰이고 추론엔 미사용이던 자산(검증: 정지 0.06m,
        #   이동 |offset|≈|v|×0.5s, 방향 -v와 2°).
        use_temporal_vel: bool = False,
        temporal_dt: float = 0.5,         # 키프레임 간격 [s]
        # extra args ignored for backward-compat with TransFusionHead config
        num_decoder_layers: int = None,
        num_heads: int = None,
        ffn: int = None,
        dropout: float = None,
        gaussian_overlap: float = None,
        min_radius: int = None,
        match_cls_weight: float = None,
        match_box_weight: float = None,
        focal_alpha: float = None,
        focal_gamma: float = None,
        grad_checkpoint: bool = None,
        # [①] 클래스별 log(w,l,h) 사전. CenterPointLoss.size_prior와 동일해야 한다.
        size_prior=None,
        **_ignored,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.ped_cone_classes = tuple(ped_cone_classes)   # 1x1 NMS를 쓸 작고 밀집한 클래스들
        self.track_embed_dim = track_embed_dim
        self.bev_h, self.bev_w = bev_h, bev_w
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        # BEV grid follows the renderer/`view` convention (common.py:get_view_matrix,
        # render.py:full_proj_transform): col = W/2 - y/res_col, row = H/2 - x/res_row.
        # i.e. lidar +x → row (negated), lidar +y → col (negated). res = meters per cell.
        # BEV 격자 좌표 규약(렌더러와 동일): row는 -x, col은 -y. res_*는 셀 1칸당 미터.
        self.res_row = (x_max - x_min) / bev_h   # 행(row) 방향 해상도(미터/셀)
        self.res_col = (y_max - y_min) / bev_w   # 열(col) 방향 해상도(미터/셀)
        self.key = key

        # 클래스별 NMS 반경 배율을 buffer로 등록(없으면 전부 1.0)
        if nms_rescale_factors is None:
            self.register_buffer("nms_rescale_factors", torch.ones(num_classes))
        else:
            self.register_buffer("nms_rescale_factors", torch.tensor(list(nms_rescale_factors), dtype=torch.float32))

        # Shared conv
        # 모든 task 헤드가 공유하는 공통 conv (BEV feature → hidden_channel)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
        )

        # Heatmap head with focal-prior bias init (p=0.01)
        # heatmap 헤드: 마지막 bias를 -log((1-p)/p)로 초기화 → 초기 sigmoid≈0.01(focal loss 안정화)
        prior_prob = 0.01
        bias_init = float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        self.heatmap_head = _SeparateHead(hidden_channel, num_classes, hidden=hidden_channel, init_bias=bias_init)

        # Regression heads (per cell) — 셀마다 박스 회귀값 예측
        self.center_head = _SeparateHead(hidden_channel, 2, hidden=hidden_channel)   # offset wrt cell center (셀 중심 대비 Δcol,Δrow)
        self.height_head = _SeparateHead(hidden_channel, 1, hidden=hidden_channel)   # cz (real m) (중심 높이, 실제 미터)
        self.dim_head    = _SeparateHead(hidden_channel, 3, hidden=hidden_channel)   # log(w, l, h) (박스 크기의 로그값)
        # [A-yaw] MultiBin: 4 bin logits + 4×(sin,cos) 잔차 = 12ch. 기존: (sin,cos) 2ch.
        self.use_multibin_yaw = use_multibin_yaw
        rot_out = 12 if use_multibin_yaw else 2
        self.rot_head    = _SeparateHead(hidden_channel, rot_out, hidden=hidden_channel)   # yaw 회전
        # bin 중심 [0, π/2, π, 3π/2] — decode에서 bin센터+잔차로 yaw 복원
        self.register_buffer("yaw_bin_centers",
                             torch.tensor([0.0, 1.5707963, 3.1415927, -1.5707963]))
        # [A recipe] radar 융합 시 vel_head 입력 = shared feature + radar 맵 5채널
        #   (vx, vy, occ, losx, losy) — LOS 채널로 radial→full 재구성 학습 가능 (8/14 진단 확정)
        self.use_radar_vel_fusion = use_radar_vel_fusion
        # [V-recon] 재구성 3채널(v_recon_x, v_recon_y, valid) 추가
        self.use_radar_vel_recon = use_radar_vel_recon and use_radar_vel_fusion
        self.recon_min_cos = float(recon_min_cos)
        self.recon_max_speed = float(recon_max_speed)
        # [T-vel] temporal 속도 2채널(vx, vy) 추가
        self.use_temporal_vel = use_temporal_vel
        self.temporal_dt = float(temporal_dt)
        vel_in = (hidden_channel + (5 if use_radar_vel_fusion else 0)
                  + (3 if self.use_radar_vel_recon else 0)
                  + (2 if use_temporal_vel else 0))
        self.vel_head    = _SeparateHead(vel_in, 2, hidden=hidden_channel)   # vx, vy (속도)
        # [B②] direction-bin 분류 head (2-class: cos(yaw)>0 여부)
        self.use_dir_classifier = use_dir_classifier
        if use_dir_classifier:
            self.dir_head = _SeparateHead(hidden_channel, 2, hidden=hidden_channel)
        # [AMOTP 리팩토링 ①] CenterTrack식 temporal offset head: 현재 중심 → 이전 프레임 같은 객체
        #   중심까지의 변위(dx,dy) [BEV 셀 단위] 회귀. temporal_warp로 trunk에 prev 컨텍스트가 주입돼
        #   있어 sf만으로 예측 가능. 추론 시 예측 offset으로 위치 기반 association → AMOTP 직결.
        self.track_offset_head = _SeparateHead(hidden_channel, 2, hidden=hidden_channel)  # (dx, dy) prev까지 변위

        # Per-detection track embedding head
        # 검출마다 추적용 임베딩을 만드는 MLP (shared feature → track_embed_dim)
        self.track_embed_head = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, track_embed_dim),
        )

    def _gather_at(self, feat_2d: torch.Tensor, idx_flat: torch.Tensor) -> torch.Tensor:
        """feat_2d: [B, C, H, W], idx_flat: [B, K] in [0, H*W) → [B, K, C].

        2D feature map을 평탄화(flatten)한 뒤, top-K 셀 인덱스 위치의 채널 벡터만 모음.
        """
        B, C, H, W = feat_2d.shape
        feat_flat = feat_2d.reshape(B, C, -1)  # [B, C, HW] (H,W를 펼침)
        # reshape: channels_last 입력은 view가 요구하는 연속 stride를 만족하지 않는다.
        idx = idx_flat.unsqueeze(1).expand(-1, C, -1)  # [B, C, K] (모든 채널에 같은 인덱스 적용)
        return feat_flat.gather(dim=2, index=idx).transpose(1, 2)  # [B, K, C]

    def _dense_yaw_bev(self, rot: torch.Tensor) -> torch.Tensor:
        """[V-recon] dense rot 출력 → 셀별 BEV yaw [B,H,W]. MultiBin/legacy 공통."""
        if self.use_multibin_yaw:
            B, _, H, W = rot.shape
            bin_idx = rot[:, :4].argmax(dim=1)                          # [B,H,W]
            res = rot[:, 4:].reshape(B, 4, 2, H, W)
            idx = bin_idx[:, None, None].expand(-1, 1, 2, -1, -1)
            rs = res.gather(1, idx).squeeze(1)                          # [B,2,H,W]
            return self.yaw_bin_centers[bin_idx] + torch.atan2(rs[:, 0], rs[:, 1])
        return torch.atan2(rot[:, 0], rot[:, 1])

    def _reconstruct_velocity(self, rot: torch.Tensor, radar_vel_map: torch.Tensor) -> torch.Tensor:
        """[V-recon] Doppler radial 속도 + 예측 heading → 전체속도 해석적 재구성.

        radar는 시선(LOS) 방향 성분 s = v·r̂ 만 측정하므로, 물체가 heading ĥ 방향으로만
        움직인다는 가정(차량 비홀로노믹 운동) 하에 v = (s / ĥ·r̂) ĥ 로 전체속도가 복원된다.
        |ĥ·r̂|가 작으면(측면 통과) 발산하므로 recon_min_cos로 게이팅하고 valid 채널로 알린다.
        yaw는 detach — 속도 오차가 yaw를 왜곡하지 않도록(재구성은 입력 feature 역할).

        반환: [B, 3, H, W] = (v_recon_x, v_recon_y, valid)
        """
        yaw_bev = self._dense_yaw_bev(rot).detach()
        # BEV yaw → ego 프레임 heading (metrics.bev_yaw_to_lidar_yaw와 동일 규약)
        yaw_ego = -yaw_bev - 1.5707963
        hx, hy = yaw_ego.cos(), yaw_ego.sin()
        vx, vy, occ, lx, ly = (radar_vel_map[:, 0], radar_vel_map[:, 1],
                               radar_vel_map[:, 2], radar_vel_map[:, 3], radar_vel_map[:, 4])
        s = vx * lx + vy * ly                     # 측정된 radial 속도(부호 포함)
        hd = hx * lx + hy * ly                    # ĥ·r̂
        valid = (occ > 0.5) & (hd.abs() > self.recon_min_cos)
        hd_safe = torch.where(valid, hd, torch.ones_like(hd))
        speed = (s / hd_safe).clamp(-self.recon_max_speed, self.recon_max_speed)
        speed = torch.where(valid, speed, torch.zeros_like(speed))
        return torch.stack([speed * hx, speed * hy, valid.to(rot.dtype)], dim=1)

    def forward(self, x, gt_boxes_list=None, gt_labels_list=None, radar_vel_map=None) -> Dict[str, torch.Tensor]:
        # x: BEV feature [B, in_channels, H, W]
        B, _, H, W = x.shape
        # BEV 해상도가 설정값과 일치하는지 확인(좌표 디코딩이 격자 크기에 의존)
        assert H == self.bev_h and W == self.bev_w, f"BEV mismatch {H}x{W} vs {self.bev_h}x{self.bev_w}"
        device, dtype = x.device, x.dtype

        # Shared feature
        sf = self.shared_conv(x)  # [B, C, H, W] 모든 헤드가 공유하는 feature

        # Dense predictions — 셀마다 dense하게 예측
        heatmap = self.heatmap_head(sf)  # [B, num_classes, H, W] 클래스별 중심 확률(logit)
        offset  = self.center_head(sf)   # [B, 2, H, W] 셀 중심 대비 위치 보정(Δcol, Δrow)
        height  = self.height_head(sf)   # [B, 1, H, W] 중심 높이 cz
        dim_log = self.dim_head(sf)      # [B, 3, H, W] log(w,l,h)
        rot     = self.rot_head(sf)      # [B, 2, H, W] (sin, cos)
        # [T-vel] track_offset을 vel_head보다 먼저 계산(속도 입력으로 사용하기 위해)
        track_offset = self.track_offset_head(sf)  # [B, 2, H, W] (dx, dy) 현재→이전 프레임 변위 [m]
        # [A recipe] radar 맵(5ch)을 vel_head 입력에 concat. 맵이 없으면 0으로 채워
        #   fusion off와 동일하게 동작 — 어떤 경우에도 shape/gradient 경로는 일정하게 유지.
        vel_inputs = [sf]
        if self.use_radar_vel_fusion:
            if radar_vel_map is None:
                radar_vel_map = sf.new_zeros(B, 5, H, W)
            radar_vel_map = radar_vel_map.to(sf.dtype)
            vel_inputs.append(radar_vel_map)
            if self.use_radar_vel_recon:
                vel_inputs.append(self._reconstruct_velocity(rot, radar_vel_map))  # [B,3,H,W]
        if self.use_temporal_vel:
            # offset = (이전 위치 − 현재 위치)[m] → v = −offset/dt. detach: 속도 오차가
            #   offset head(추적용)를 왜곡하지 않도록 입력 feature로만 사용.
            vel_inputs.append(-track_offset.detach() / self.temporal_dt)   # [B,2,H,W]
        vel = self.vel_head(torch.cat(vel_inputs, dim=1) if len(vel_inputs) > 1 else sf)  # [B,2,H,W]
        dir_logits = self.dir_head(sf) if self.use_dir_classifier else None  # [B, 2, H, W] dir-bin

        # Peak extraction (Scale-NMS): max-pool per class with class-specific kernel.
        # 피크 추출(Scale-NMS): 클래스마다 다른 커널 크기로 max-pool하여 국소 최대만 남김
        hm_sig = heatmap.detach().sigmoid()  # [B, C, H, W] 확률화(역전파 차단: 피크 선택은 미분 불필요)
        pooled = torch.zeros_like(hm_sig)
        for c in range(self.num_classes):
            rs = float(self.nms_rescale_factors[c].item())            # 클래스별 NMS 반경 배율
            ksize = max(1, int(round(self.nms_kernel_size * rs)))     # 배율 적용한 커널 크기(최소 1)
            if c in self.ped_cone_classes:
                ksize = 1                                             # 작고 밀집한 클래스는 1x1(NMS 거의 안 함)
            if ksize == 1:
                pooled[:, c] = hm_sig[:, c]                          # 커널 1이면 그대로 복사
            else:
                if ksize % 2 == 0:
                    ksize += 1                                       # max_pool 패딩 위해 홀수 보장
                pooled[:, c:c+1] = F.max_pool2d(
                    hm_sig[:, c:c+1], kernel_size=ksize, stride=1, padding=ksize // 2
                )                                                    # 같은 해상도 max-pool(국소 최대값 맵)
        # 원래 값 == 풀링 최대값인 셀만 피크(국소 최대)로 표시
        peak_mask = (hm_sig == pooled).float()
        hm_peaks = hm_sig * peak_mask  # [B, num_classes, H, W] 피크 위치만 점수 유지, 나머지 0
        flat = hm_peaks.reshape(B, -1)  # [B, num_classes * H * W] 전체를 1차원으로 펼침
        K = self.num_proposals
        # 전체에서 점수 상위 K개 선택 → top-K 검출 후보
        topk_v, topk_i = flat.topk(K, dim=-1)
        proposal_class = topk_i // (H * W)       # [B, K] 평탄 인덱스에서 클래스 분리
        proposal_index = topk_i % (H * W)        # [B, K] 평탄 인덱스에서 (row,col) 셀 위치 분리

        # Gather per-cell predictions at top-K spatial indices
        # logits: gather full num_classes vector at each (y, x) cell
        # → For loss compatibility (heatmap-based), expose per-class score directly.
        # cls "logits" for downstream BBoxDecoder: we re-create a (K, num_classes) tensor
        # where the predicted class column has the top score and others are very low.
        # 다운스트림 BBoxDecoder/loss가 (K, num_classes) logit을 기대하므로 재구성한다.
        # 기본값 -10(sigmoid≈0)으로 채우고, 선택된 클래스 칸만 점수의 inverse-sigmoid로 채움.
        cls_logits = torch.zeros(B, K, self.num_classes, device=device, dtype=dtype) - 10.0  # sigmoid ≈ 0
        # Set the chosen class to inverse-sigmoid(score) so downstream gets sigmoid → score back.
        # inverse-sigmoid(logit) = log(p/(1-p)) → 뒤에서 sigmoid 하면 원래 점수 복원됨
        eps = 1e-6
        logit_scores = torch.log(topk_v.clamp(min=eps) / (1.0 - topk_v.clamp(max=1 - eps)))
        # 각 후보의 예측 클래스 위치에만 점수 logit을 기록(나머지는 -10 유지)
        cls_logits.scatter_(2, proposal_class.unsqueeze(-1), logit_scores.unsqueeze(-1))

        # Gather regression at peaks — 피크 셀 위치의 회귀값/feature만 모음
        peak_off = self._gather_at(offset, proposal_index)    # [B, K, 2] 위치 보정
        peak_h   = self._gather_at(height, proposal_index)    # [B, K, 1] 높이 cz
        peak_dim = self._gather_at(dim_log, proposal_index)   # [B, K, 3] log 크기
        if self.size_prior is not None:
            # [①] 학습 타깃에서 뺀 클래스별 사전을 되돌려 절대 log 크기로 복원한다.
            peak_dim = peak_dim + self.size_prior.to(peak_dim.device, peak_dim.dtype)[proposal_class]
        peak_rot = self._gather_at(rot, proposal_index)       # [B, K, 2] (sin, cos)
        peak_vel = self._gather_at(vel, proposal_index)       # [B, K, 2] (vx, vy)
        shared_feat = self._gather_at(sf, proposal_index)     # [B, K, C] track 임베딩용 feature

        # Decode boxes (cell -> world), renderer/`view` convention.
        # offset channels are (Δcol, Δrow); cell indices: col = idx % W, row = idx // W.
        # 셀 좌표 → 실세계 좌표 디코딩. 평탄 인덱스에서 col=idx%W, row=idx//W, +0.5는 셀 중심.
        col = (proposal_index % W).to(dtype) + 0.5 + peak_off[..., 0]   # 보정 적용한 실수 col
        row = (proposal_index // W).to(dtype) + 0.5 + peak_off[..., 1]  # 보정 적용한 실수 row
        cx = (self.bev_h / 2.0 - row) * self.res_row   # row = H/2 - x/res_row  → x로 역변환
        cy = (self.bev_w / 2.0 - col) * self.res_col   # col = W/2 - y/res_col  → y로 역변환
        # 보정 전 셀 중심(col,row) — API 호환용 anchor 정보
        anchor_cell = torch.stack([
            (proposal_index % W).to(dtype) + 0.5,
            (proposal_index // W).to(dtype) + 0.5,
        ], dim=-1)  # [B, K, 2] (col, row)
        cz = peak_h[..., 0]              # 중심 높이
        w = peak_dim[..., 0].exp()       # log → 실제 폭(항상 양수)
        l = peak_dim[..., 1].exp()       # log → 실제 길이
        hh = peak_dim[..., 2].exp()      # log → 실제 높이
        if self.use_multibin_yaw:
            # [A-yaw] MultiBin decode: argmax bin 중심 + 해당 bin의 (sin,cos) 잔차 각
            bin_logits = peak_rot[..., :4]                        # [B, K, 4]
            bin_idx = bin_logits.argmax(dim=-1)                   # [B, K]
            res = peak_rot[..., 4:].reshape(B, K, 4, 2)           # [B, K, 4, 2]
            rs = res.gather(2, bin_idx[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)  # [B,K,2]
            yaw = self.yaw_bin_centers[bin_idx] + torch.atan2(rs[..., 0], rs[..., 1])
            sin_yaw, cos_yaw = yaw.sin(), yaw.cos()
        else:
            sin_yaw, cos_yaw = peak_rot[..., 0], peak_rot[..., 1]   # yaw 회전 성분
        # [B②] dir-bin 보정: 분류가 말하는 cos 부호(bin1 ⇔ cos>0)와 회귀 cos 부호가
        #   어긋나면 yaw에 π를 더한 것과 동일하게 (sin,cos)를 동시 반전.
        if self.use_dir_classifier:
            peak_dir = self._gather_at(dir_logits, proposal_index)        # [B, K, 2]
            dir_pos = peak_dir[..., 1] > peak_dir[..., 0]                 # True ⇔ 분류: cos>0
            flip = dir_pos != (cos_yaw > 0)                               # 부호 불일치 → 반전
            sign = torch.where(flip, -torch.ones_like(cos_yaw), torch.ones_like(cos_yaw))
            sin_yaw = sin_yaw * sign
            cos_yaw = cos_yaw * sign
        vx, vy = peak_vel[..., 0], peak_vel[..., 1]             # 속도 성분
        # 최종 박스 10D: (cx, cy, cz, w, l, h, sin, cos, vx, vy)
        decoded = torch.stack([cx, cy, cz, w, l, hh, sin_yaw, cos_yaw, vx, vy], dim=-1)  # [B, K, 10]

        # Per-detection track embedding — 검출별 추적 임베딩(L2 정규화)
        track_embed = self.track_embed_head(shared_feat)
        track_embed = F.normalize(track_embed, dim=-1)   # cosine 유사도 association용 단위벡터화

        # Shape into [L=1, B, K, ...] for downstream compat (loss expects L axis)
        # 다운스트림 loss가 레이어(L) 축을 기대하므로 맨 앞에 L=1 축 추가
        logits_out = cls_logits.unsqueeze(0)
        boxes_out  = decoded.unsqueeze(0)

        # 출력 dict: top-K 검출(logits/boxes) + dense 예측들 + 추적 임베딩 + API 호환 키
        return {
            f"{self.key}_pred_logits": logits_out,
            f"{self.key}_pred_boxes": boxes_out,
            f"{self.key}_dense_heatmap": heatmap,
            f"{self.key}_dense_offset": offset,
            f"{self.key}_dense_height": height,
            f"{self.key}_dense_dim_log": dim_log,
            f"{self.key}_dense_rot": rot,
            f"{self.key}_dense_vel": vel,
            f"{self.key}_dense_track_offset": track_offset,   # [B,2,H,W] CenterTrack offset (prev까지 변위, 셀 단위)
            **({f"{self.key}_dense_dir": dir_logits} if dir_logits is not None else {}),  # [B,2,H,W] [B②] dir-bin logits
            f"{self.key}_proposal_class": proposal_class,
            f"{self.key}_anchor_cell": anchor_cell,
            f"{self.key}_query_track_embed": track_embed,
        }
