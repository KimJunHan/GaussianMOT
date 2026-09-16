import math
import torch
import torch.nn as nn
# diff_gaussian_rasterization: 3D Gaussian Splatting CUDA 래스터라이저.
# GaussianMOT의 핵심 "universal view transformer" — Gaussian들을 BEV 평면으로 splat 한다.
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class BEVCamera:
    # BEV(Bird's-Eye-View) 평면을 정사영(orthographic)으로 내려다보는 가상 카메라.
    # 일반 GS는 원근(perspective) 카메라를 쓰지만, 여기서는 위에서 수직으로 내려다보는
    # 정사영 카메라를 흉내내서 3D Gaussian들을 BEV 그리드(예: 200x200)로 투영한다.
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), image_size=(200, 200)):
        # Orthographic projection parameters
        # BEV가 커버하는 metric 범위(미터)와 출력 이미지 크기(픽셀).
        self.x_min, self.x_max = x_range          # x 축(전방-후방) metric 범위
        self.y_min, self.y_max = y_range          # y 축(좌-우) metric 범위
        self.image_width = image_size[1]          # 출력 BEV 가로 픽셀 수
        self.image_height = image_size[0]         # 출력 BEV 세로 픽셀 수

        # Set up FoV to cover the range [-50, 50] for both X and Y
        # 정사영이므로 FoV는 각도가 아니라 장면의 metric 폭/높이로 둔다(아래 tan에 그대로 들어감).
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        # 카메라 중심을 원점에 두고 Z축을 따라 아래를 내려다보는 BEV 시점.
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)  # High above Z-axis

        # Orthographic projection matrix for BEV
        # 정사영 변환 행렬(world→view, view→이미지) 초기화.
        self.set_transform()

    def set_transform(self, h=200, w=200, h_meters=100, w_meters=100):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        # metric 1m 당 픽셀 수(스케일): 픽셀크기 / metric크기.
        sh = h / h_meters                          # 세로 방향 스케일 (픽셀/미터)
        sw = w / w_meters                          # 가로 방향 스케일 (픽셀/미터)
        # world_view_transform: world 좌표 → view(카메라) 좌표.
        # x/y 축을 서로 맞바꿔(swap) BEV 좌표축 규약에 맞추고 스케일을 적용한다.
        self.world_view_transform = torch.tensor([
            [ 0.,  sh,  0.,         0.],
            [ sw,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        # full_proj_transform: world → 이미지(픽셀) 좌표(부호 반전 + h/2, w/2 평행이동으로 중앙 정렬).
        self.full_proj_transform = torch.tensor([
            [ 0., -sh,  0.,          h/2.],
            [-sw,   0.,  0.,         w/2.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        # 렌더 해상도(출력 BEV 픽셀 크기)를 갱신.
        self.image_height = h
        self.image_width = w


class GaussianRenderer(nn.Module):
    # 3D Gaussian들을 BEV 평면으로 splat(래스터화)해서 BEV feature map을 만드는 모듈.
    # 입력: 각 Gaussian의 (feature, 중심, 공분산, opacity) → 출력: [B, C, H, W] BEV feature.
    # camera 경로와 radar 경로가 각각 독립 인스턴스로 같은 BEV 그리드에 렌더한다.
    def __init__(
        self,
        embed_dims,           # 각 Gaussian feature 채널 수(= BEV feature 채널 수)
        threshold=0.05,       # opacity 필터 임계값: 이보다 낮은 Gaussian은 렌더에서 제외
        x_min: float = -50,
        x_max: float = 50,
        y_min: float = -50,
        y_max: float = 50,
        bev_h: int = 200,
        bev_w: int = 200,
    ):
        super().__init__()
        # 위에서 내려다보는 정사영 BEV 카메라 생성.
        self.viewpoint_camera = BEVCamera(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            image_size=(bev_h, bev_w),
        )
        # 실제 splat을 수행하는 CUDA 래스터라이저(설정은 forward 직전에 주입).
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.threshold = threshold      # opacity 필터 임계값 저장
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, features, means3D, cov3D, opacities):
        """
        features: b G d
        means3D: b G 3
        uncertainty: b G 6
        opacities: b G 1
        """
        # features: [b, G, d] / means3D: [b, G, 3] / cov3D: [b, G, 6] / opacities: [b, G, 1]
        # b=batch, G=Gaussian 개수, d=feature 채널.
        #
        # [실제 런타임 값 (현 config)]
        #   - b = 1 (학습, batch_size=1) / 2 (평가, evaluate batch_size=2)
        #   - d = embed_dims = 128
        #   - G (Gaussian 수)는 이 렌더러를 부르는 경로에 따라 다름:
        #       * camera 경로: G = N_cam(6) × H_f × W_f = 6 × 56 × 100 = 33,600
        #           (ResNet-50 stride-8 feature; 입력 448×800 → 56×100, AGPNeck로 정렬, per-pixel Gaussian)
        #       * radar 경로:  G = max_points = 3500 (PointsToGaussians 패딩, 빈 슬롯은 opacity=0)
        #     → opacity 필터(mask) 통과분만 실제 splat되므로 유효 G는 이보다 작다.
        b = features.shape[0]
        device = means3D.device

        bev_out = []
        # opacity 필터: threshold 초과인 Gaussian만 렌더 대상으로 남긴다(노이즈/빈 슬롯 제거).
        mask = (opacities > self.threshold)
        mask = mask.squeeze(-1)               # [b, G, 1] -> [b, G] (Gaussian별 boolean 마스크)
        # 현재 해상도에 맞춰 정사영 행렬과 래스터 설정을 갱신.
        self.set_render_scale(self.bev_h, self.bev_w)
        self.set_Rasterizer(device)
        # 래스터라이저는 batch를 한 번에 못 받으므로 배치 원소별로 순회하며 splat.
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i][mask[i]],          # 마스크 통과한 Gaussian 중심만 전달
                means2D=None,
                shs=None,  # No SHs used               # 구면조화(SH) 색상 미사용 — feature를 직접 splat
                colors_precomp=features[i][mask[i]],  # "색상" 자리에 feature 벡터를 넣어 BEV feature 생성
                opacities=opacities[i][mask[i]],      # 가중 합성(alpha)을 위한 opacity
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i][mask[i]]        # scale/rotation 대신 미리 계산한 3D 공분산 직접 전달
            )
            bev_out.append(rendered_bev)              # [d, h, w] 누적

        x = torch.stack(bev_out, dim=0) # b d h w     # 배치별 결과를 다시 [b, d, h, w]로 스택
        # 실제 출력 shape: [b, 128, 200, 200] (d=embed_dims=128, BEV 200×200)
        # 로깅/모니터링용: 배치 평균 유효 Gaussian 수(역전파에는 영향 X).
        num_gaussians = (mask.detach().float().sum(1)).mean().cpu()

        return x, num_gaussians                       # BEV feature map과 평균 Gaussian 수 반환

    @torch.no_grad()
    def set_Rasterizer(self, device):
        # FoV(여기선 metric 폭/높이)로부터 정사영용 tan 값 계산.
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        # 배경색 = 0 벡터(embed_dims 길이): 어떤 Gaussian도 덮지 않은 BEV 셀의 기본값.
        bg_color = torch.zeros((self.embed_dims)).to(device) # self.embed_dims
        # bg_color[-1] = -4
        # 래스터라이저 동작에 필요한 카메라/투영 설정 묶음 구성.
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),   # world→view 정사영 행렬
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),    # world→이미지 투영 행렬
            sh_degree=0,  # No SHs used                                          # SH 색상 미사용
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        # 위 설정을 CUDA 래스터라이저에 주입.
        self.rasterizer.set_raster_settings(raster_settings)

    @torch.no_grad()
    def set_render_scale(self, h, w):
        # 출력 해상도 변경 시 카메라 크기와 정사영 행렬을 함께 재계산.
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w)
