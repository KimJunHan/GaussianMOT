import torch
import torch.nn as nn
import torch.nn.functional as F

# PointTransformerV3(PTv3): sparse 포인트클라우드용 트랜스포머 백본
from gaussianmot.ops.ptv3.model import PointTransformerV3


class PointsToGaussians(nn.Module):
    """
    [파이프라인에서의 역할]
    레이더 포인트클라우드를 입력받아 "포인트마다 하나의 3D Gaussian"의 파라미터를
    예측하는 모듈. 카메라 분기(PixelsToGaussians)와 짝을 이루는 레이더 분기로,
    출력 Gaussian이 GS 렌더러를 거쳐 레이더 BEV feature가 된다.

    동작 흐름:
      1) 가변 길이 포인트 리스트를 하나로 concat하고 PTv3가 요구하는 dict(feat/coord/offset)로 구성
      2) PTv3 encoder로 포인트별 feature 추출
      3) 여러 MLP head로 Gaussian 속성 분기:
           - means(centers): PTv3 좌표 + offset_mlp 예측 보정
           - covariances:    covs_mlp → make_valid_covariances로 양의 정부호 보장
           - opacities:      sigmoid (불투명도)
           - features:       feats_mlp (외형/색)
           - velocities:     속도 (vx, vy) — GaussMOT 증강 속성
           - identities:     동일성 임베딩 e — GaussMOT 증강 속성
      4) 가변 길이 결과를 [B, max_points, ...] 고정 크기 텐서로 패딩해 반환
    """
    def __init__(
        self,
        in_channels: int = 51,            # 포인트 feature 입력 차원(좌표 제외 속성 수)
        hidden_channels: int = 256,       # PTv3 마지막 encoder 채널 = MLP 입력 차원
        out_channels: int = 128,          # main feature 출력 차원
        max_points: int = 3500,           # 배치당 최대 포인트 수(패딩 기준)
        opacity_bias_init: float = 3.0,   # opacity head의 마지막 bias 초기값(초기엔 불투명하게)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.max_points = max_points

        # PTv3 인코더: U-Net식 encoder(5단계)/decoder(4단계) 구조
        # enc/dec_channels 마지막이 hidden_channels와 맞도록 설정
        self.encoder = PointTransformerV3(
            in_channels=in_channels,
            enc_depths=(1, 1, 1, 1, 1),
            enc_num_head=(1, 2, 4, 8, 16),
            enc_patch_size=(64, 64, 64, 64, 64),
            enc_channels=(64, 64, 128, 128, self.hidden_channels),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(self.hidden_channels, 128, 64, 64),
            dec_num_head=(4, 4, 4, 8),
            dec_patch_size=(64, 64, 64, 64),
            mlp_ratio=4,
            qkv_bias=True,
        )

        # feats head: PTv3 feature → out_channels(main feature)
        self.feats_mlp = self._create_mlp(
            self.hidden_channels,
            self.out_channels,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        # offset head: 포인트 좌표에 더할 3D 위치 보정(Δx,Δy,Δz)
        self.offset_mlp = self._create_mlp(
            self.hidden_channels,
            3,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        # covariance head: 6개 raw 값 → 이후 유효 공분산(3x3 대칭)으로 변환
        self.covs_mlp = self._create_mlp(
            self.hidden_channels,
            6,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        # opacity head: 1채널, 마지막 bias를 opacity_bias_init으로 초기화
        self.opacities_mlp = self._create_mlp(
            self.hidden_channels,
            1,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
            final_bias_init=opacity_bias_init,
        )
        # Augmented Gaussian primitive (GaussMOT ①): per-Gaussian velocity (vx, vy in
        # BEV/ego frame, m/s) and identity embedding e (K-dim, L2-normalised at use).
        # GaussMOT 증강 속성: 속도(vx,vy)와 동일성 임베딩
        self.identity_dim = 8
        # velocity head: 2채널(vx, vy)
        self.velocity_mlp = self._create_mlp(
            self.hidden_channels,
            2,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )
        # identity head: identity_dim 채널(동일성 임베딩 e)
        self.identity_mlp = self._create_mlp(
            self.hidden_channels,
            self.identity_dim,
            hidden_channels=2 * self.hidden_channels,
            dropout=0.1,
        )

    def _create_mlp(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        dropout=0.1,
        final_bias_init=None,
    ) -> nn.Sequential:
        # 헤드 공통 MLP 생성기: (Linear-LayerNorm-GELU-Dropout) x2 → Linear

        if hidden_channels is None:
            hidden_channels = in_channels * 2   # 중간 채널 미지정 시 입력의 2배

        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        # 마지막 Linear의 bias를 지정값으로 초기화(opacity 등에서 사용)
        if final_bias_init is not None:
            nn.init.constant_(mlp[-1].bias, final_bias_init)

        return mlp

    def forward(self, radar_points: list[torch.Tensor]):

        B = len(radar_points)   # 리스트 길이 = 배치 크기 (샘플별 포인트 수는 다름)

        # Extract features from the backbone.
        # 각 샘플의 포인트 개수를 누적합 → PTv3가 요구하는 "offset"(샘플 경계 인덱스)
        offset = torch.tensor([i.shape[0] for i in radar_points]).cumsum(0)
        # 모든 샘플 포인트를 하나로 concat: [sum(N_i), C]
        radar_points = torch.cat(radar_points, 0)
        # PTv3 입력 dict 구성:
        #   feat = 좌표 제외 속성([:,3:]), coord = xyz([:,:3]), offset = 샘플 경계, grid_size = voxel 크기
        radar_dict = {
            "feat": radar_points[:, 3:].float(),
            "coord": radar_points[:, :3].float(),
            "offset": offset.to(radar_points.device),
            "grid_size": 2.0,
        }
        # PTv3 인코더 통과 → 포인트별 feature/coord가 담긴 dict 반환
        radar_point_features = self.encoder(radar_dict)

        # Compute the means, covariances, and opacities.
        means = radar_point_features["coord"].float()                 # 포인트 원래 3D 좌표(Gaussian 중심 초기값)
        offsets = self.offset_mlp(radar_point_features["feat"]).float()  # 중심 보정 Δ(x,y,z)
        covs = self.covs_mlp(radar_point_features["feat"]).float()     # 공분산 raw 6값
        covs = self.make_valid_covariances(covs)                      # 유효 공분산으로 변환
        opacities = torch.sigmoid(self.opacities_mlp(radar_point_features["feat"]).float())  # 불투명도 0~1
        features = self.feats_mlp(radar_point_features["feat"]).float()      # main feature
        velocities = self.velocity_mlp(radar_point_features["feat"]).float() # 속도(vx,vy)
        identities = self.identity_mlp(radar_point_features["feat"]).float() # 동일성 임베딩

        # Recompute the means in 3D space.
        # Gaussian 중심 = 원래 좌표 + 예측한 보정 offset
        means = means + offsets

        # Reconvert to list of tensors.
        # 가변 길이 결과를 [B, max_points, ...] 고정 크기 텐서로 패딩(나머지는 0)
        means_out = torch.zeros(B, self.max_points, 3, device=means.device)
        offsets_out = torch.zeros(B, self.max_points, 3, device=means.device)
        features_out = torch.zeros(B, self.max_points, 128, device=means.device)
        covs_out = torch.zeros(B, self.max_points, 6, device=means.device)
        opacities_out = torch.zeros(B, self.max_points, 1, device=means.device)
        velocities_out = torch.zeros(B, self.max_points, 2, device=means.device)
        identities_out = torch.zeros(B, self.max_points, self.identity_dim, device=means.device)

        # Fill the output tensors with the computed values.
        # offset(누적합) 앞에 0을 붙여 [start, end) 구간을 만들고, 샘플별로 결과를 잘라 채움
        batch_offsets = [0] + radar_dict["offset"].tolist()
        for b, start, end in zip(range(B), batch_offsets[:-1], batch_offsets[1:]):
            # b번째 샘플의 포인트 (end-start)개를 앞쪽에 채움
            means_out[b, :end - start] = means[start:end]
            offsets_out[b, :end - start] = offsets[start:end]
            features_out[b, :end - start] = features[start:end]
            covs_out[b, :end - start] = covs[start:end]
            opacities_out[b, :end - start] = opacities[start:end]
            velocities_out[b, :end - start] = velocities[start:end]
            identities_out[b, :end - start] = identities[start:end]

        # 포인트별 Gaussian 파라미터 dict 반환 (모두 [B, max_points, ...])
        return {
            "centers": means_out,
            "offsets": offsets_out,
            "features": features_out,
            "covariances": covs_out,
            "opacities": opacities_out,
            "velocities": velocities_out,
            "identities": identities_out,
        }

    def make_valid_covariances(self, covs_raw):
        """Convert raw MLP output to valid covariance matrices

        raw 6값을 양의 정부호(symmetric positive-definite)에 가까운 공분산으로 변환.
        대각은 softplus로 양수 보장, 비대각은 tanh*sqrt(분산곱)*0.9로 상관관계를 제한.
        """
        # Create individual components without in-place operations
        # 대각 성분(xx, yy, zz): softplus로 항상 양수 + 작은 floor(1e-4)
        xx = F.softplus(covs_raw[:, 0]) + 1e-4
        yy = F.softplus(covs_raw[:, 3]) + 1e-4
        zz = F.softplus(covs_raw[:, 5]) + 1e-4

        # Off-diagonal elements
        # 비대각 크기 상한 계산용: sqrt(분산_i * 분산_j)
        sqrt_xx_yy = torch.sqrt(xx * yy)
        sqrt_xx_zz = torch.sqrt(xx * zz)
        sqrt_yy_zz = torch.sqrt(yy * zz)

        # 비대각 성분: tanh(-1~1) * 상한 * 0.9 → 상관계수 |ρ|<0.9로 제한(정부호 안정성)
        xy = torch.tanh(covs_raw[:, 1]) * sqrt_xx_yy * 0.9
        xz = torch.tanh(covs_raw[:, 2]) * sqrt_xx_zz * 0.9
        yz = torch.tanh(covs_raw[:, 4]) * sqrt_yy_zz * 0.9

        # Stack all components at once instead of in-place assignment
        # 6개 성분을 (xx, xy, xz, yy, yz, zz) 순서로 묶어 반환
        covs = torch.stack([xx, xy, xz, yy, yz, zz], dim=1)

        return covs
