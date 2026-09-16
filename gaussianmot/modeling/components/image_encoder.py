from typing import Iterable, Optional

# timm: 다양한 이미지 backbone(EfficientViT/ResNet 등)을 features_only로 쉽게 불러오는 라이브러리
import timm
import torch
import torch.nn as nn
# ResNet의 Bottleneck 블록을 neck의 residual stack 재료로 재사용
from torchvision.models.resnet import Bottleneck

# 채널 x 를 받아 Bottleneck(in=x, mid=x//4) 블록을 만드는 헬퍼 (in/out 채널 동일하게 유지)
BottleneckBlock = lambda x: Bottleneck(x, x // 4)


class AlignRes(nn.Module):
    """Align resolutions of the outputs of the backbone.

    backbone이 여러 stage에서 서로 다른 해상도의 feature를 내놓는데, 이를
    같은 해상도로 맞춰주는(보통 작은 것을 업샘플) 모듈.
    """

    def __init__(
        self,
        mode="upsample",                                  # "upsample" 또는 "conv2dtranspose"
        scale_factors: Iterable[int] = [1, 2],            # 각 입력별 업샘플 배율
        in_channels: Iterable[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if mode == "upsample":
            # 배율 s가 1이 아니면 bilinear 업샘플, 1이면 그대로(Identity)
            for s in scale_factors:
                if s != 1:
                    self.layers.append(
                        nn.Upsample(
                            scale_factor=s, mode="bilinear", align_corners=False
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        elif mode == "conv2dtranspose":
            # 학습형 업샘플: ConvTranspose2d로 2배 키움 (배율 1이면 Identity)
            for i, in_c in enumerate(in_channels):
                if scale_factors[i] != 1:
                    self.layers.append(
                        nn.ConvTranspose2d(
                            in_c, in_c, kernel_size=2, stride=2, padding=0
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        else:
            raise NotImplementedError
        return

    def forward(self, x):
        # 입력 리스트 x의 i번째 feature에 i번째 정렬 레이어 적용 → 같은 해상도 리스트 반환
        return [self.layers[i](xi) for i, xi in enumerate(x)]


class PrepareChannel(nn.Module):
    """Transform the feature map to align with Network.

    [역할] backbone에서 나온(해상도 정렬·concat된) feature를 받아, 픽셀마다
    Gaussian Splatting에 필요한 여러 속성으로 "분기(branch)"해 예측하는 멀티-헤드.
      - feats:    Gaussian의 외형/색을 결정할 main feature
      - depth:    픽셀이 카메라로부터 얼마나 떨어졌는지(깊이 분포 logits, depth_num bins)
      - opacity:  Gaussian 불투명도 (sigmoid로 0~1)
      - offsets:  픽셀 3D 위치 보정(Δx,Δy,Δz)
      - velocity: per-pixel 속도 (vx, vy) — GaussMOT 증강 속성
      - identity: per-pixel 동일성 임베딩 e — GaussMOT 증강 속성
    """

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],   # 정렬 후 concat할 각 stage 채널
        interm_c=128,                          # 각 head 내부 중간 채널
        out_c: Optional[int] = 128,            # main feature 출력 채널
        depth_num=0,                           # 깊이 bin 개수(반드시 0이 아니어야 함)
    ):
        super().__init__()
        assert depth_num != 0

        in_c = sum(in_channels)   # 모든 stage feature를 채널 concat하므로 합산이 입력 채널
        # main feature head: conv-BN-ReLU → Bottleneck 3개 → 1x1로 out_c 사영
        self.feats = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, out_c, kernel_size=1, padding=0),
        )
        # depth head: 동일 구조, 출력 채널 = depth_num (깊이 분포 logits)
        self.depth = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, depth_num, kernel_size=1, padding=0)
        )
        # opacity head: 출력 1채널 (forward에서 sigmoid 적용 → Gaussian 불투명도)
        self.opacity = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, 1, kernel_size=1, padding=0)
        )
        # offsets head: 출력 3채널 (픽셀 3D 위치의 Δx,Δy,Δz 보정값)
        self.offsets = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, 3, kernel_size=1, padding=0)
        )
        # Augmented Gaussian primitive (GaussMOT ①): per-pixel velocity (vx, vy) and
        # identity embedding e (K-dim). Lightweight heads (no Bottleneck stack) — these
        # are auxiliary attributes, not the main feature.
        # GaussMOT 증강 Gaussian: 속도/동일성은 보조 속성이라 Bottleneck 없이 가벼운 head 사용
        self.identity_dim = 8   # 동일성 임베딩 차원 K
        # velocity head: conv-BN-ReLU → 1x1로 2채널(vx, vy)
        self.velocity = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, 2, kernel_size=1, padding=0),
        )
        # identity head: conv-BN-ReLU → 1x1로 identity_dim 채널(동일성 임베딩 e)
        self.identity = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, self.identity_dim, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # 입력 x(concat된 feature)를 각 head로 분기.
        # opacity만 sigmoid로 0~1 범위로 변환, 나머지는 raw 출력.
        return (
            self.feats(x), self.depth(x), self.opacity(x).sigmoid(), self.offsets(x),
            self.velocity(x), self.identity(x),
        )

class AGPNeck(nn.Module):
    """
    Upsample outputs of the backbones, group them and align them to be compatible with Network.

    [역할] backbone의 multi-stage 출력을 (1) 해상도 정렬 → (2) concat으로 묶고 →
    (3) PrepareChannel로 Gaussian 속성들로 분기하는 neck.

    Note: mimics UpsamplingConcat in SimpleBEV.
    """

    def __init__(
        self,
        align_res_layer,                                  # AlignRes 인스턴스(해상도 정렬)
        prepare_c_layer,                                  # PrepareChannel 인스턴스(속성 분기)
        group_method=lambda x: torch.cat(x, dim=1),       # 정렬된 feature를 묶는 방법(채널 concat)
        list_output=False,                                # main feature를 리스트로 감쌀지 여부
    ):
        """
        Args:
            - align_res_layer: upsample layers at different resolution to the same.
            - group_method: how to gather the upsampled layers.
            - prepare_c_layer: change the channels of the upsampled layers in order to align with the network.
        """
        super().__init__()

        self.align_res_layer = align_res_layer
        self.group_method = group_method
        self.prepare_c_layer = prepare_c_layer
        self.list_output = list_output

    def forward(self, x: Iterable[torch.Tensor]):
        # 입력이 5차원([B, N, C, H, W]; N=카메라 대수)이면 batch와 N을 합쳐 4D로 펼침
        # [B, N, C, H, W] -> [B*N, C, H, W]
        if x[0].ndim == 5:
            x = [y.flatten(0,1) for y in x]

        # Align resolution of inputs.
        # 각 stage feature를 같은 해상도로 정렬
        x = self.align_res_layer(x)

        # Group inputs.
        # 정렬된 feature들을 채널 방향으로 concat → 하나의 feature map
        x = self.group_method(x)

        # Change channels of final input.
        # PrepareChannel로 Gaussian 속성 6종 분기
        x, depth, opacity, offsets, velocity, identity = self.prepare_c_layer(x)
        if self.list_output:
            x = [x]   # main feature를 리스트로 감쌈(다운스트림 호환용)
        return x, depth, opacity, offsets, velocity, identity


class PixelsToGaussians(nn.Module):
    """
    [파이프라인에서의 역할]
    카메라 이미지를 입력받아 "픽셀마다 하나의 3D Gaussian"의 파라미터를 예측하는 모듈.
    GaussMOT 핵심 기여인 Gaussian Splatting의 카메라 분기로, 출력 속성들이 그대로
    GS 렌더러로 들어가 BEV feature를 만든다.

    구성:
      - backbone: timm으로 만든 EfficientViT(기본 efficientvit_l2)/ResNet 등 (features_only)
      - neck:     AGPNeck (해상도 정렬 + concat + PrepareChannel 분기)
    출력(dict) — 픽셀별 Gaussian 파라미터:
      - features: main feature (Gaussian의 외형/색)
      - depth:    깊이 분포 logits → mean(중심) 결정에 사용
      - opacity:  불투명도(0~1)
      - offsets:  3D 위치 보정(Δx,Δy,Δz) → scale/rotation/mean 산출에 사용
      - velocity: per-pixel 속도(vx, vy)
      - identity: per-pixel 동일성 임베딩
    """
    def __init__(
        self,
        model_name: str = "efficientvit_l2.r384_in1k",  # 기본 backbone: EfficientViT-L2
        out_indices: tuple = (0, 1, 2, 3),               # backbone에서 뽑아올 stage 인덱스
        pretrained: bool = True,                          # 사전학습 가중치 사용 여부
        in_channels: int = 3,                             # 입력 이미지 채널(RGB=3)
        out_channels: int = 128,                          # main feature 출력 채널
        neck: Optional[nn.Module] = None,                 # AGPNeck (없으면 backbone 출력 그대로)
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_name = model_name
        self.out_indices = out_indices
        self.pretrained = pretrained

        # timm으로 backbone 생성: features_only=True면 stage별 feature map 리스트를 반환
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels
        )
        self.neck = neck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 카메라 이미지 → backbone으로 multi-stage feature 리스트 추출
        x = self.backbone(x)
        # neck(AGPNeck)이 있으면 해상도 정렬·concat·속성 분기를 수행
        if self.neck is not None:
            x = self.neck(x)

        # neck 출력 튜플(features, depth, opacity, offsets, velocity, identity)을
        # 의미별 키로 dict화하여 반환 → 뒤단 Gaussian 생성/렌더에 사용
        return {
            "features": x[0],
            "depth": x[1],
            "opacity": x[2],
            "offsets": x[3],
            "velocity": x[4],
            "identity": x[5],
        }
