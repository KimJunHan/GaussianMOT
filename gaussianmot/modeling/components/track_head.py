# 타입 힌트용 Dict/Optional, PyTorch 텐서/모듈/함수형 API import
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """3x3 Conv-BN-ReLU ×2 + skip connection (residual block).

    [track 정체 수정 B] track_head가 detection-최적화된 shared 'decoded' feature 위에서
    instance-discriminative 표현을 키울 수 있도록 깊이를 늘리는 빌딩블록.
    얕은 2-conv로는 det가 지배하는 trunk에서 추적 표현이 안 자라 track loss가 정체했음.
    """
    def __init__(self, c: int):                                 # c: 채널 수(입출력 동일)
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)  # 3x3 conv (채널 유지, BN이 bias 흡수)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)  # 3x3 conv (채널 유지)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        r = x                                                   # skip(잔차) 경로 보존
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)       # conv-bn-relu
        x = self.bn2(self.conv2(x))                             # conv-bn (relu는 합산 후)
        return F.relu(x + r, inplace=True)                      # residual 합 후 relu


class TrackHead(nn.Module):
    """
    Tracking head - per-pixel embedding 예측 (track 정체 수정 B + 수렴 개선 ③⑤).

    [파이프라인에서의 역할]
    BEV(조감도) feature map의 "픽셀마다" 추적용 임베딩 벡터를 뽑아내는 head이다.
    검출(CenterHead)이 "무엇이 어디에 있는가"를 맡는다면, 이 TrackHead는
    "같은 물체인가(동일성)"를 판별할 수 있는 표현(embedding)을 학습한다.
    각 픽셀 임베딩을 채널 방향으로 L2 정규화하므로, inference 시 두 검출의
    임베딩 내적 = cosine similarity 가 되어 프레임 간 association(연결)에 그대로 쓸 수 있다.

    [B] residual block ×2로 표현력 강화(det가 지배하는 trunk에서 추적 표현 학습).

    [⑤ fused_bev skip] 입력은 detection-decoder를 거쳐 "검출용으로 뭉개진" `decoded`(256ch).
    여기에 decoder 이전의 융합 BEV(`fused_bev` scale-0, 128ch)를 skip으로 추가 주입해
    추적이 검출용으로 collapse되지 않은 풍부한 feature를 보게 한다. 검증된 검출 decoder는
    그대로 두므로 oracle 파이프라인을 깨지 않는다(근본원인 완화).

    [③ projection head (SimCLR/MoCo)] contrastive loss는 별도 projection(MLP) 출력
    `{key}_track_proj`으로 계산하고, association/inference는 표현 `{key}_track_embed`을 쓴다.
    loss-공간과 매칭-공간을 분리해 대조학습 표현 품질을 높인다(projection은 추론 때 미사용).

    입력: decoded BEV [B, dim_last, H, W] (+ 옵션 skip [B, skip_dim, H, W])
    출력:
      - {key}_track_embed: [B, D, H, W]  L2-normalized  (association/inference용)
      - {key}_track_proj : [B, P, H, W]  L2-normalized  (contrastive loss 전용)
    """
    def __init__(
        self,
        dim_last: int,              # 입력 BEV feature 채널 수 C (decoded)
        embed_dim: int = 64,        # association 임베딩 차원 D
        key: str = 'vehicle',       # 출력 dict 키 prefix (예: "vehicle_track_embed")
        skip_dim: Optional[int] = None,  # ⑤ skip(fused_bev scale-0) 채널 수. None이면 skip 비활성
        proj_dim: int = 64,         # ③ projection 출력 차원 P (contrastive 전용)
    ):
        super().__init__()
        self.key = key
        self.embed_dim = embed_dim

        # ⑤ skip 투영: fused_bev(skip_dim) → dim_last로 사영해 decoded에 잔차 주입.
        self.skip_proj = nn.Conv2d(skip_dim, dim_last, 1) if skip_dim else None

        # 임베딩 추출 head: stem(3x3 conv-BN-ReLU) → residual block ×2 → 1x1 conv(C -> D)
        self.embed_head = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            _ResBlock(dim_last),
            _ResBlock(dim_last),
            nn.Conv2d(dim_last, embed_dim, 1),
        )

        # ③ projection head: 표현 h(embed_dim) → MLP → proj(proj_dim). contrastive loss 전용.
        #   1x1 conv(D->2D) → BN → ReLU → 1x1 conv(2D->P) (SimCLR식 2-layer MLP, per-pixel)
        self.proj_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, proj_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # ⑤ skip 주입: fused_bev scale-0를 dim_last로 사영해 decoded에 잔차로 더함.
        if self.skip_proj is not None and skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:                 # 해상도 다르면 맞춤(보통 동일 200x200)
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_proj(skip.to(x.dtype))            # 풍부한 융합 feature 잔차 주입

        # 표현 h: [B, C(+skip), H, W] → [B, D, H, W]
        h = self.embed_head(x)
        # association 임베딩: 채널 L2 정규화 (cosine similarity = 단위벡터 내적)
        embed = F.normalize(h, dim=1)
        # ③ contrastive 전용 projection: h → proj → L2 정규화
        proj = F.normalize(self.proj_head(h), dim=1)
        # 출력: association용 embed + loss 전용 proj 둘 다 반환
        return {
            f"{self.key}_track_embed": embed,
            f"{self.key}_track_proj": proj,
        }
