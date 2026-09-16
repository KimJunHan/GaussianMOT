"""Simple concat fuser (CR3DT-style).

[파이프라인에서의 역할]
카메라 BEV feature와 레이더 BEV feature를 "융합(fusion)"하는 모듈.
GaussMOT 파이프라인에서 카메라(PixelsToGaussians→GS 렌더)와
레이더(PointsToGaussians→GS 렌더)는 각각 BEV grid 위 feature map을 만든다.
이 둘을 합쳐 하나의 BEV feature로 만들어 뒤단 DetBEVEncoder/검출·추적 head로 넘긴다.

Replaces the segmentation-specialized CMXFuser (FRM+FFM channel/spatial
attention) with the simplest possible fusion: channel-wise concatenation of
camera-BEV and radar-BEV, followed by a small ConvNeXt-2D residual stack.
(분할용으로 무겁던 CMX 어텐션 융합 대신, CR3DT처럼 "채널 concat + 가벼운 conv"로 단순화)

Output is a single full-resolution tensor (200x200) — drop multi-scale
pyramid since DetBEVEncoder already operates at single scale.
"""
from typing import Sequence

import torch
import torch.nn as nn


class _ConvNeXtBlock2D(nn.Module):
    """ConvNeXt-style 2D residual block (7x7 depthwise + 1x1 expand/contract).

    구조: 7x7 depthwise conv(큰 수용영역) → norm → 1x1로 4배 확장 → GELU →
          1x1로 원래 채널 복원 → residual(입력 더하기).
    """

    def __init__(self, dim: int):
        super().__init__()
        # 7x7 depthwise conv: 채널별로 따로 큰 공간 필터 적용(넓은 수용영역, 적은 연산량)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # GroupNorm(1, dim) = LayerNorm 유사 정규화(배치 크기에 둔감)
        self.norm = nn.GroupNorm(1, dim)
        # 1x1 conv로 채널 4배 확장 (MLP의 첫 층 역할)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        # 1x1 conv로 채널을 원래 dim으로 축소
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x          # 잔차 연결용 입력 보관
        x = self.dwconv(x)    # [B, dim, H, W] 공간 혼합
        x = self.norm(x)
        x = self.pwconv1(x)   # [B, 4*dim, H, W] 채널 확장
        x = self.act(x)
        x = self.pwconv2(x)   # [B, dim, H, W] 채널 복원
        return x + residual   # 잔차 더해 출력


class ConcatFuser(nn.Module):
    """Channel-concat fusion of camera-BEV and radar-BEV (CR3DT-style).

    [동작 요약]
    1) 카메라 BEV와 레이더 BEV를 채널 방향으로 concat → 2C 채널
    2) 1x1 conv로 out_channels로 사영 + GroupNorm
    3) ConvNeXt 블록 N개로 융합 feature 정제
    4) (옵션) stride-2 conv로 1/2,1/4,1/8 해상도까지 만들어 multi-scale 리스트로 반환

    Input shapes:
        camera_bev: [B, embed_dims, H, W]
        radar_bev:  [B, embed_dims, H, W]

    Output:
        - When `return_multi_scale=False` (default): single tensor
          [B, out_channels, H, W]
        - When `return_multi_scale=True`: list [out, out_down2, out_down4, out_down8]
          for backward compatibility with components expecting multi-scale.
    """

    def __init__(
        self,
        embed_dims: int = 128,            # 입력 카메라/레이더 BEV 각각의 채널 수 C
        out_channels: int = 128,          # 융합 후 출력 채널 수
        num_blocks: int = 2,              # ConvNeXt 정제 블록 개수
        return_multi_scale: bool = True,  # True면 피라미드(1, 1/2, 1/4, 1/8) 반환
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.return_multi_scale = return_multi_scale

        # 2 * embed_dims (concat) → out_channels via 1x1
        # concat된 2C 채널을 1x1 conv로 out_channels로 줄이고 GroupNorm으로 정규화
        self.proj = nn.Sequential(
            nn.Conv2d(2 * embed_dims, out_channels, kernel_size=1),
            nn.GroupNorm(1, out_channels),
        )
        # 융합 feature를 정제하는 ConvNeXt 블록들
        self.blocks = nn.ModuleList(
            [_ConvNeXtBlock2D(out_channels) for _ in range(num_blocks)]
        )
        if self.return_multi_scale:
            # stride=2 conv 3개로 해상도를 절반씩 줄여 1/2, 1/4, 1/8 scale 생성
            self.down2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.down4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.down8 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, camera_bev: torch.Tensor, radar_bev: torch.Tensor):
        # 카메라/레이더 BEV를 채널 방향으로 결합: [B,C,H,W]+[B,C,H,W] -> [B, 2C, H, W]
        x = torch.cat([camera_bev, radar_bev], dim=1)  # [B, 2C, H, W]
        # 1x1 conv + norm으로 out_channels 채널의 융합 feature로 사영: [B, 2C, H, W] -> [B, out_channels, H, W]
        x = self.proj(x)
        # ConvNeXt 블록들을 차례로 통과시켜 융합 feature 정제
        for block in self.blocks:
            x = block(x)
        # 단일 스케일만 필요하면 여기서 [B, out_channels, H, W] 그대로 반환
        if not self.return_multi_scale:
            return x
        # Build multi-scale list for DetBEVEncoder compatibility (cr1@H, cr2@H/2, cr3@H/4, cr4@H/8)
        # multi-scale 요청 시: 해상도를 절반씩 줄여 피라미드 생성
        x2 = self.down2(x)    # [B, out_channels, H/2, W/2]
        x4 = self.down4(x2)   # [B, out_channels, H/4, W/4]
        x8 = self.down8(x4)   # [B, out_channels, H/8, W/8]
        # [원본, 1/2, 1/4, 1/8] 순서 리스트로 반환 (DetBEVEncoder가 cr1..cr4로 사용)
        return [x, x2, x4, x8]
