"""Detection-friendly BEV encoder (replaces DPTHead).

[파이프라인에서의 역할]
ConcatFuser가 만든 (멀티스케일) BEV feature를 받아, "검출(detection)에 친화적인"
단일 해상도 BEV feature map으로 다듬는 인코더이다. 이 출력이 CenterHead/TrackHead의
입력으로 들어간다.

DPTHead is the ViT Dense Prediction Transformer head — designed for dense
prediction tasks (segmentation, depth). Its progressive upsampling smooths
spatial discrimination, which hurts sparse object detection.
(원래 쓰던 DPTHead는 분할/깊이용이라 점진적 업샘플로 공간 경계가 뭉개져서
 "드문드문 떨어진 객체 검출"에는 불리 → 이 모듈로 대체)

This module:
  - Uses cr1 (full-res 200x200) from CMXFuser as the primary feature
    (가장 큰 해상도 cr1을 주 feature로 사용)
  - Adds upsampled cr4 (25x25) for global context as additive injection
    (가장 작은 cr4를 업샘플해 전역 맥락으로 더해줌)
  - Stacks N lightweight ConvNeXt-style 2D blocks
  - Outputs single-scale [B, out_channels, 200, 200] for TransFusionHead

Multi-scale features (cr2, cr3) are intentionally not used here — the
detection head's own heatmap branch + transformer decoder handle multi-scale
context via attention.
(중간 스케일 cr2, cr3는 의도적으로 사용하지 않음)
"""
from typing import List, Sequence

import torch
import torch.nn as nn


class _ConvNeXtBlock2D(nn.Module):
    """ConvNeXt-style 2D block: 7x7 depthwise + 1x1 expand/contract.

    concat_fuser의 동명 블록과 동일한 구조(7x7 dw → norm → 1x1 확장 → GELU → 1x1 축소 + residual).
    """

    def __init__(self, dim: int):
        super().__init__()
        # 7x7 depthwise conv: 넓은 수용영역을 적은 연산으로 확보
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)            # LayerNorm 유사 정규화
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # 채널 4배 확장
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # 채널 복원

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x          # 잔차 보관
        x = self.dwconv(x)    # 공간 혼합
        x = self.norm(x)
        x = self.pwconv1(x)   # 채널 확장
        x = self.act(x)
        x = self.pwconv2(x)   # 채널 복원
        return x + residual   # 잔차 더하기


class DetBEVEncoder(nn.Module):
    """Detection-friendly BEV encoder. Drop-in replacement for DPTHead.

    [동작 요약]
    1) cr1(최대 해상도)을 1x1 conv로 out_channels로 사영
    2) (옵션) cr4(최소 해상도)를 사영 후 8배 업샘플해 전역 맥락으로 더함
    3) ConvNeXt 블록 N개로 정제 → 단일 스케일 BEV feature 반환

    Args:
        in_channels: channel dim of fuser features (cr1..cr4 all share this).
        out_channels: channel dim expected by detection head (256).
        num_blocks: number of ConvNeXt-2D residual blocks after projection.
        use_global_context: if True, also project cr4 (smallest) and add it
                           upsampled for receptive-field enlargement without
                           DPT-style smoothing.
    """

    def __init__(
        self,
        in_channels: int = 128,           # ConcatFuser 출력 채널(cr1..cr4 공통)
        out_channels: int = 256,          # 검출 head가 기대하는 출력 채널
        num_blocks: int = 3,              # 사영 후 ConvNeXt 정제 블록 수
        use_global_context: bool = True,  # cr4 전역맥락 주입 여부
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_global_context = use_global_context

        # cr1(주 feature)을 1x1 conv로 out_channels 채널로 사영 + 정규화
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(1, out_channels),
        )
        if use_global_context:
            # cr4(전역 맥락)를 같은 채널로 사영하는 별도 1x1 conv + 정규화
            self.ctx_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(1, out_channels),
            )
            # Upsample 25x25 -> 200x200 via bilinear (additive, no learned upsampler).
            # cr4(25x25)를 8배 bilinear 업샘플 → cr1(200x200)과 더할 수 있게 함
            self.ctx_upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        # 융합 feature 정제용 ConvNeXt 블록들
        self.blocks = nn.ModuleList(
            [_ConvNeXtBlock2D(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, fused: Sequence[torch.Tensor]) -> torch.Tensor:
        """fused = [cr1@200, cr2@100, cr3@50, cr4@25]."""
        cr1 = fused[0]                  # 최대 해상도 feature [B, in_channels, 200, 200]
        x = self.input_proj(cr1)        # -> [B, out_channels, 200, 200]
        if self.use_global_context:
            cr4 = fused[-1]             # 최소 해상도 feature [B, in_channels, 25, 25]
            # cr4를 사영 후 8배 업샘플: [B, out_channels, 25, 25] -> [B, out_channels, 200, 200]
            ctx = self.ctx_upsample(self.ctx_proj(cr4))
            x = x + ctx                 # 전역 맥락을 주 feature에 더함(잔차식 주입)
        # ConvNeXt 블록들로 최종 정제
        for block in self.blocks:
            x = block(x)
        # 단일 스케일 BEV feature 반환: [B, out_channels, 200, 200]
        return x
