"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

[한글 설명 - GaussMOT 맥락]
이 파일은 PTv3(Point Transformer V3) 포인트클라우드 트랜스포머 백본의 벤더(외부) 코드다.
원본은 Pointcept 프로젝트의 분리(detached) 버전이며, 우리 GaussMOT 파이프라인에서는
레이더 인코더(PointsToGaussians, radar_encoder.py)가 이 모듈을 import 하여
"레이더 포인트의 per-point feature를 추출"하는 feature extractor 로 사용한다(검출기 아님).

입출력 개요:
- 입력: dict {feat(per-point 특징), coord(xyz 좌표), offset(샘플 경계: 배치별 누적 점 개수)}
- 출력: per-point feature 가 담긴 Point(dict) 구조체

핵심 아이디어:
- serialization(z-order/Hilbert 등 공간충전곡선)으로 순서 없는 3D 점을 1D 정렬열로 만들어,
  KNN 없이도 인접 점끼리 윈도우(patch) attention 이 가능하게 한다(PTv3 핵심 트릭).
- U-Net 형태의 encoder(다운샘플)/decoder(업샘플) 로 멀티스케일 특징을 학습한다.
"""

import sys
from functools import partial
from addict import Dict  # 키를 속성처럼(.) 접근 가능한 dict — Point 구조체의 베이스
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv  # 희소 3D 컨볼루션(SparseConv) — voxel/grid 위 점에만 연산
import torch_scatter  # segment 단위 reduce(scatter) 연산 — pooling 시 클러스터별 집계에 사용
from timm.models.layers import DropPath  # stochastic depth(잔차 경로 확률적 drop)
from collections import OrderedDict

from typing import Tuple, Literal

try:
    # flash attention 은 선택적 의존성. 없으면 None 으로 두고 표준 attention 경로를 사용한다.
    import flash_attn
except ImportError:
    flash_attn = None

# serialization 인코더: grid 좌표를 z-order/Hilbert 등의 1D 정수 코드로 변환하는 함수
from .serialization import encode


# --- offset / batch 표현 상호 변환 유틸 ---
# offset: 각 샘플의 누적 점 개수 경계(예: [n0, n0+n1, ...]). batch: 각 점이 속한 샘플 인덱스.
# 모두 추론 전용(@torch.inference_mode)으로 grad 추적 없이 가볍게 동작.


@torch.inference_mode()
def offset2bincount(offset):
    # offset(누적 개수)을 차분하여 샘플별 점 개수(bincount)로 변환.
    # 맨 앞에 0을 prepend 하여 첫 샘플 개수도 정확히 나오게 함.
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    # offset -> 각 점의 배치 인덱스 벡터(batch). 예: 개수 [2,3] -> [0,0,1,1,1].
    bincount = offset2bincount(offset)  # 샘플별 점 개수
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)  # 샘플 인덱스를 그 개수만큼 반복


@torch.inference_mode()
def batch2offset(batch):
    # batch(각 점의 샘플 인덱스) -> offset(누적 개수). bincount 의 누적합.
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        # offset 와 batch 중 하나만 주어지면 나머지를 자동 생성(둘은 동등한 표현).
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        [한글] 비정렬 3D 점들을 공간충전곡선(z-order/Hilbert)으로 1D 정수 코드(code)로 인코딩하고,
        그 코드 기준 정렬 순서(order)와 역매핑(inverse)을 만들어 둔다.
        이렇게 만든 순서를 따라 점을 일렬로 늘어놓으면 공간상 인접한 점들이 가까이 모여,
        고정 길이 윈도우(patch) 단위로 잘라 attention 을 적용할 수 있다(KNN 불필요).
        """
        assert "batch" in self.keys()  # 배치 정보 필수
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            # grid_coord 가 없으면 연속 좌표 coord 를 grid_size 로 양자화(voxel index)하여 생성.
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()  # 최소좌표를 0으로 평행이동 후 grid_size 로 나눠 내림 -> 정수 격자 좌표

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            # depth: 직렬화 큐브 한 변 길이를 2^depth 로 표현. 최대 격자좌표값의 비트수로 자동 결정.
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        # int64 안에 코드를 담아야 하므로, 좌표 3축(depth*3) + 배치비트 합이 63 이하여야 함.
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        # 좌표 직렬화 깊이는 16(48bit)로 제한(OCNN 관례). grid 0.01m 기준 655.36^3 m^3 큐브까지 표현.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        # 여러 order(z, hilbert 등 k개)에 대해 각각 인코딩 -> (k, n) 코드 텐서.
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)  # (k, n): k개 직렬화 방식 x n개 점
        order = torch.argsort(code)  # 각 방식별로 코드 오름차순 정렬 인덱스(=직렬화 순서)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )  # inverse[order[i]] = i : 정렬된 순서 -> 원래 위치로 되돌리는 역매핑

        if shuffle_orders:
            # 학습 시 여러 order 중 사용 순서를 무작위 섞어 특정 곡선에 과적합되지 않게 함.
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 직렬화 결과를 Point 에 저장(이후 attention/pooling 에서 재사용).
        self["serialized_code"] = code      # (k, n) 정수 코드
        self["serialized_order"] = order    # (k, n) 정렬 순서
        self["serialized_inverse"] = inverse  # (k, n) 역매핑

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.

        [한글] 점이 격자에서 희소하게 분포하므로, SpConv 가 쓰는 SparseConvTensor 를 만든다.
        feat(특징) + indices(batch,grid_coord) + spatial_shape 로 희소 텐서를 구성한다.
        pad 는 공간 shape 여유분(경계 컨볼루션 안정용)으로 보임.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            # grid_coord 없으면 serialization 과 동일 방식으로 좌표 양자화하여 생성.
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape  # 이미 정해진 희소 공간 shape 재사용
        else:
            # 최대 격자좌표 + pad 를 공간 shape 로 사용(모든 점을 담을 크기 확보).
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,  # (N, C) per-point 특징
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),  # (N, 4): [batch_idx, gx, gy, gz] 형식의 희소 인덱스
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,  # 마지막 batch 인덱스+1 = 배치 크기
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat  # 이후 SpConv 모듈에서 사용


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.

    [한글] Point(dict) 구조체를 입력으로 받는 모듈임을 표시하는 마커 베이스 클래스.
    PointSequential 은 이 타입 여부로 Point 를 통째로 넘길지 feat 만 넘길지를 구분한다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    [한글] nn.Sequential 의 Point 버전. 들어 있는 모듈을 순서대로 적용하되,
    각 모듈 종류(PointModule / SpConv / 일반 nn.Module)에 맞게 입력을 적절히 전달한다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # OrderedDict 로 주면 키를 모듈 이름으로, 아니면 위치 인덱스를 이름으로 등록.
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # 키워드 인자로 넘긴 모듈도 이름으로 등록(중복 이름 방지).
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        # 정수 인덱스로 내부 모듈에 접근(음수 인덱스 지원).
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)  # idx 번째까지 순회하여 해당 모듈 반환
        return next(it)

    def __len__(self):
        return len(self._modules)  # 등록된 모듈 개수

    def add(self, module, name=None):
        # 컨테이너에 모듈 추가(이름 미지정 시 현재 길이를 이름으로 사용).
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # 등록된 모듈을 순서대로 적용. 모듈 종류에 따라 입력 전달 방식이 다르다.
        for k, module in self._modules.items():
            # Point module: Point 를 통째로 받아 처리(직접 호출).
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module: 희소 텐서에 작용. Point 면 sparse_conv_feat 를 갱신하고 feat 동기화.
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features  # 희소 결과 -> feat 로 반영
                else:
                    input = module(input)
            # PyTorch module: 일반 텐서 연산. Point 면 feat 에만 적용 후 sparse_conv_feat 도 동기화.
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )  # feat 변경을 희소 텐서에도 반영(둘을 일치시킴)
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:  # 점이 없으면 건너뜀(빈 텐서 보호)
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    """[한글] Prompt-Driven Normalization. 데이터셋/도메인 조건(condition)에 따라
    서로 다른 정규화 파라미터를 쓰거나(decouple), context 로 정규화 결과를 변조(adaptive)한다.
    멀티 데이터셋 학습용 기능으로, 우리 단일-레이더 용도에서는 보통 비활성(기본 LayerNorm/BN 사용)으로 보임."""

    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions  # 지원 도메인 목록
        self.decouple = decouple      # 도메인별 별도 norm 사용 여부
        self.adaptive = adaptive      # context 기반 변조 사용 여부
        if self.decouple:
            # 도메인마다 독립된 norm 레이어를 둠.
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer  # 공통 norm 하나
        if self.adaptive:
            # context -> (scale, shift) 변조 파라미터를 생성하는 MLP.
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())  # 특징과 도메인 조건 필요
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]  # 배치 내 첫 조건 사용
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]  # 해당 도메인 norm 선택
        else:
            norm = self.norm
        point.feat = norm(point.feat)  # 정규화 적용
        if self.adaptive:
            assert "context" in point.keys()
            # context 로부터 scale/shift 를 만들어 FiLM 식 변조: feat*(1+scale)+shift.
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    """[한글] Relative Position Encoding(상대 위치 인코딩).
    윈도우 내 두 점의 격자 좌표 차이(상대위치)를 학습 가능한 테이블에서 조회해
    attention 점수에 더하는 바이어스를 만든다. flash attention 사용 시에는 비활성화된다."""

    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        # 상대 위치 좌표를 제한할 경계값(윈도우 크기에서 경험적으로 산출).
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1  # [-bnd, +bnd] 범위의 인덱스 개수
        # 학습 테이블: 3축 x 위치빈 수, head 별 바이어스.
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)  # 절단 정규분포 초기화

    def forward(self, coord):
        # coord: 윈도우 내 점쌍의 상대 격자 좌표 (..., 3).
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd  (경계로 클램프)
            + self.pos_bnd  # relative position to positive index (음수->0기준 양수 인덱스)
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride (축별 테이블 구간 분리)
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))  # 테이블에서 바이어스 조회
        out = out.view(idx.shape + (-1,)).sum(3)  # x,y,z 3축 바이어스 합산
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)  attention 텐서 레이아웃에 맞춤
        return out


class SerializedAttention(PointModule):
    """[한글] PTv3 의 핵심 모듈: Serialized(윈도우) Attention.
    직렬화 순서(serialized_order)로 점을 일렬로 늘어놓은 뒤, 고정 길이 patch_size(K) 단위로
    잘라 각 윈도우(patch) 안에서만 self-attention 을 수행한다.
    -> 공간상 인접한 점끼리 attention 하게 되어 전역 O(N^2) 없이 효율적이다.
    윈도우 경계를 patch 배수로 맞추기 위한 padding/unpadding 처리가 핵심 디테일이다."""

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,    # 여러 직렬화 order 중 이 블록이 사용할 인덱스
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0  # head 별로 채널이 균등 분할되어야 함
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5  # attention 스케일 1/sqrt(d)
        self.order_index = order_index
        self.upcast_attention = upcast_attention  # attention 계산을 float32 로 승격할지
        self.upcast_softmax = upcast_softmax      # softmax 를 float32 로 승격할지
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            # flash attention 경로: RPE/upcast 류는 호환 안 되므로 모두 꺼져 있어야 함.
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size  # 고정 윈도우 크기 그대로 사용
            self.attn_drop = attn_drop    # flash 함수에 dropout 확률(float)로 전달
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            # 비-flash 경로: 마스크를 안 쓰려고 실제 윈도우 크기를 점 개수와 patch_size_max 의 min 으로 동적 설정.
            self.patch_size_max = patch_size
            self.patch_size = 0  # forward 에서 동적으로 채워짐
            self.attn_drop = torch.nn.Dropout(attn_drop)  # 표준 dropout 모듈

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)  # Q,K,V 한 번에 투영
        self.proj = torch.nn.Linear(channels, channels)  # attention 출력 투영
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None  # 선택적 상대위치 인코딩

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        # [한글] RPE 용 윈도우 내 점쌍의 상대 격자좌표(K x K) 계산. 한 번 계산 후 point 에 캐시.
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]  # 직렬화 순서로 재배열
            grid_coord = grid_coord.reshape(-1, K, 3)  # (윈도우수, K, 3)
            # broadcast 차분으로 윈도우 내 모든 쌍의 상대좌표 (윈도우수, K, K, 3).
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        # [한글] 각 샘플의 점 개수를 patch_size(K) 의 배수로 맞추기 위한 padding/unpadding 인덱스 생성.
        # pad: 패딩 포함 인덱스로 원본 feat 를 재배열(부족분은 같은 윈도우 점 복제로 채움).
        # unpad: 패딩 결과에서 원본 점만 다시 골라내는 인덱스.
        # cu_seqlens: flash attention 의 가변길이 시퀀스 경계(누적 시퀀스 길이).
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):  # 캐시 없으면 계산
            offset = point.offset
            bincount = offset2bincount(offset)  # 샘플별 점 개수
            # 각 샘플 개수를 K 의 배수로 올림(ceil) -> 패딩 후 개수.
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            # 점 개수가 K 보다 클 때만 패딩(작으면 그대로 두어 단일 윈도우 처리).
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))  # 앞에 0 추가한 원본 누적경계
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))  # 패딩 누적경계
            pad = torch.arange(_offset_pad[-1], device=offset.device)   # 패딩 인덱스 초기값(항등)
            unpad = torch.arange(_offset[-1], device=offset.device)     # 언패딩 인덱스 초기값(항등)
            cu_seqlens = []
            for i in range(len(offset)):  # 샘플별로 처리
                # unpad: 원본 i번째 구간을 패딩 좌표계 시작점만큼 이동.
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:  # 패딩이 발생한 샘플
                    # 마지막 미완성 윈도우의 빈 자리를, 바로 앞 윈도우의 동일 위치 점으로 복제하여 채움.
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                # pad 인덱스를 원본 feat 좌표계로 보정(패딩 좌표계 -> 원본 인덱스).
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                # 이 샘플의 윈도우 경계들(K 간격)을 flash attention 시퀀스 경계로 기록.
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            # 마지막에 전체 길이를 추가하여 cu_seqlens 를 닫음(flash attention 규격).
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            # 비-flash: 실제 윈도우 크기를 (최소 점 개수, max) 중 작은 값으로 설정해 마스크 없이 처리.
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size  # 윈도우(패치) 길이
        C = self.channels

        # 패딩/언패딩 인덱스 및 flash 시퀀스 경계 획득.
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # 이 블록이 쓸 직렬화 순서를 패딩 인덱스로 재배열 -> 윈도우 단위로 정렬된 점 순서.
        order = point.serialized_order[self.order_index][pad]
        # attention 후 원래 점 순서로 되돌릴 역매핑(언패딩 반영).
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        # feat 를 QKV 로 투영한 뒤 직렬화 순서로 재배열(윈도우별로 점이 연속하도록).
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            # (윈도우수, K, 3, H, head채널) 로 펴서 q/k/v 로 분리. N'=윈도우 개수.
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:  # 수치 안정 위해 float32 승격(옵션)
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)  윈도우 내 attention 점수
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))  # 상대위치 바이어스 더함
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)  # 윈도우 내 정규화
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)  # 가중합 후 (N, C) 로 복원
        else:
            # flash attention 경로(메모리 효율적). 가변길이 윈도우를 cu_seqlens 로 구분.
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),  # fp16 패킹 QKV
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,  # 학습 시에만 dropout
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)  # 원래 dtype 로 복귀
        feat = feat[inverse]  # 윈도우 정렬 -> 원래 점 순서로 되돌림

        # ffn
        feat = self.proj(feat)        # 출력 투영
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    """[한글] 트랜스포머 블록의 FFN. fc1 -> 활성화 -> fc2 의 2층 MLP(채널 확장 후 복원)."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels       # 미지정 시 입력 채널과 동일
        hidden_channels = hidden_channels or in_channels  # 미지정 시 입력 채널과 동일
        self.fc1 = nn.Linear(in_channels, hidden_channels)  # 확장
        self.act = act_layer()                              # 비선형
        self.fc2 = nn.Linear(hidden_channels, out_channels)  # 복원
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        # [한글] PTv3 트랜스포머 블록: CPE -> (norm)attention -> (norm)MLP, 각 단계 잔차 연결.
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm  # True 면 pre-norm(잔차 전 정규화), False 면 post-norm

        # CPE(Conditional Positional Encoding): 희소 3D 컨볼루션으로 위치 정보를 특징에 주입.
        # serialized attention 자체엔 절대위치가 없으므로 SubMConv3d 가 지역 위치 단서를 보강한다.
        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,  # 같은 stage 끼리 인덱스 맵 공유(연산 재사용)
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))  # attention 앞/뒤 정규화
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))  # MLP 앞/뒤 정규화
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),  # mlp_ratio 배 확장
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        # stochastic depth: 학습 시 잔차 경로를 확률적으로 0으로(정규화 효과).
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        # 1) CPE 잔차: 위치 인코딩을 더함.
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        # 2) Attention 잔차(+ pre/post norm).
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)  # pre-norm
        point = self.drop_path(self.attn(point))  # serialized attention (+ drop path)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)  # post-norm

        # 3) MLP(FFN) 잔차(+ pre/post norm).
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        # 갱신된 feat 를 희소 텐서에도 반영(다음 SpConv/블록과 일관성 유지).
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    """[한글] U-Net 인코더의 다운샘플 단계.
    직렬화 코드의 하위 비트를 잘라(>> pooling_depth*3) 같은 상위코드를 공유하는 점들을 한 클러스터로 묶고,
    클러스터별로 feat 를 reduce(max 등), coord 를 평균하여 점 개수를 stride 배만큼 줄인다.
    serialization 의 비트 구조 덕분에 코드 시프트만으로 공간적 voxel 풀링이 구현된다.
    원복(unpooling)을 위해 cluster(역매핑)와 부모 point 를 기록해 둔다(traceable)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",          # 클러스터 내 feat 집계 방식
        shuffle_orders=True,
        traceable=True,  # record parent and cluster  (unpooling 용 역정보 저장)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8  (2의 거듭제곱만 허용)
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)  # 채널 확장 투영(풀링 전 적용)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        # 풀링 깊이: stride 에 해당하는 비트수. stride=2 -> 1.
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0  # 깊이 초과 시 풀링 생략(안전장치)
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        # 코드 하위 pooling_depth*3 비트를 버려(>>) 상위 코드만 남김 -> 같은 상위코드=같은 voxel 클러스터.
        code = point.serialized_code >> pooling_depth * 3
        # 첫 order 코드 기준으로 고유 클러스터 식별. cluster: 각 점->클러스터 인덱스, counts: 클러스터 크기.
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)  # 점들을 클러스터 순으로 정렬한 인덱스
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])  # segment 경계 포인터
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]  # 각 클러스터의 대표(첫) 점 인덱스
        # generate down code, order, inverse
        # 다운샘플된 점들의 새 코드/순서/역매핑 생성(상위 serialization 정보 갱신).
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])  # order 축 무작위 셔플
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        # 다운샘플 결과 Point 구성: feat 는 클러스터별 reduce, coord 는 평균.
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),  # 투영 후 클러스터별 집계 -> 다운샘플 feat
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),  # 클러스터 점들의 평균 좌표
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,  # 격자좌표도 시프트로 다운스케일
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,  # 직렬화 깊이 감소
            batch=point.batch[head_indices],  # 대표 점의 배치 인덱스 승계
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition  # 도메인 조건 전파
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster  # unpooling 시 점->클러스터 복원용
            point_dict["pooling_parent"] = point      # 스킵 연결용 부모(고해상도) point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()  # 다운샘플 점들로 새 희소 텐서 준비
        return point


class SerializedUnpooling(PointModule):
    """[한글] U-Net 디코더의 업샘플 단계. SerializedPooling 의 역연산.
    풀링 때 저장한 부모(고해상도) point 와 cluster 역매핑을 이용해, 저해상도 특징을
    부모 점 개수로 펼치고(point.feat[inverse]) 인코더 스킵 특징과 더해 해상도를 복원한다."""

    def __init__(
        self,
        in_channels,
        skip_channels,   # 인코더 같은 단계의 스킵 연결 채널
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))       # 저해상도(디코더) 특징 투영
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))  # 스킵(인코더) 특징 투영

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()   # 풀링 시 저장된 고해상도 부모 필요
        assert "pooling_inverse" in point.keys()  # 점->클러스터 역매핑 필요
        parent = point.pop("pooling_parent")  # 고해상도 인코더 특징(스킵)
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)        # 저해상도 특징 투영
        parent = self.proj_skip(parent)  # 스킵 특징 투영
        # 저해상도 특징을 inverse 로 부모 점 개수만큼 펼쳐 스킵 특징에 더함(업샘플 + skip).
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent  # 복원된 고해상도 point 반환


class Embedding(PointModule):
    """[한글] 입력 임베딩(stem). 원시 in_channels 특징을 희소 3D 컨볼루션(SubMConv3d)으로
    embed_channels 차원에 매핑하여 백본 입력 특징을 만든다. 트랜스포머 단계 진입 전 1회 적용."""

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )  # 5x5x5 희소 컨볼루션으로 초기 특징 추출
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class PointTransformerV3(PointModule):
    """[한글] PTv3 백본 본체. U-Net 형태로 인코더 5단계(다운샘플)와 디코더 4단계(업샘플)를 쌓는다.
    각 단계는 SerializedPooling/Unpooling + 여러 트랜스포머 Block 으로 구성된다.

    GaussMOT 에서는 radar_encoder 가 이 클래스를 feature extractor 로 호출하여,
    레이더 포인트의 per-point feature 를 디코더 최종 해상도로 받아 Gaussian 생성에 쓴다.
    cls_mode=True 면 디코더 없이 인코더 최저해상도 특징만 반환(분류용)."""

    def __init__(
        self,
        in_channels=6,  # 입력 per-point 특징 차원
        order=("z", "z-trans", "hilbert", "hilbert-trans"),  # 사용할 직렬화 곡선 목록
        stride=(2, 2, 2, 2),  # 인코더 단계별 다운샘플 배수
        enc_depths=(2, 2, 2, 6, 2),   # 인코더 단계별 Block 수
        enc_channels=(32, 64, 128, 256, 512),  # 인코더 단계별 채널
        enc_num_head=(2, 4, 8, 16, 32),        # 인코더 단계별 attention head 수
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),  # 인코더 단계별 윈도우 크기
        dec_depths=(2, 2, 2, 2),       # 디코더 단계별 Block 수
        dec_channels=(64, 64, 128, 256),  # 디코더 단계별 채널
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)  # 인코더 단계 수(=5)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        # 단계 수가 각 하이퍼파라미터 길이와 일관되는지 검증.
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        # BatchNorm: pdnorm_bn 이면 도메인별 PDNorm 으로 감싸고, 아니면 표준 BN1d 사용.
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # LayerNorm: 트랜스포머 블록 정규화. pdnorm_ln 이면 도메인별 PDNorm, 아니면 표준 LayerNorm.
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU  # 전 모듈 공통 활성화

        # 입력 임베딩(stem): in_channels -> 첫 인코더 채널.
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        # drop_path 비율을 전체 블록에 0->drop_path 선형 증가로 분배(stochastic depth 스케줄).
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):  # 인코더 단계 루프
            # 이 단계에 해당하는 drop_path 구간 슬라이스.
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                # 첫 단계 제외 각 단계 시작에 다운샘플(풀링) 추가: 채널 확장 + 점 수 감소.
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):  # 단계 내 트랜스포머 블록들
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),  # 블록마다 직렬화 곡선을 번갈아 사용
                        cpe_indice_key=f"stage{s}",       # 단계별 SpConv 인덱스 맵 공유
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")  # 단계 모듈 등록

        # decoder
        # cls_mode 면 디코더 없이 인코더 출력만 사용(분류). 아니면 업샘플 디코더 구성.
        if not self.cls_mode:
            # 디코더용 drop_path 스케줄.
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            # 디코더 입력 채널 리스트에 인코더 최종 채널(최저해상도)을 끝에 붙임.
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):  # 저해상도->고해상도 역순
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()  # 역순 단계에 맞춰 drop_path 도 뒤집음
                dec = PointSequential()
                # 단계 시작에 업샘플(언풀링): 저해상도 특징을 펼쳐 인코더 스킵과 합침.
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],  # 한 단계 저해상도 입력
                        skip_channels=enc_channels[s],    # 같은 단계 인코더 스킵
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):  # 디코더 단계 내 트랜스포머 블록들
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",  # 인코더와 같은 단계 키로 인덱스 맵 공유
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")  # 디코더 단계 등록

    def forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset

        [한글] 전체 forward 흐름:
        1) dict -> Point 구조체로 래핑(offset/batch 자동 보완).
        2) serialization: 점들을 z-order/Hilbert 1D 코드/순서로 인코딩(윈도우 attention 준비).
        3) sparsify: SpConv 용 희소 텐서 준비.
        4) embedding(stem) -> encoder(5단계 다운샘플) -> (cls 아니면) decoder(4단계 업샘플).
        반환된 Point.feat 가 per-point feature(레이더 인코더가 이를 사용).
        """
        point = Point(data_dict)  # dict -> Point(offset/batch 자동 생성)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)  # 직렬화
        point.sparsify()  # 희소 텐서 준비

        point = self.embedding(point)  # 입력 임베딩(stem)
        point = self.enc(point)        # 인코더(다운샘플 + 트랜스포머)
        if not self.cls_mode:
            point = self.dec(point)    # 디코더(업샘플 + 스킵 + 트랜스포머)
        return point  # per-point feature 가 담긴 Point 반환

