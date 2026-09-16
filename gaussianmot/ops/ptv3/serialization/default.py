import torch
# 실제 비트연산 구현은 z_order / hilbert 모듈에 있고, 여기서는 별칭(_) 붙여 가져와 래핑한다.
from .z_order import xyz2key as z_order_encode_   # Z-order(Morton) 인코더 원본 구현
from .z_order import key2xyz as z_order_decode_   # Z-order(Morton) 디코더 원본 구현
from .hilbert import encode as hilbert_encode_    # Hilbert 곡선 인코더 원본 구현
from .hilbert import decode as hilbert_decode_    # Hilbert 곡선 디코더 원본 구현


@torch.inference_mode()  # 직렬화 코드 계산은 학습 대상이 아니므로 autograd 추적을 끈다(속도/메모리 이득)
def encode(grid_coord, batch=None, depth=16, order="z"):
    """정수 격자좌표(grid_coord, shape (N,3))를 공간충전곡선 코드(int64, shape (N,))로 인코딩하는 통합 진입점.

    order 값에 따라 Z-order 또는 Hilbert 곡선을 선택한다. 공간적으로 가까운 점이
    1D 정수 순서에서도 가깝게 정렬되어, KNN 없이 윈도우 attention을 가능케 한다.

    Args:
        grid_coord: (N, 3) 정수 격자좌표 텐서. 각 축은 0..2**depth-1 범위로 가정.
        batch:      (N,) 배치 인덱스(옵션). 주어지면 상위 비트에 끼워 넣어 배치별 코드 분리.
        depth:      축당 비트 수(기본 16). 좌표 1축이 차지하는 비트폭.
        order:      "z"/"z-trans"/"hilbert"/"hilbert-trans" 중 하나.
    """
    # 허용된 곡선 종류인지 방어적으로 검증("-trans"는 x/y축을 맞바꾼 변형)
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        # 기본 Z-order(Morton) 인코딩
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        # 열 인덱싱 [1,0,2]로 x/y를 맞바꾼 뒤 Z-order 인코딩(곡선 방향 다양화)
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        # 기본 Hilbert 곡선 인코딩(Z-order보다 지역성 우수)
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        # x/y 축을 맞바꾼 뒤 Hilbert 인코딩(곡선 방향 다양화)
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()  # 비트시프트 위해 정수형(int64) 보장
        # 좌표 코드는 depth*3 비트를 사용하므로, 배치 인덱스를 그 상위 비트로 올려 OR 결합.
        # → 같은 배치끼리 코드값이 묶이고 배치 경계를 넘어 섞이지 않는다.
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()  # 디코딩도 추론 전용 연산이라 autograd 추적 비활성화
def decode(code, depth=16, order="z"):
    """공간충전곡선 코드(int64) → 정수 격자좌표(grid_coord)와 배치 인덱스로 복원."""
    assert order in {"z", "hilbert"}  # 디코드는 정방향(비-trans)만 지원
    # 상위 비트(depth*3 이상)는 배치 인덱스이므로 우측 시프트로 분리해 꺼낸다.
    batch = code >> depth * 3
    # 하위 depth*3 비트만 남기는 마스크를 AND 하여 순수 좌표 코드만 추출.
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """grid_coord (N,3) → Z-order(Morton) 코드 (N,)로 변환하는 얇은 래퍼."""
    # 세 축을 각각 분리하고 int64로 변환(비트 인터리브 연산을 위해).
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    # (여기서는 배치(b)를 None으로 막고, 배치 결합은 상위 encode()/Point 클래스가 담당)
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    """Z-order 코드 (N,) → grid_coord (N,3)로 복원하는 얇은 래퍼."""
    # 원본 디코더가 x,y,z를 따로 돌려주므로 마지막 축으로 stack 해 (N,3)으로 합친다.
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    """grid_coord (N,3) → Hilbert 코드로 변환. depth가 곧 축당 비트수(num_bits)."""
    # Hilbert 구현은 차원수(num_dims)와 비트수(num_bits) 인자명을 쓰므로 매핑해 호출.
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    """Hilbert 코드 → grid_coord (N,3)로 복원. depth=축당 비트수."""
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
