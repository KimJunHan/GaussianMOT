# PTv3(Point Transformer V3) 직렬화 패키지 진입점.
# default.py 가 노출하는 공개 API(인코드/디코드 함수들)를 패키지 레벨로 re-export 한다.
# 이렇게 하면 외부에서 `from ...serialization import encode` 처럼 곧장 가져다 쓸 수 있다.
from .default import (
    encode,          # order("z"/"hilbert" 등)에 따라 적절한 공간충전곡선 코드로 인코딩하는 통합 진입점
    decode,          # 코드(int64) → 정수 격자좌표(grid_coord)로 복원하는 통합 진입점
    z_order_encode,  # Z-order(Morton) 인코딩 래퍼: grid_coord → Morton 코드
    z_order_decode,  # Z-order(Morton) 디코딩 래퍼: Morton 코드 → grid_coord
    hilbert_encode,  # Hilbert 곡선 인코딩 래퍼: grid_coord → Hilbert 코드
    hilbert_decode,  # Hilbert 곡선 디코딩 래퍼: Hilbert 코드 → grid_coord
)
