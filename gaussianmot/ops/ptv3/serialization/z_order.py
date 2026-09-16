# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:
    """Z-order(Morton) 코드 인코드/디코드용 사전계산 룩업테이블(LUT) 관리 클래스.

    한 바이트(8비트) 단위로 미리 인터리브 결과를 계산해 두면, for-loop 비트연산을
    텐서 인덱싱 한 번으로 대체할 수 있어 훨씬 빠르다. 디바이스별로 LUT를 캐싱한다.
    """
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)  # 0..255: 한 축 1바이트의 모든 값
        r512 = torch.arange(512, dtype=torch.int64)  # 0..511: 디코드용 9비트(=3비트×3축) 키 범위
        zero = torch.zeros(256, dtype=torch.int64)   # 다른 축을 0으로 둘 때 쓰는 영벡터
        device = torch.device("cpu")  # LUT는 우선 CPU에서 생성·보관(필요 시 GPU로 복제)

        # 인코드 LUT: 각 축의 8비트 값을 Morton 위치로 펼친(인터리브된) 결과를 축별로 저장.
        #  - EX: x축만 채운 경우(다른 축 0) → x비트가 비트2,5,8,...에 배치된 결과
        #  - EY: y축만 채운 경우 → y비트가 비트1,4,7,...에 배치
        #  - EZ: z축만 채운 경우 → z비트가 비트0,3,6,...에 배치
        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),  # EX
                self.xyz2key(zero, r256, zero, 8),  # EY
                self.xyz2key(zero, zero, r256, 8),  # EZ
            )
        }
        # 디코드 LUT: 9비트 키 → (x,y,z) 역인터리브 결과를 미리 계산.
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        # 요청 디바이스에 인코드 LUT가 없으면 CPU본을 해당 디바이스로 복사해 캐싱.
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        # 요청 디바이스에 디코드 LUT가 없으면 CPU본을 복사해 캐싱.
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        """(x,y,z) 정수좌표를 비트 인터리브하여 Morton 키로 만드는 기준(for-loop) 구현.

        각 비트 위치 i에서 x,y,z의 i번째 비트를 뽑아 키의 (3i+2, 3i+1, 3i) 위치에 배치한다.
        즉 출력 비트열은 ... x2 y2 z2 x1 y1 z1 x0 y0 z0 순서로 인터리브된다.
        """
        key = torch.zeros_like(x)  # 결과 키 누적용(0으로 초기화)
        for i in range(depth):
            mask = 1 << i  # i번째 비트만 1인 마스크
            key = (
                key
                | ((x & mask) << (2 * i + 2))  # x의 i번째 비트를 키의 (3i+2)번째 위치로 이동(i+2i+2 = 3i+2)
                | ((y & mask) << (2 * i + 1))  # y의 i번째 비트를 키의 (3i+1)번째 위치로 이동
                | ((z & mask) << (2 * i + 0))  # z의 i번째 비트를 키의 (3i+0)번째 위치로 이동
            )
        return key

    def key2xyz(self, key, depth):
        """Morton 키를 역인터리브하여 (x,y,z) 좌표로 분해하는 기준(for-loop) 구현.

        키의 (3i+2,3i+1,3i) 비트를 각각 x,y,z의 i번째 비트로 되돌린다(인코드의 역연산).
        """
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))  # 키 (3i+2)비트 → x의 i번째 비트로 복원
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))  # 키 (3i+1)비트 → y의 i번째 비트로 복원
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))  # 키 (3i+0)비트 → z의 i번째 비트로 복원
        return x, y, z


_key_lut = KeyLUT()  # 모듈 로드시 LUT를 한 번만 생성해 전역 재사용(매 호출 재계산 방지)


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    # 입력 텐서가 놓인 디바이스에 맞춰 축별 인코드 LUT 3개를 가져온다.
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()  # 비트연산을 위해 int64로 통일

    # 하위 바이트(0..7비트) 처리용 마스크: depth>8이면 한 바이트(255), 아니면 depth비트만.
    mask = 255 if depth > 8 else (1 << depth) - 1
    # 각 축 하위 8비트를 LUT로 인터리브한 뒤 OR로 합쳐 하위 Morton 키 생성.
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        # 9비트 이상이면 상위 바이트(8비트~)도 처리. (depth-8)비트만 남기는 마스크.
        mask = (1 << (depth - 8)) - 1
        # 각 축을 8비트 우측 시프트해 상위 부분을 LUT 인터리브 → 상위 Morton 키.
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        # 하위 8비트×3축 = 24비트를 차지하므로, 상위 키를 24비트 올려 하위 키와 OR 결합.
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        # 배치 인덱스를 48비트 위로 올려 OR 결합(좌표 키는 최대 16비트×3축=48비트 사용).
        key = b << 48 | key

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    # 입력 디바이스에 맞춰 축별 디코드 LUT 3개를 가져온다.
    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)  # 좌표 누적용 초기화

    b = key >> 48                    # 상위 48비트~ : 배치 인덱스 추출
    key = key & ((1 << 48) - 1)      # 하위 48비트만 남겨 순수 좌표 키만 유지

    # 키를 9비트(=3비트×3축) 청크 단위로 끊어 처리. 필요한 청크 수 n = ceil(depth/3).
    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511     # i번째 9비트 청크 추출(511 = 9비트 마스크)
        x = x | (DX[k] << (i * 3))   # LUT로 청크 디코드한 x조각(3비트)을 제자리(i*3)로 올려 누적
        y = y | (DY[k] << (i * 3))   # y조각 누적
        z = z | (DZ[k] << (i * 3))   # z조각 누적

    return x, y, z, b
