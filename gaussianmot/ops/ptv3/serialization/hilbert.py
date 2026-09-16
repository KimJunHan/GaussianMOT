"""
Hilbert Order
Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
Please cite our work if the code is helpful to you.
"""

import torch


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified."""

    # If we're shifting the whole thing, just return zeros.
    # 시프트 양이 비트열 길이 이상이면 모든 비트가 밀려나가므로 전부 0 반환.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the padding pattern.
    # padding = [(0,0)] * len(binary.shape)
    # padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    # 해당 axis에서 마지막 k개 비트를 잘라내는 슬라이스 패턴을 만든다(:, ..., :-k).
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    # 끝쪽 k비트를 버린 뒤, 앞쪽(상위 비트 자리)에 0을 k개 패딩 → 우측 시프트와 동일 효과.
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    """Convert an array of binary values into Gray codes.

    This uses the classic X ^ (X >> 1) trick to compute the Gray code.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     axis: The axis along which to compute the gray code. Default=-1.

    Returns:
    --------
     Returns an ndarray of Gray codes.
    """
    shifted = right_shift(binary, axis=axis)  # X >> 1 (한 비트 우측 시프트)

    # Do the X ^ (X >> 1) trick.
    # 그레이코드 = X XOR (X >> 1). 인접 값이 1비트만 달라지도록 하는 고전적 변환.
    gray = torch.logical_xor(binary, shifted)

    return gray


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    # 그레이→이진 역변환은 시프트 폭을 절반씩 줄여가며 누적 XOR(prefix-XOR)로 수행한다.
    # 시작 시프트 폭 = 2**(ceil(log2(비트수))-1), 즉 비트수보다 작은 최대 2의 거듭제곱.
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        # 현재 시프트 폭만큼 우측 시프트한 값과 XOR → prefix-XOR 한 단계 적용.
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")  # 시프트 폭을 절반(정수 나눗셈)으로 축소
    return gray


def encode(locs, num_dims, num_bits):
    """Decode an array of locations in a hypercube into a Hilbert integer.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     locs - An ndarray of locations in a hypercube of num_dims dimensions, in
            which each dimension runs from 0 to 2**num_bits-1.  The shape can
            be arbitrary, as long as the last dimension of the same has size
            num_dims.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of uint64 integers with the same shape as the
     input, excluding the last dimension, which needs to be num_dims.
    """

    # Keep around the original shape for later.
    orig_shape = locs.shape  # 원본 입력 shape 보관(반환 형태 맞추는 용도는 아님; 검증/참고용)
    # 비트 ↔ 바이트 변환용 마스크: [1,2,4,...,128] (각 비트 자리값).
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)  # 역순 [128,...,2,1] : MSB-우선 비트 추출용

    if orig_shape[-1] != num_dims:
        raise ValueError(
            """
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits, num_dims * num_bits)
        )

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    # 좌표 int64를 8개의 uint8 바이트로 쪼개 (N, num_dims, 8) 형태로 본다.
    # flip(-1): 리틀엔디언 바이트 순서를 뒤집어 빅엔디언(MSB가 앞)으로 정렬.
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # Now turn these into bits and truncate to num_bits.
    # 각 바이트를 8비트로 펼치고(bitwise_and + ne(0)), 차원별 비트열로 만든 뒤
    # 마지막 num_bits개 비트만 취해 실제 사용 비트폭으로 자른다.
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)  # 각 바이트에 [128..1] 마스크 AND → 비트별 존재 여부
        .ne(0)                          # 0이 아니면 해당 비트가 1
        .byte()
        .flatten(-2, -1)[..., -num_bits:]  # 8비트들을 펼친 뒤 하위 num_bits개만 사용
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    # Skilling의 Hilbert 알고리즘: 상위 비트부터 차례로 좌표 비트들을 회전/교환하여
    # (좌표→Hilbert) 변환을 수행한다. 아래 이중 루프가 핵심 변환부.
    for bit in range(0, num_bits):  # 상위 비트(0)부터 하위로 진행
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):  # 각 차원에 대해
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]  # 현재 (차원,비트) 위치 비트가 켜져 있는지

            # Where this bit is on, invert the 0 dimension for lower bits.
            # 이 비트가 켜진 샘플은 0번 차원의 하위 비트들을 반전(XOR 1).
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            # 이 비트가 꺼진 경우, 0번 차원과 현재 차원의 하위 비트가 다른 자리를 골라 맞교환(swap).
            # to_flip = (비트 꺼짐) AND (두 차원 하위비트 XOR) → 교환해야 할 비트 위치.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            # 두 차원에 동일 to_flip을 XOR하면 서로의 비트가 교환(swap)되는 효과.
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    # (N, num_dims, num_bits) → 비트/차원 축을 바꿔 (N, num_bits*num_dims)로 평탄화.
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Convert Gray back to binary.
    # 위 변환 결과(그레이코드 표현)를 이진수로 되돌려 최종 Hilbert 비트열 획득.
    hh_bin = gray2binary(gray)

    # Pad back out to 64 bits.
    # 사용 비트수(num_bits*num_dims)를 64비트로 맞추기 위해 앞쪽(상위)에 0 패딩.
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)

    # Convert binary values into uint8s.
    # 64비트를 8바이트(8×8)로 재배열하고, 각 바이트를 비트가중합(* bitpack_mask)으로 uint8화.
    # flip(-1): 다시 바이트 순서를 뒤집어 엔디언 정렬.
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )

    # Convert uint8s into uint64s.
    # 8개 uint8 바이트를 하나의 int64(=Hilbert 코드)로 재해석.
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()

    return hh_uint64


def decode(hilberts, num_dims, num_bits):
    """Decode an array of Hilbert integers into locations in a hypercube.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     hilberts - An ndarray of Hilbert integers.  Must be an integer dtype and
                cannot have fewer bits than num_dims * num_bits.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of unsigned integers with the same shape as hilberts
     but with an additional dimension of size num_dims.
    """

    if num_dims * num_bits > 64:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits)
        )

    # Handle the case where we got handed a naked integer.
    hilberts = torch.atleast_1d(hilberts)  # 스칼라가 와도 최소 1차원 텐서로 승격

    # Keep around the shape for later.
    orig_shape = hilberts.shape  # 마지막에 좌표 차원을 덧붙여 복원할 원본 shape 보관
    bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)  # [1,2,4,...,128] 비트 자리값
    bitpack_mask_rev = bitpack_mask.flip(-1)  # 역순 [128..1] : MSB-우선 비트 추출용

    # Treat each of the hilberts as a s equence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    # Hilbert 코드(int64)를 8개의 uint8 바이트로 분해, flip으로 엔디언 정렬.
    hh_uint8 = (
        hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    # 바이트들을 비트열로 펼치고, 실제 필요한 하위 num_dims*num_bits개 비트만 취한다.
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)  # 각 바이트에 [128..1] AND → 비트별 추출
        .ne(0)                          # 0이 아니면 비트=1
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]  # 사용 비트수만큼만 보존
    )

    # Take the sequence of bits and Gray-code it.
    # 이진 비트열을 그레이코드로 변환(인코드의 gray2binary와 짝이 되는 역과정 준비).
    gray = binary2gray(hh_bits)

    # There has got to be a better way to do this.
    # I could index them differently, but the eventual packbits likes it this way.
    # (N, num_bits, num_dims)로 재배열 후 비트/차원 축을 바꿔 (N, num_dims, num_bits) 형태로.
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    # 인코드와 반대 방향(하위 비트→상위, 마지막 차원→0)으로 회전/교환을 되돌려 좌표 복원.
    for bit in range(num_bits - 1, -1, -1):  # 비트 역순 진행
        # Iterate backwards through the dimensions.
        for dim in range(num_dims - 1, -1, -1):  # 차원 역순 진행
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]  # 현재 (차원,비트) 비트가 켜졌는지

            # Where this bit is on, invert the 0 dimension for lower bits.
            # 비트가 켜진 샘플은 0번 차원 하위 비트들을 반전(인코드 반전의 역연산).
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            # 비트가 꺼진 경우 0번 차원과 현재 차원의 다른 하위 비트를 골라 맞교환(swap).
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            # 동일 to_flip을 양 차원에 XOR → 비트 교환.
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Pad back out to 64 bits.
    # 차원별 비트(num_bits)를 64비트로 맞추기 위해 상위에 0 패딩.
    extra_dims = 64 - num_bits
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

    # Now chop these up into blocks of 8.
    # 64비트를 8바이트(8×8) 블록으로 잘라 (N, num_dims, 8, 8) 형태로 재배열. flip으로 엔디언 정렬.
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))

    # Take those blocks and turn them unto uint8s.
    # from IPython import embed; embed()
    # 각 8비트 블록을 비트가중합(* bitpack_mask)으로 uint8 값으로 환원.
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)

    # Finally, treat these as uint64s.
    # 8바이트를 다시 int64 좌표값으로 재해석.
    flat_locs = locs_uint8.view(torch.int64)

    # Return them in the expected shape.
    # 원본 shape에 좌표 차원(num_dims)을 덧붙인 형태로 반환.
    return flat_locs.reshape((*orig_shape, num_dims))
