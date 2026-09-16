#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# NamedTuple: 래스터라이저 설정(카메라/투영 파라미터)을 불변 튜플로 묶기 위해 사용
from typing import NamedTuple
# nn.Module 상속(GaussianRasterizer)을 위해 PyTorch 신경망 모듈 임포트
import torch.nn as nn
# 텐서 연산 및 autograd.Function 정의를 위해 torch 임포트
import torch
# _C: 컴파일된 CUDA 확장 모듈(rasterize_gaussians 등 실제 GPU splat 커널). 여기서는 호출만 함
from . import _C

# 디버그 시 인자 튜플을 CPU로 깊은 복사하는 헬퍼(GPU 텐서 손상 전 스냅샷 보존용)
def cpu_deep_copy_tuple(input_tuple):
    # 튜플의 각 원소가 텐서면 cpu()로 옮기고 clone()으로 복제, 텐서가 아니면 그대로 둠
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    # 복사된 리스트를 다시 튜플로 묶어 반환(원본 args와 동일 구조)
    return tuple(copied_tensors)

# 외부에서 호출하는 래스터화 진입 함수: autograd Function을 통해 forward/backward 연결
def rasterize_gaussians(
    means3D,          # [N,3] 각 Gaussian의 3D 중심 좌표(월드 좌표)
    means2D,          # [N,3] 스크린 공간 평균(2D gradient 누적용 placeholder)
    sh,               # Spherical Harmonics 계수(여기선 미사용, 빈 텐서 전달)
    colors_precomp,   # [N,C] 미리 계산된 색/feature 값(GaussMOT에서 splat할 feature)
    opacities,        # [N,1] 각 Gaussian의 불투명도(알파)
    scales,           # [N,3] 각 Gaussian의 축별 스케일(공분산 구성용)
    rotations,        # [N,4] 각 Gaussian의 회전 쿼터니언(공분산 구성용)
    cov3Ds_precomp,   # [N,6] 미리 계산된 3D 공분산(scales/rotations 대신 사용 가능)
    raster_settings,  # GaussianRasterizationSettings 설정 namedtuple
):
    # autograd.Function.apply 호출 → forward 실행 및 backward 그래프 등록
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

# Gaussian splat 렌더링을 위한 커스텀 autograd Function(forward=CUDA 렌더, backward=gradient 역전파)
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,              # autograd 컨텍스트: backward에서 쓸 텐서/값 저장소
        means3D,          # [N,3] Gaussian 3D 중심
        means2D,          # [N,3] 2D 스크린 평균(forward에선 직접 미사용, backward gradient용)
        sh,               # SH 계수(미사용, 빈 텐서)
        colors_precomp,   # [N,C] splat할 색/feature
        opacities,        # [N,1] 불투명도
        scales,           # [N,3] 스케일
        rotations,        # [N,4] 회전 쿼터니언
        cov3Ds_precomp,   # [N,6] 사전 계산 3D 공분산
        raster_settings,  # 렌더 설정 namedtuple
    ):

        # C++ 라이브러리가 기대하는 순서/형태로 인자 재구성(positional 인자 순서가 ext.cpp 시그니처와 일치해야 함)
        args = (
            raster_settings.bg,                  # 배경색 텐서(빈 픽셀 채울 색)
            means3D,                             # Gaussian 3D 중심
            colors_precomp,                      # 사전 계산 색/feature
            opacities,                           # 불투명도(알파)
            scales,                              # 스케일
            rotations,                           # 회전 쿼터니언
            raster_settings.scale_modifier,      # 전역 스케일 배율(공분산 크기 조정)
            cov3Ds_precomp,                      # 사전 계산 3D 공분산
            raster_settings.viewmatrix,          # 월드→카메라 뷰 변환 행렬
            raster_settings.projmatrix,          # 카메라→클립 투영 행렬
            # raster_settings.tanfovx,           # (미사용) 수평 화각 tan — 본 BEV 투영에선 projmatrix로 대체
            # raster_settings.tanfovy,           # (미사용) 수직 화각 tan
            raster_settings.image_height,        # 출력 이미지 높이(H, BEV 격자 행 수)
            raster_settings.image_width,         # 출력 이미지 너비(W, BEV 격자 열 수)
            sh,                                  # SH 계수(빈 텐서)
            raster_settings.sh_degree,           # SH 차수(색을 SH로 계산할 때 사용, 여기선 0)
            # raster_settings.campos,            # (미사용) 카메라 위치 — SH 방향 색 계산용이라 불필요
            raster_settings.prefiltered,         # 프러스텀 컬링 사전수행 여부 플래그
            raster_settings.debug                # 디버그 모드 플래그(에러 시 스냅샷 저장)
        )

        # C++/CUDA 래스터라이저 호출
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # 커널이 인자를 손상시키기 전에 CPU로 복사해 보존
            try:
                # CUDA splat 실행 → 렌더 개수/색 이미지/반경 및 backward용 내부 버퍼들 반환
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                # 실패 시 보존해둔 인자 스냅샷을 파일로 덤프(디버깅 제출용)
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex  # 예외 재전파
        else:
            # 비디버그 경로: 곧바로 CUDA 래스터라이저 실행
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # backward에서 필요한 값/텐서를 컨텍스트에 저장
        ctx.raster_settings = raster_settings    # 설정 namedtuple 보관(텐서 아님이라 별도 속성)
        ctx.num_rendered = num_rendered          # 렌더된 Gaussian 개수(backward 인자로 필요)
        # backward gradient 계산에 필요한 텐서들 저장(색/중심/스케일/회전/공분산/반경/SH/내부버퍼들)
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        # 최종 렌더 결과: color=[C,H,W] feature 이미지, radii=각 Gaussian 스크린 반경
        return color, radii

    @staticmethod
    # grad_out_color: 출력 color 이미지에 대한 상류 gradient, 두 번째(_) = radii용 gradient(미사용)
    def backward(ctx, grad_out_color, _):

        # forward에서 컨텍스트에 저장해둔 값들 복원
        num_rendered = ctx.num_rendered          # 렌더된 Gaussian 개수
        raster_settings = ctx.raster_settings    # 렌더 설정 namedtuple
        # save_for_backward로 저장한 텐서들을 동일 순서로 언팩
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # C++ backward 메서드가 기대하는 순서대로 인자 재구성
        args = (raster_settings.bg,              # 배경색(forward와 동일)
                means3D,                         # Gaussian 3D 중심
                radii,                           # forward에서 계산된 스크린 반경
                colors_precomp,                  # 사전 계산 색/feature
                scales,                          # 스케일
                rotations,                       # 회전 쿼터니언
                raster_settings.scale_modifier,  # 전역 스케일 배율
                cov3Ds_precomp,                  # 사전 계산 3D 공분산
                raster_settings.viewmatrix,      # 뷰 행렬
                raster_settings.projmatrix,      # 투영 행렬
                # raster_settings.tanfovx,       # (미사용) 수평 화각 tan
                # raster_settings.tanfovy,       # (미사용) 수직 화각 tan
                grad_out_color,                  # 출력 이미지에 대한 상류 gradient(역전파 시작점)
                sh,                              # SH 계수(빈 텐서)
                raster_settings.sh_degree,       # SH 차수
                # raster_settings.campos,        # (미사용) 카메라 위치
                geomBuffer,                      # forward 기하 중간버퍼(역전파 재사용)
                num_rendered,                    # 렌더 개수
                binningBuffer,                   # forward 타일 비닝 버퍼(역전파 재사용)
                imgBuffer,                       # forward 픽셀 누적 버퍼(역전파 재사용)
                raster_settings.debug)           # 디버그 플래그

        # backward 메서드를 호출해 관련 텐서들의 gradient 계산
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # 손상 전 인자 CPU 복사(스냅샷 보존)
            try:
                # CUDA backward 커널: 입력별 gradient 반환
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                # 실패 시 인자 스냅샷 덤프 후 예외 재전파
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            # 비디버그 경로: 바로 backward 커널 실행
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # forward 입력 인자 순서(means3D, means2D, sh, ... )와 1:1 대응하는 gradient 튜플 구성
        grads = (
            grad_means3D,           # means3D에 대한 gradient
            None, # grad_means2D,   # means2D는 backward로 학습 안 함(None)
            None, # grad_sh,        # sh 미사용이므로 gradient 없음(None)
            grad_colors_precomp,    # colors_precomp(=feature)에 대한 gradient → feature 학습 경로
            # None,
            grad_opacities,         # opacities에 대한 gradient
            None, # grad_scales,    # scales gradient 전파 안 함(None)
            None, # grad_rotations, # rotations gradient 전파 안 함(None)
            grad_cov3Ds_precomp,    # cov3Ds_precomp에 대한 gradient
            None,                   # raster_settings는 텐서 아님 → gradient 없음(None)
        )

        # forward 입력 개수와 동일한 길이의 gradient 튜플 반환
        return grads

# 래스터화 카메라/투영 설정을 담는 불변 namedtuple(GaussianRenderer가 채워 전달)
class GaussianRasterizationSettings(NamedTuple):
    image_height: int           # 출력 이미지(BEV 격자) 높이 H
    image_width: int            # 출력 이미지(BEV 격자) 너비 W
    tanfovx : float             # 수평 화각 tan(현재 forward args에선 주석처리, projmatrix 사용)
    tanfovy : float             # 수직 화각 tan(동일하게 미사용)
    bg : torch.Tensor           # 배경색 텐서(렌더 안 된 픽셀 채움)
    scale_modifier : float      # 전역 스케일 배율(Gaussian 크기 일괄 조정)
    viewmatrix : torch.Tensor   # 월드→카메라 뷰 변환 행렬
    projmatrix : torch.Tensor   # 카메라→클립 투영 행렬(BEV 투영 정의)
    sh_degree : int             # SH 차수(0이면 색을 SH로 계산 안 함)
    campos : torch.Tensor       # 카메라 위치(SH 방향 색 계산용, 현재 미사용)
    prefiltered : bool          # 프러스텀 컬링 사전수행 여부
    debug : bool                # 디버그 모드(에러 시 스냅샷 덤프)

# GaussMOT의 render.py에서 감싸 사용하는 래스터라이저 모듈(nn.Module)
class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()           # nn.Module 초기화
        self.raster_settings = None  # 설정은 set_raster_settings로 나중에 주입

    def markVisible(self, positions):
        # 카메라 프러스텀 컬링 기준으로 보이는 점들을 boolean 마스크로 표시(no_grad: 학습 불필요)
        with torch.no_grad():
            raster_settings = self.raster_settings  # 현재 설정 참조
            # CUDA mark_visible 커널: 위치와 뷰/투영 행렬로 가시성 판정
            visible = _C.mark_visible(
                positions,                    # [N,3] 점 위치
                raster_settings.viewmatrix,   # 뷰 행렬
                raster_settings.projmatrix)   # 투영 행렬

        return visible  # [N] boolean 가시성 마스크

    # 실제 렌더 호출: 색(shs/colors_precomp)과 형상(scale·rotation/cov3D)을 받아 splat
    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):

        raster_settings = self.raster_settings  # 주입된 설정 참조

        # 색 입력 검증: SHs와 precomputed colors 중 정확히 하나만 제공해야 함
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        # 형상 입력 검증: (scale+rotation 쌍) 또는 (사전 계산 cov3D) 중 정확히 하나만 제공해야 함
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 미제공 인자는 빈 텐서로 채워 CUDA ext의 positional 시그니처를 맞춤
        if shs is None:
            shs = torch.Tensor([])            # SH 미사용 → 빈 텐서
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]) # 사전 계산 색 미사용 → 빈 텐서

        if scales is None:
            scales = torch.Tensor([])         # 스케일 미사용 → 빈 텐서
        if rotations is None:
            rotations = torch.Tensor([])      # 회전 미사용 → 빈 텐서
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])  # 사전 계산 공분산 미사용 → 빈 텐서

        # C++/CUDA 래스터화 루틴 호출(autograd Function 경유)
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    # 외부(GaussianRenderer)에서 카메라/투영 설정을 주입하는 세터
    def set_raster_settings(self, settings):
        self.raster_settings = settings