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

# 패키지 빌드/설치 진입점 함수
from setuptools import setup
# CUDA 확장(.cu) 컴파일을 위한 PyTorch 빌드 도구: CUDAExtension(확장 정의), BuildExtension(빌드 명령)
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
# 경로 조작(glm include 디렉터리 절대경로 계산)에 사용
import os
# setup.py 디렉터리 절대경로 계산(결과 미사용; include 경로는 아래에서 다시 계산)
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",            # 설치될 파이썬 패키지 이름
    # packages=['diff_gaussian_rasterization'],    # (비활성) 순수 파이썬 패키지 목록 — 여기선 확장만 빌드
    ext_modules=[                                  # 빌드할 C++/CUDA 확장 모듈 목록
        CUDAExtension(
            name="diff_gaussian_rasterization._C", # 생성될 확장 모듈명(=__init__.py가 import하는 _C)
            sources=[                              # 컴파일할 소스 파일들
            "cuda_rasterizer/rasterizer_impl.cu",  # 래스터라이저 핵심 구현(타일 정렬/비닝 등)
            "cuda_rasterizer/forward.cu",          # forward splat 커널(색/feature 누적)
            "cuda_rasterizer/backward.cu",         # backward gradient 커널
            "rasterize_points.cu",                 # 파이썬↔CUDA 브리지(텐서 입출력 정리)
            "ext.cpp"],                            # pybind11 바인딩(파이썬에 함수 노출)
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})  # (비활성) glm include만 주던 이전 버전
            # nvcc 플래그: -Xcompiler로 host 컴파일러에 -fno-gnu-unique 전달(심볼 중복/언로드 이슈 회피), 그리고 번들된 glm 수학 라이브러리 include 경로 추가
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={                                     # 커스텀 빌드 명령 매핑
        'build_ext': BuildExtension                # build_ext 단계를 PyTorch BuildExtension으로 대체(nvcc/혼합 컴파일 처리)
    }
)
