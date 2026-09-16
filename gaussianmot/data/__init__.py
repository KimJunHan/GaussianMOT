# === 표준 라이브러리 import 묶음 ===
import os          # 파일 경로 결합(os.path.join) 등에 사용
import sys         # sys.modules — 모듈 레지스트리 shim에서 자기 자신 모듈 반환용
import time        # NFS I/O 재시도 시 sleep 백오프용
import json        # scene 단위 JSON(샘플 목록) 파싱용
import pathlib     # 라벨 디렉터리 경로 처리 (pathlib.Path)
from pathlib import Path                  # Path 단축 import
from dataclasses import dataclass         # 이미지 augmentation 파라미터 dataclass용
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple  # 타입 힌트

# === 서드파티 라이브러리 import 묶음 ===
import cv2                                # BEV 마스크 polyline/fillPoly 래스터화
import numpy as np                        # 행렬/좌표 변환 핵심 연산
import torch                              # 텐서 변환 및 DataLoader
import torchvision                        # 이미지 ToTensor/Normalize transform
from torch.utils.data import Dataset      # NuScenesDataset 베이스 클래스
from torchvision.transforms import ToTensor  # 마스크 → 텐서 변환
from PIL import Image                     # 이미지 로딩(PIL)
from PIL.ImageTransform import AffineTransform  # ida_mat 기반 affine 워프
from pyquaternion import Quaternion       # 회전(quaternion) ↔ 행렬 변환
from scipy.spatial.transform import Rotation as R  # BEV augmentation Euler 회전 생성
import lightning as L                     # LightningDataModule
from nuscenes.nuscenes import NuScenes    # nuScenes devkit 메인 핸들
from nuscenes.utils.data_classes import Box, RadarPointCloud, LidarPointCloud  # GT 박스/포인트클라우드
from nuscenes.utils.geometry_utils import transform_matrix  # 센서→ego 변환 행렬 생성
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer  # HD맵 API
from shapely.geometry import LineString   # 맵 레이어 선분 → 마스크 변환용

import rootutils
# 프로젝트 루트(.project-root 표식)를 찾아 sys.path에 추가 — 어디서 실행해도 import 일관성 확보.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# ============================================================================
# nuscenes_map_api.py  (Shapely>2.0.0 compatibility fix)
# (inlined from gaussianmot/utils/nuscenes_map_api.py)
# ============================================================================
# Shapely 2.0+ 에서 MultiLineString 순회 방식이 바뀌어 devkit 원본이 깨짐 → 패치한 explorer.
class FixedMapExplorer(NuScenesMapExplorer):
    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
        # 여러 선분 묶음(MultiLineString)이면 .geoms 로 개별 선분 순회 (Shapely 2.0 호환 핵심).
        if lines.geom_type == 'MultiLineString':
            for line in lines.geoms:
                coords = np.asarray(list(line.coords), np.int32)   # 선분 좌표 → 정수 배열
                coords = coords.reshape((-1, 2))                    # (점수, 2) 형태로 정리
                cv2.polylines(mask, [coords], False, 1, 2)          # 마스크에 두께 2 선으로 그림
        else:
            # 단일 LineString 처리 (위와 동일 로직).
            coords = np.asarray(list(lines.coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)

        return mask  # 선이 그려진 마스크 반환


# NuScenesMap 을 상속하되 explorer 만 패치 버전으로 교체한 래퍼.
class FixedNuScenesMap(NuScenesMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)           # 원본 맵 로딩은 그대로 수행
        self.explorer = FixedMapExplorer(self)      # explorer 만 Shapely 호환 버전으로 교체


# ============================================================================
# common.py
# ============================================================================

INTERPOLATION = cv2.LINE_8  # cv2.fillPoly 의 lineType — BEV 마스크 래스터화 시 사용

# sin/cos(yaw) → z축 회전만 있는 Quaternion 생성 (GT 박스 yaw 복원용).
def sincos2quaternion(sin, cos):
    rotation = [
        [cos, sin, 0.0],   # x축 행
        [-sin, cos, 0.0],  # y축 행
        [0.0, 0.0, 1.0],   # z축 행 (회전 없음)
    ]
    return Quaternion(matrix=np.array(rotation))  # 3x3 회전행렬 → Quaternion

def get_split(split, dataset_name):
    # nuScenes 표준 split을 devkit에서 직접 가져온다 (splits/*.txt 파일 불필요).
    # train(700)/val(150)/mini_train(8)/mini_val(2) — txt 사본과 내용 동일함 확인됨.
    from nuscenes.utils.splits import create_splits_scenes
    return create_splits_scenes()[split]   # {split명: [scene명...]} 에서 해당 split 리스트 반환


# 미터 좌표(ego/lidar) → BEV 픽셀 좌표로 보내는 view 행렬(3x3 homogeneous) 생성.
def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):

    sh = h / h_meters   # 세로(전후 x) 방향 미터→픽셀 스케일
    sw = w / w_meters   # 가로(좌우 y) 방향 미터→픽셀 스케일

    # col = -sw*y + w/2, row = -sh*x + h/2 (BEV frame: col=-y, row=-x 규약과 일치).
    return np.float32([
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ])


# 회전행렬 R, 평행이동 t → 4x4 동차변환행렬. inv=True면 역변환을 반환.
def get_transformation_matrix(R, t, inv=False):

    pose = np.eye(4, dtype=np.float32)              # 4x4 단위행렬로 초기화
    pose[:3, :3] = R if not inv else R.T            # 회전부: 역이면 전치
    pose[:3, -1] = t if not inv else R.T @ -t       # 이동부: 역이면 -R^T t

    return pose


# (quaternion 회전, 평행이동) → 4x4 pose 행렬. flat=True면 yaw만 남긴 평탄화 회전.
def get_pose(rotation, translation, inv=False, flat=False):

    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]   # yaw 각도만 추출
        # yaw만 가진 z축 회전행렬 재구성 (pitch/roll 제거 — BEV 평면 가정).
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix       # 전체 3D 회전행렬

    t = np.array(translation, dtype=np.float32)        # 평행이동 벡터

    return get_transformation_matrix(R, t, inv=inv)    # 4x4 pose 조립


# ============================================================================
# transforms/augmentations.py
# ============================================================================

class RandomTransformBev:
    """
    Handles random data augmentation for Bird's Eye View (BEV) transformation matrices.
    """

    def __init__(
        self,
        bev_aug_conf: Optional[List[float]] = None,
        training: bool = True
    ) -> None:
        """
        Args:
            bev_aug_conf: Configuration list containing [tx, ty, tz, rx, ry, rz] coefficients.
            training: Whether to apply augmentation (True) or return identity (False).
        """
        self.training = training         # False면 augmentation 없이 단위행렬 반환
        self.bev_aug_conf = bev_aug_conf # [tx,ty,tz, rx,ry,rz] 한계 계수

    def get_random_ref_matrix(self) -> np.ndarray:
        """
        Generates a random reference transformation matrix using SciPy.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix (float32).
        """
        # Unpack configuration: first 3 are translation, last 3 are rotation
        coeffs = self.bev_aug_conf                          # [tx,ty,tz, rx,ry,rz] 계수
        trans_coeff = np.array(coeffs[:3], dtype=np.float32)  # 앞 3개 = 평행이동 한계(미터)
        rot_coeff = np.array(coeffs[3:], dtype=np.float32)    # 뒤 3개 = 회전 한계(도)

        # Initialize 4x4 Identity matrix
        mat = np.eye(4, dtype=np.float32)                   # 4x4 augmentation 행렬 초기화

        # 1. Translation
        # Logic: Generate random values in range [-1, 1) and scale by coefficients
        random_trans_noise = np.random.random(3).astype(np.float32) * 2 - 1  # [-1,1) 난수 3개
        mat[:3, 3] = random_trans_noise * trans_coeff       # 난수 × 한계 → 평행이동 성분

        # 2. Rotation
        # Logic: Generate random Euler angles (zyx) in range [-1, 1), scale, and convert to matrix
        random_rot_noise = np.random.random(3).astype(np.float32) * 2 - 1   # [-1,1) 난수 3개
        random_zyx = random_rot_noise * rot_coeff           # 난수 × 회전한계 → zyx 오일러각(도)

        mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()  # 오일러각 → 3x3 회전

        return mat  # 4x4 무작위 BEV augmentation 행렬

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transformation.

        Args:
            data_dict: A dictionary containing data to be augmented.

        Returns:
            Dict[str, Any]: The updated data dictionary with the transformation matrix.
        """
        # 기하 ego-warp 제약: 공유 augmentation 모드에서는 prev 프레임이 현재 프레임의
        # bev_augm을 미리 주입받는다. 이미 주입돼 있으면 새로 뽑지 않고 그대로 재사용한다.
        if 'bev_augm' not in data_dict:
            bev_augm = self.get_random_ref_matrix() if self.training else np.eye(4, dtype=np.float32)
            data_dict['bev_augm'] = bev_augm
        return data_dict



# 카메라 1대분 이미지 augmentation 파라미터를 담는 dataclass.
@dataclass
class AugmentationParams:
    """Holds configuration for a single image augmentation."""
    scale: float                            # resize 스케일 배율
    resize_dims: Tuple[int, int]  # (width, height)        # resize 후 크기
    crop: Tuple[int, int, int, int]  # (left, top, right, bottom)  # 하늘 영역 crop 박스
    flip: bool                              # 좌우 반전 여부
    rotate: float                           # 회전 각도(도)
    crop_zoom: Tuple[int, int, int, int]    # zoom용 crop 박스(좌,상,우,하)
    final_dims: Tuple[int, int]             # 최종 출력 (W, H)

    @property
    def ida_mat_args(self) -> Tuple:
        """Helper to unpack args for affinity matrix calculation."""
        # affinity 행렬 계산에 필요한 값만 추려서 반환 (crop[1]=하늘 crop 높이).
        return (
            self.scale, self.crop[1], self.crop_zoom,
            self.flip, self.rotate, self.final_dims
        )

# 카메라 이미지 augmentation 행렬(ida_mat)을 샘플링하는 transform.
class RandomTransformImage:
    def __init__(
        self,
        img_params: dict,                       # H/W/scale/crop 등 augmentation 설정
        training: bool = True,                  # True=무작위, False=결정적(평균값)
        transform: Optional[Any] = None,        # 사용하지 않으면 ToTensor 기본값
        max_range: float = 80.0,
        orig_img_size: Tuple[int, int] = (1600, 900),  # 원본 이미지 (W, H)
    ):
        self.img_params = img_params
        self.training = training
        self.transform = (
            transform if transform is not None
            else torchvision.transforms.ToTensor()
        )
        self.max_range = max_range
        self.orig_img_size = orig_img_size

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # 공유 augmentation 모드: prev 프레임이 현재 프레임의 ida_mat을 미리 주입받았으면
        # 새로 샘플링하지 않고 그대로 사용한다 (temporal augmentation 일관성).
        if 'ida_mat' in data_dict:
            return data_dict

        data_dict['ida_mat']: List[torch.Tensor] = []   # 카메라별 augmentation 행렬 누적 리스트
        num_images = len(data_dict['images'])            # 카메라 대수 (보통 6)

        for _ in range(num_images):
            aug_params = self.sample_params()            # 카메라 1대분 augmentation 파라미터 샘플
            ida_mat = self.get_affinity_matrix(aug_params, self.orig_img_size)  # → 3x3 affine 행렬
            data_dict['ida_mat'].append(torch.tensor(ida_mat))                  # 누적

        # 카메라별 3x3 행렬을 (N_cam, 3, 3) 텐서로 stack.
        data_dict['ida_mat'] = torch.stack(data_dict['ida_mat'], dim=0)
        return data_dict


    def sample_params(self) -> AugmentationParams:
            """Generates augmentation parameters based on current mode (train/eval)."""
            H, W = self.img_params["H"], self.img_params["W"]        # 원본 이미지 높이/너비
            final_dims = tuple(self.img_params["final_dim"][::-1]) # (W, H)  # 모델 입력 최종 크기

            if self.training:
                # --- 학습: 모든 파라미터를 범위 안에서 무작위 샘플 ---
                scale = np.random.uniform(*self.img_params["scale"])     # resize 스케일 무작위
                newW, newH = int(W * scale), int(H * scale)              # resize 후 크기
                resize_dims = (newW, newH)

                crop_h = int((1 - np.random.uniform(*self.img_params["crop_up_pct"])) * newH)  # 하늘 crop 높이
                crop = (0, crop_h, newW, newH)                          # crop 박스(좌,상,우,하)

                zoom = np.random.uniform(*self.img_params["zoom_lim"])  # zoom 배율 무작위
                crop_zoomh = ((newH - crop_h) * (1 - zoom)) // 2        # zoom용 세로 여백
                crop_zoomw = (newW * (1 - zoom)) // 2                   # zoom용 가로 여백

                crop_zoom = (                                           # zoom crop 박스
                    -crop_zoomw,
                    -crop_zoomh,
                    crop_zoomw + newW,
                    crop_zoomh + newH - crop_h,
                )

                flip = self.img_params["rand_flip"] and np.random.choice([0, 1])  # 좌우 반전 무작위
                rotate = np.random.uniform(*self.img_params["rot_lim"])           # 회전각 무작위
            else:
                # --- 평가: 결정적(범위 평균값, flip/rotate 없음) ---
                scale = np.mean(self.img_params["scale"])               # 스케일은 범위 평균 고정
                newW, newH = int(W * scale), int(H * scale)
                resize_dims = (newW, newH)

                crop_h = int((1 - np.mean(self.img_params["crop_up_pct"])) * newH)  # 하늘 crop 평균
                crop = (0, crop_h, newW, newH)

                # zoom = 1.0 implicitly
                crop_zoom = (0, 0, newW, newH - crop_h)                 # zoom 없음
                flip = False                                           # 반전 없음
                rotate = 0                                             # 회전 없음

            return AugmentationParams(
                scale=scale,
                resize_dims=resize_dims,
                crop=crop,
                flip=bool(flip),
                rotate=rotate,
                crop_zoom=tuple(map(int, crop_zoom)),
                final_dims=final_dims
        )

    def get_affinity_matrix(
        self,
        params: AugmentationParams,
        input_size: Tuple[int, int],
    ) -> np.ndarray:
            """Calculates the affine transformation matrix."""
            # Unpack specific params needed for calculation
            scale, crop_sky, crop_zoom, flip, rotate, final_dims = params.ida_mat_args

            # W_H default from original code logic (1600, 900)
            res = [input_size[0] * scale, input_size[1] * scale]   # resize 후 실측 해상도 (W,H)

            affine_mat = np.eye(3)              # 3x3 affine 누적 행렬 (원본 픽셀 → 최종 픽셀)
            affine_mat[:2, :2] *= scale         # 1) resize 스케일 반영

            w, h = final_dims                   # 최종 출력 너비/높이
            affine_mat[0, :2] *= w / (crop_zoom[2] - crop_zoom[0])   # 2) zoom: 가로 스케일 보정
            affine_mat[1, :2] *= h / (crop_zoom[3] - crop_zoom[1])   #    zoom: 세로 스케일 보정
            affine_mat[0, 2] += (w - res[0] * w / (crop_zoom[2] - crop_zoom[0])) / 2     # 가로 중앙 정렬 이동
            affine_mat[1, 2] += (h - (res[1] + crop_sky) * h / (crop_zoom[3] - crop_zoom[1])) / 2  # 세로(하늘crop 반영) 정렬

            if flip:
                # 3) 좌우 반전: x좌표 부호 뒤집고 폭만큼 평행이동.
                flip_mat = np.eye(3)
                flip_mat[0, 0] = -1
                flip_mat[0, 2] += w
                affine_mat = flip_mat @ affine_mat

            # 4) 이미지 중심 기준 회전 행렬 구성.
            theta = -rotate * np.pi / 180       # 도 → 라디안 (부호: 화면 좌표계 기준)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            x, y = w / 2, h / 2                 # 회전 중심 = 출력 이미지 중앙

            rot_center_mat = np.array([
                [cos_theta, -sin_theta, -x * cos_theta + y * sin_theta + x],
                [sin_theta, cos_theta, -x * sin_theta - y * cos_theta + y],
                [0, 0, 1],
            ])

            # 회전 ∘ (zoom/crop/scale/flip) 합성한 최종 ida_mat (3x3) 반환.
            return (rot_center_mat @ affine_mat).astype(np.float32)


# ============================================================================
# transforms/nuscenes/loading.py
# ============================================================================

# 실제 이미지 파일을 디스크에서 읽어 normalize 텐서로 만들고 lidar2img/extrinsics를 계산하는 transform.
class ImageDataLoader:

    def __init__(
        self,
        img_params: Dict[str, Any],
        dataset_dir: str,
    ) -> None:
        self.img_params = img_params
        self.dataset_dir = dataset_dir
        # ToTensor(0~1) → ImageNet 평균/표준편차로 normalize 하는 파이프라인.
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=img_params.get("mean", [0.485, 0.456, 0.406]),
                std=img_params.get("std", [0.229, 0.224, 0.225]),
            ),
        ])

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:

        # Load images and camera parameters from data_dict info.
        images: List[Image.Image] = []        # 카메라별 정규화 이미지 텐서 (3,H,W)
        intrinsics: List[np.ndarray] = []      # 카메라별 내부 파라미터 3x3 (augmentation 반영)
        extrinsics: List[torch.Tensor] = []    # 카메라별 외부 파라미터 4x4 (lidar→cam)

        for i, (img_path, intr, extr) in enumerate(zip(
            data_dict["images"],               # 카메라별 이미지 상대 경로
            data_dict["intrinsics"],           # 카메라별 3x3 내부행렬
            data_dict["extrinsics"],           # 카메라별 4x4 외부행렬
        )):
            this_image = self._load_image(img_path)               # 디스크에서 RGB 로딩
            this_intr = np.array(intr, dtype=np.float32)          # 내부행렬 → float32
            this_extr = torch.tensor(extr, dtype=torch.float32)   # 외부행렬 → 텐서

            # Update intrinsics and preprocess image with augmentation matrix if available.
            if "ida_mat" in data_dict:
                ida_mat = data_dict["ida_mat"][i]                 # 이 카메라의 image-aug 행렬
                this_intr = ida_mat @ this_intr                   # 내부행렬에 augmentation 합성 (좌표 일관성)
                final_dims = tuple(self.img_params["final_dim"][::-1])  # 최종 출력 (W,H)
                this_image = self._pil_preprocess_from_affine_mat(     # ida_mat 으로 워프 + normalize
                    this_image, ida_mat, final_dims
                )
            else:
                this_image = self.to_tensor(this_image)           # augmentation 없으면 normalize만

            images.append(this_image)
            intrinsics.append(this_intr)
            extrinsics.append(this_extr)

        # image: (N_cam, 3, H, W) — 모델 카메라 입력.
        data_dict["image"] = torch.stack(images, 0)
        data_dict["intrinsics"] = intrinsics   # augmentation 반영된 내부행렬 리스트
        data_dict["extrinsics"] = extrinsics   # 외부행렬 리스트

        # Compute lidar to image transformation matrices.
        lidar2img: List[torch.Tensor] = []     # lidar 좌표 → 이미지 픽셀 투영행렬 (4x4)
        for intr, extr in zip(intrinsics, extrinsics):
            viewpad = torch.eye(4, dtype=torch.float32)            # 3x3 내부행렬을 4x4로 패딩
            viewpad[:intr.shape[0], :intr.shape[1]] = intr
            lidar2img.append(viewpad @ extr)                       # (내부4x4) @ (외부4x4) = lidar→img
        data_dict["lidar2img"] = torch.stack(lidar2img, 0)         # (N_cam, 4, 4)

        # Update lidar2img and extrinsics according to bev augmentation if available.
        if "bev_augm" in data_dict:
            # BEV augmentation은 lidar 프레임을 회전/이동시키므로 투영행렬에도 우측에서 합성해야
            # GT(augmented lidar 프레임)와 이미지 투영이 일치한다.
            bev_augm = torch.from_numpy(data_dict["bev_augm"]).float()
            data_dict["extrinsics"] = torch.stack(data_dict["extrinsics"], 0) @ bev_augm  # (N_cam,4,4)
            data_dict["lidar2img"] = data_dict["lidar2img"] @ bev_augm                     # (N_cam,4,4)

        return data_dict

    def _load_image(self, rel_image_path: str) -> Image.Image:
        path = os.path.join(self.dataset_dir, rel_image_path)   # 데이터셋 루트 + 상대경로
        # Retry on transient NFS I/O errors (e.g. intermittent "Operation not
        # permitted") so a single filesystem glitch doesn't kill a multi-day run.
        last_err = None
        for attempt in range(5):                                # 최대 5회 재시도
            try:
                with Image.open(path) as im:
                    return im.convert("RGB")                    # 성공 시 RGB 이미지 반환
            except (OSError, IOError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))                 # 점증 백오프 후 재시도
        raise last_err                                          # 5회 실패 시 마지막 오류 전파

    def _pil_preprocess_from_affine_mat(self, img, affine_mat, final_dims):
        # PIL.transform 은 출력→입력 매핑을 요구하므로 ida_mat의 역행렬을 사용.
        inv_mat = np.linalg.inv(affine_mat)
        img = img.transform(
            size=tuple(final_dims),                             # 최종 출력 크기 (W,H)
            method=AffineTransform(inv_mat[:2].ravel()),        # 2x3 affine 파라미터
            resample=Image.BILINEAR,                            # 양선형 보간
        )
        return self.to_tensor(img)                              # 워프된 이미지 → normalize 텐서


# 5개 레이더의 multi-sweep 포인트를 ego 좌표로 모으고 20+차원 feature로 인코딩하는 transform.
class RadarDataLoader:
    def __init__(
        self,
        nusc,
        num_sweeps: int = 7,                  # 누적할 sweep 수 (시간축 밀도)
        lidar_name: str = "LIDAR_TOP",        # 기준 좌표계 = LIDAR_TOP
        radar_names: Optional[list] = [       # 사용할 레이더 5개
            "RADAR_BACK_RIGHT",
            "RADAR_BACK_LEFT",
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
        ],
    ) -> None:
        self.nusc = nusc
        self.num_sweeps = num_sweeps
        self.lidar_name = lidar_name
        self.radar_names = radar_names

    def __call__(self, data_dict: Dict[str, any]) -> Dict[str, any]:

        # Assert necesary keys are present.
        assert "bev_augm" in data_dict, "Key 'bev_augm' not found in data_dict"  # augmentation 행렬 필수
        assert "token" in data_dict, "Key 'token' not found in data_dict"        # 샘플 토큰 필수
        bev_augm = data_dict["bev_augm"]      # 4x4 BEV augmentation 행렬

        # Extract sample information.
        sample_token = data_dict["token"]
        sample_rec = self.nusc.get("sample", sample_token)             # 샘플 메타 레코드
        ref_token = sample_rec["data"][self.lidar_name]                # 기준 lidar sample_data 토큰
        ref_sd_record = self.nusc.get("sample_data", ref_token)        # 기준 lidar sample_data 레코드

        # Iterate through radar sensors and load points.
        radar_points = []                      # 레이더별 인코딩 feature 누적
        for radar_name in self.radar_names:

            radar_token = sample_rec["data"][radar_name]               # 이 레이더 sample_data 토큰
            sd_record = self.nusc.get("sample_data", radar_token)      # 이 레이더 sample_data 레코드

            # Points are loaded in LiDAR coordinates.
            # devkit이 ref_chan=lidar 기준으로 multi-sweep를 누적해 lidar 좌표로 반환. pc.points: (18, N).
            pc, times = RadarPointCloud.from_file_multisweep(
                nusc=self.nusc,
                sample_rec=sample_rec,
                chan=radar_name,
                ref_chan=self.lidar_name,
                nsweeps=self.num_sweeps,
                min_distance=1.0,                                      # 자차 반사 제거(1m 이내)
            )

            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
            # point cloud.
            # 속도는 devkit이 변환하지 않으므로 직접 ego 좌표로 회전시킨다 (레이더→lidar→ego 순).
            radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])      # 레이더 캘리브
            ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])    # lidar 캘리브
            velocities = pc.points[8:10, :]  # Compensated velocity    # 보정된 vx,vy (2, N)
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))        # z=0 추가 → (3, N)
            # [수정 ①③] 속도를 ego 프레임으로 = 위치(sensor_to_ego)·GT와 동일 프레임. radar→ego 1회전만.
            #   제거한 2개 회전: ego_pose(→global, ~86° 버그) + ref_cs.T(ego→lidar). LIDAR_TOP yaw가 -90°라
            #   ref_cs.T가 속도를 +90° 돌려 ego 위치와 어긋나게 했음(doppler 타깃이 GT와 90° 어긋난 원인).
            #   CRN은 위치도 lidar라 ref_cs.T를 유지했지만, 우리는 위치를 ego로 바꾸므로 속도도 ego여야 함.
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)        # radar→ego
            # [수정 ②] GT 속도(_apply_bev_aug_to_velocity)·위치와 동일한 bev_augm inverse 회전 적용
            #          (학습 시 augmentation 일관성; eval은 bev_augm=identity라 무영향).
            bev_aug_np = bev_augm.cpu().numpy() if torch.is_tensor(bev_augm) else np.asarray(bev_augm)
            velocities = np.dot(bev_aug_np[:3, :3].T, velocities)                                            # bev_augm 회전
            velocities = velocities[:2, :].T  # Keep only x and y components          # (N, 2) vx,vy

            # Transform points from LiDAR to ego coordinates.
            sensor_to_ego = transform_matrix(                          # lidar 좌표 → ego 좌표 4x4
                ref_cs_record["translation"],
                Quaternion(ref_cs_record["rotation"]),
            )
            homog_points = np.concatenate([pc.points[:3, :], np.ones((1, pc.points.shape[1]))], axis=0)  # (4, N) 동차좌표
            homog_points = sensor_to_ego @ homog_points # Shape: (4, N)               # ego 좌표로 변환
            homog_points = self._augment_boxes(bev_augm, homog_points)                # BEV augmentation 역적용
            xyz_ego = homog_points[:3, :].T  # Shape: (N, 3)                           # (N, 3) 좌표

            # Perform encoding of RADAR variables.
            # 원시 18채널 + ego좌표/ego속도/시간 → 약 50채널 feature로 인코딩.
            features = self.encode_radar_features(pc.points.T, xyz_ego, velocities, times.T, bev_augm)

            radar_points.append(features)

        # Stack all radar points into a single array
        if len(radar_points) > 0:
            radar_points = np.vstack(radar_points)         # 5개 레이더 feature 세로로 합침 (N_total, F)

            # Mask the points that are outside the BEV grid.
            # BEV 격자(±50m, z±10m) 밖 포인트 제거.
            mask = (
                (radar_points[:, 0] >= -50.0) & (radar_points[:, 0] < 50.0) &
                (radar_points[:, 1] >= -50.0) & (radar_points[:, 1] < 50.0) &
                (radar_points[:, 2] >= -10.0) & (radar_points[:, 2] < 10.0)
            )
            radar_points = radar_points[mask]

        # radar_points: (N, F) 텐서. 샘플마다 N이 달라 collate에서 stack 못하고 list로 둔다.
        data_dict["radar_points"] = torch.from_numpy(radar_points).float()
        return data_dict

    def encode_radar_features(
        self,
        points: np.ndarray,
        xyz_ego: np.ndarray,
        velocities_ego: np.ndarray,
        times: np.ndarray,
        bev_augm: np.ndarray = None,
    ):
        """Perform encoding of RADAR variables from raw point clouds in
        the NuScenes dataset.

        Original features:
        - x, y, z [0, 1, 2]: Cartesian coordinates of the point with
            respect to the sensor.
        - dyn_prop [3]: Dynamic property of the point.
        - id [4]: Unique ID of the point.
        - rcs [5]: Radar Cross Section of the point.
        - vx, vy [6, 7]: Velocity of the point in x and y directions
            with respect to the sensor.
        - vx_comp, vy_comp [8, 9]: Compensated velocity of the point
            in x and y directions with respect to the sensor.
        - is_quality_valid [10]: Boolean flag indicating whether the
            quality of the point is valid.
        - ambig_state [11]: State of Doppler (radial velocity)
            ambiguity solution.
        - x_rms, y_rms [12, 13]: Root Mean Square of the point
            in x and y directions with respect to the sensor.
        - invalid_state [14]: State of the cluster validity.
        - pdh0 [15]: False alarm probability of the point. Probabilty
            of being an artifact caused by multipath or similar.
        - vx_rms, vy_rms [16, 17]: Root Mean Square of the
            compensated velocity of the point in x and y directions
            with respect to the sensor.

        We add two additional features:
        - times [18]: Delta of time between the present time and the
            capture time of the point.
        - nusc_filter [19]: Boolean flag indicating whether the point
            passes the NuScenes filter.

        Transformations are applied as follows:
        - x, y, z [0, 1, 2]: Move to ego coordinates. WARNING: This is
            precomputed and passes as xyz_ego.
        - dyn_prop [3]: One-hot encoded with 8 classes.
        - vx_comp, vy_comp [6, 7]: Move to ego coordinates and
            compensate multi-sweeps. TODO.
        - ambig_state [11]. One-hot encoded with 5 classes.
        - invalid_state [14]. One-hot encoded with 18 classes.
        - pdh0 [15]. Ordinal encoding with 8 possible values.
        - nusc_filter [19].

        Args:
            points (N, 18): Raw RADAR point cloud data.
            xyz_ego (N, 3): Cartesian coordinates of the point in ego coordinates.
            velocities_ego (N, 2): Compensated velocity of the point in ego coordinates.
            times (N, 1): Time delta of the point.
            bev_augm (4, 4): BEV augmentation matrix.
        Returns:
            features (N, F): Encoded RADAR features.
        """

        # === 원시 18채널을 의미별로 분리 ===
        xyz = points[:, :3]  # Shape: (N, 3)              # 센서좌표 x,y,z (ego좌표는 xyz_ego로 별도 전달)
        dyn_prop = points[:, 3]  # Shape: (N,)            # 동적 속성 (0~7)
        ids = points[:, 4]  # Shape: (N,)                 # 포인트 ID
        rcs = points[:, 5]  # Shape: (N,)                 # 레이더 단면적(RCS)
        vxy = points[:, 6:8]  # Shape: (N, 2)             # 비보정 속도 vx,vy
        vxy_comp = points[:, 8:10]  # Shape: (N, 2)       # 보정 속도(이미 ego속도로 변환되어 별도 전달)
        is_quality_valid = points[:, 10]  # Shape: (N,)   # 품질 유효 플래그
        ambig_state = points[:, 11]  # Shape: (N,)        # Doppler 모호성 상태 (0~4)
        xy_rms = points[:, 12:14]  # Shape: (N, 2)        # 위치 RMS
        invalid_state = points[:, 14]  # Shape: (N,)      # 클러스터 무효 상태 (0~17)
        pdh0 = points[:, 15]  # Shape: (N,)               # 허위경보 확률 등급 (0~7)
        vxy_rms = points[:, 16:18]  # Shape: (N, 2)       # 속도 RMS

        # One-hot encode dynamic property.
        dyn_prop_one_hot = np.zeros((xyz.shape[0], 8), dtype=np.float32)   # (N, 8)
        dyn_prop_one_hot[np.arange(xyz.shape[0]), np.rint(dyn_prop).astype(int)] = 1.0  # 해당 클래스만 1

        # Move compensated velocities to ego coordinates.

        # One-hot encode ambiguity state.
        ambig_state_one_hot = np.zeros((xyz.shape[0], 5), dtype=np.float32)  # (N, 5)
        ambig_state_one_hot[np.arange(xyz.shape[0]), np.rint(ambig_state).astype(int)] = 1.0

        # One-hot encode invalid state.
        invalid_state_one_hot = np.zeros((xyz.shape[0], 18), dtype=np.float32)  # (N, 18)
        invalid_state_one_hot[np.arange(xyz.shape[0]), np.rint(invalid_state).astype(int)] = 1.0

        # Ordinal encode pdh0.
        pdh0_encoded = np.zeros((xyz.shape[0], 7), dtype=np.float32)   # (N, 7) 순서형 인코딩
        for i in range(7):
            pdh0_encoded[:, i] = (np.rint(pdh0) > i).astype(np.float32)  # pdh0>i 이면 1 (누적)

        # Calculate nusc_filter.
        # nuScenes 표준 필터: 유효 클러스터 & 동적 & 모호성 해소된 포인트만 1.
        nusc_filter = np.zeros(xyz.shape[0], dtype=np.float32)
        mask1 = (invalid_state == 0)       # 유효 클러스터
        mask2 = (dyn_prop < 7)             # 정적(7)이 아닌 포인트
        mask3 = (ambig_state == 3)         # 모호성 해소(state 3)
        nusc_filter[mask1 & mask2 & mask3] = 1.0

        # Concatenate all features.
        # 좌표/속도/시간/플래그를 가로로 이어 붙여 최종 feature (N, F) 생성.
        features = np.concatenate(
            [
                xyz_ego,  # (N, 3)
                dyn_prop_one_hot,  # (N, 8)
                ids[:, None],  # (N, 1)
                rcs[:, None],  # (N, 1)
                vxy,  # (N, 2)
                velocities_ego,  # (N, 2)
                is_quality_valid[:, None],  # (N, 1)
                ambig_state_one_hot,  # (N, 5)
                xy_rms,  # (N, 2)
                invalid_state_one_hot,  # (N, 18)
                pdh0_encoded,  # (N, 7)
                vxy_rms,  # (N, 2)
                times,  # (N, 1)
                nusc_filter[:, None],  # (N, 1)
            ],
            axis=1,
        )  # Shape: (N, F)

        return features

    def _augment_boxes(self, bev_aug, points, inverse=True):
        """from PointBeV."""
        # BEV augmentation을 포인트(또는 박스 코너)에 적용. inverse=True면 역변환(쿼리 관점).

        points_in = np.copy(points)            # (3 또는 4, N) 입력 복사
        Rquery = np.zeros((3, 3))              # 적용할 회전행렬

        if isinstance(bev_aug, torch.Tensor):
            bev_aug = bev_aug.cpu().numpy()    # 텐서면 numpy로

        if inverse:
            # Inverse query aug:
            # Ex: when tx=10, the query is 10/res meters front,
            # so points are fictivelly 10/res meters back.
            Rquery[:3, :3] = bev_aug[:3, :3].T            # 역회전 = 전치
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]  # x,y 이동 부호 반전(z 유지)
            tquery = tquery[:, None]                       # (3,1) 브로드캐스트용

            # Rquery @ (X + tquery)
            points_out = Rquery @ (points_in[:3, :] + tquery)  # 먼저 이동 후 역회전
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]               # 정방향 회전
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3] # 정방향 이동(z 부호 반전)
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = (Rquery @ points_in[:3, :]) + tquery  # 회전 후 이동

        return points_out  # (3, N) 변환된 좌표


# LIDAR_TOP multi-sweep 포인트를 ego 좌표로 모으고 (x,y,z,intensity,time) 5채널로 인코딩.
class LidarDataLoader:
    """
    Data loader for LiDAR point clouds from the nuScenes dataset.
    Processes multi-sweep data, transforms to ego coordinates, and applies BEV augmentation.
    """

    def __init__(
        self,
        nusc,
        num_sweeps: int = 10,
        lidar_names: List[str] = ["LIDAR_TOP"],
        min_distance: float = 1.0,
    ) -> None:
        """
        Args:
            nusc: The NuScenes instance.
            num_sweeps: Number of sweeps to aggregate.
            lidar_names: List of LiDAR sensors to use (usually just LIDAR_TOP).
            min_distance: Minimum distance to filter ego-vehicle reflections.
        """
        self.nusc = nusc
        self.num_sweeps = num_sweeps
        self.lidar_names = lidar_names
        self.min_distance = min_distance

    def __call__(self, data_dict: Dict[str, any]) -> Dict[str, any]:
        """
        Loads, transforms, and encodes LiDAR data for a specific sample.
        """
        # Assert necessary keys are present.
        assert "bev_augm" in data_dict, "Key 'bev_augm' not found in data_dict"
        assert "token" in data_dict, "Key 'token' not found in data_dict"

        bev_augm = data_dict["bev_augm"] # Shape (4, 4)   # BEV augmentation 행렬

        # Extract sample information.
        sample_token = data_dict["token"]
        sample_rec = self.nusc.get("sample", sample_token)            # 샘플 메타 레코드

        # We define the main LiDAR as the reference frame (usually LIDAR_TOP).
        ref_channel = self.lidar_names[0]                             # 기준 채널 = LIDAR_TOP
        ref_token = sample_rec["data"][ref_channel]                   # 기준 sample_data 토큰
        ref_sd_record = self.nusc.get("sample_data", ref_token)       # 기준 sample_data 레코드
        ref_cs_record = self.nusc.get("calibrated_sensor", ref_sd_record["calibrated_sensor_token"])  # lidar 캘리브

        lidar_points_list = []                                        # lidar별 feature 누적

        for lidar_name in self.lidar_names:

            # Points are loaded and transformed to the reference frame (LIDAR_TOP)
            # by the SDK logic internally if ref_chan is set.
            pc, times = LidarPointCloud.from_file_multisweep(
                nusc=self.nusc,
                sample_rec=sample_rec,
                chan=lidar_name,
                ref_chan=ref_channel,
                nsweeps=self.num_sweeps,
                min_distance=self.min_distance,
            )

            # pc.points shape: (4, N) -> x, y, z, intensity

            # Create transformation matrix: Sensor (Lidar) -> Ego
            sensor_to_ego = transform_matrix(
                ref_cs_record["translation"],
                Quaternion(ref_cs_record["rotation"]),
            )

            # Homogeneous coordinates for geometric transformation
            # Shape: (4, N) -> x, y, z, 1
            homog_points = np.concatenate([pc.points[:3, :], np.ones((1, pc.points.shape[1]))], axis=0)

            # Transform to Ego frame
            homog_points = sensor_to_ego @ homog_points       # lidar → ego 좌표 변환

            # Apply Data Augmentation (Inverse BEV Augmentation)
            homog_points = self._augment_boxes(bev_augm, homog_points)   # BEV augmentation 역적용

            xyz_ego = homog_points[:3, :].T  # Shape: (N, 3)  # (N,3) augmented ego 좌표

            # Perform encoding of LiDAR variables.
            features = self.encode_lidar_features(pc.points.T, xyz_ego, times.T)  # (N,5) 인코딩

            lidar_points_list.append(features)

        # Stack all points (if multiple LiDARs were used)
        if len(lidar_points_list) > 0:
            lidar_points = np.vstack(lidar_points_list)     # lidar별 feature 합침 (N, 5)

            # Mask points outside the BEV grid bounds.
            # BEV 격자(±50m, z±10m) 밖 포인트 제거.
            mask = (
                (lidar_points[:, 0] >= -50.0) & (lidar_points[:, 0] < 50.0) &
                (lidar_points[:, 1] >= -50.0) & (lidar_points[:, 1] < 50.0) &
                (lidar_points[:, 2] >= -10.0) & (lidar_points[:, 2] < 10.0)
            )
            lidar_points = lidar_points[mask]
        else:
            # Fallback for empty clouds
            lidar_points = np.zeros((0, 5), dtype=np.float32)  # 빈 포인트 클라우드 대비

        # lidar_points: (N, 5) = (x,y,z,intensity,time). 가변 길이라 모델에서 list로 처리.
        data_dict["lidar_points"] = torch.from_numpy(lidar_points).float()
        return data_dict

    def encode_lidar_features(
        self,
        points: np.ndarray,
        xyz_ego: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Encodes LiDAR features.

        Original features from LidarPointCloud:
        - x, y, z [0, 1, 2]: Coordinates in sensor frame.
        - intensity [3]: Reflection intensity (0-255).

        Output features:
        - x, y, z [0, 1, 2]: Coordinates in Ego frame (augmented).
        - intensity [3]: Scaled or raw intensity.
        - time [4]: Time delta.

        Args:
            points (N, 4): Raw LiDAR data [x, y, z, intensity].
            xyz_ego (N, 3): Transformed coordinates.
            times (N, 1): Time lag.

        Returns:
            features (N, 5): Encoded features.
        """

        intensity = points[:, 3:4] # Shape: (N, 1)    # 반사 강도 채널

        # Optional: Normalize intensity to [0, 1] if it's in [0, 255]
        # intensity = intensity / 255.0

        # Concatenate features
        # ego좌표 + 강도 + 시간차 → (N, 5) feature.
        features = np.concatenate(
            [
                xyz_ego,    # (N, 3)   # augmented ego 좌표
                intensity,  # (N, 1)   # 반사 강도
                times,      # (N, 1)   # sweep 시간차
            ],
            axis=1
        ) # Shape: (N, 5)

        return features

    def _augment_boxes(self, bev_aug, points, inverse=True):
        """
        Applies BEV augmentation transformation to points.

        Args:
            bev_aug (4, 4): Augmentation matrix.
            points (4, N): Homogeneous points.
            inverse (bool): Whether to apply inverse transformation.
        """
        points_in = np.copy(points)
        Rquery = np.zeros((3, 3))

        if isinstance(bev_aug, torch.Tensor):
            bev_aug = bev_aug.cpu().numpy()

        if inverse:
            # Inverse query aug:
            # Rotates and translates points to match the inverse of the BEV crop/rotation
            Rquery[:3, :3] = bev_aug[:3, :3].T
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ (X + tquery)
            points_out = Rquery @ (points_in[:3, :] + tquery)
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = (Rquery @ points_in[:3, :]) + tquery

        return points_out


# ============================================================================
# transforms/nuscenes/labels.py
# ============================================================================

# nuScenes detection 10-class 중 vehicle 군집 (TrackLoss instance_id_map용).
# pedestrian(5), traffic_cone(8), barrier(9)는 정적/비차량 → tracking 대상 아님.
_VEHICLE_CLASS_IDS = (0, 1, 2, 3, 4, 6, 7)


class LoadSegmentationLabels:
    """Multi-class detection 학습용 GT 로더.

    출력:
      - {key}_gt_boxes  : [N, 11] (cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy, inst_id)
      - {key}_gt_labels : [N] long, nuScenes detection class id (0-9)
      - {key}_instance_id : [H, W] long, vehicle 클래스 픽셀별 instance id (TrackLoss용)
    """
    def __init__(
        self,
        labels_dir,
        mode: Literal["vehicle"] = "vehicle",
        bev_conf: Dict[str, Any] = {"h": 200, "w": 200},
        load_velocity: bool = True,
        load_instance_id: bool = True,
        # 모델 BEV 범위 (±50m default). GT가 이 범위 밖이면 query head가
        # 절대 도달 못 하므로 학습에서 제외 — L1 loss가 영원히 깎이지 않게.
        bev_x_min: float = -50.0,
        bev_x_max: float = 50.0,
        bev_y_min: float = -50.0,
        bev_y_max: float = 50.0,
        # 학습에 사용할 nuScenes class id 화이트리스트 (None = 전체 10 class).
        # 예: [0]만 → car-only 학습. labels는 0~len(class_filter)-1로 remap.
        class_filter: Optional[Sequence[int]] = None,
    ) -> None:
        self.labels_dir = pathlib.Path(labels_dir)   # GT .npz가 들어있는 라벨 루트
        self.mode = mode                              # 출력 키 접두사 ("vehicle")
        self.bev_H = bev_conf["h"]                    # BEV 격자 높이(픽셀)
        self.bev_W = bev_conf["w"]                    # BEV 격자 너비(픽셀)
        self.to_tensor = ToTensor()
        self.load_velocity = load_velocity            # 속도 GT 로드 여부
        self.load_instance_id = load_instance_id      # instance id(추적용) 로드 여부
        self.bev_x_min = bev_x_min                    # BEV 유효 범위(미터) — 밖이면 제외
        self.bev_x_max = bev_x_max
        self.bev_y_min = bev_y_min
        self.bev_y_max = bev_y_max
        # class_filter: id → local index 매핑. None이면 0~9 그대로.
        if class_filter is not None:
            self.class_filter = {int(cid): i for i, cid in enumerate(class_filter)}
        else:
            self.class_filter = None

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # GT 렌더에 필요한 키 4종 확인.
        assert "bev_augm" in data_dict       # BEV augmentation 행렬
        assert "gt_box" in data_dict         # GT npz 파일명
        assert "scene" in data_dict          # scene 이름(디렉터리)
        assert "view" in data_dict           # 미터→픽셀 view 행렬
        bev_augm = data_dict["bev_augm"]
        gt_box = data_dict["gt_box"]
        scene = data_dict["scene"]
        view = np.array(data_dict["view"])   # 3x3 view 행렬

        scene_dir = self.labels_dir / scene                               # scene별 라벨 디렉터리
        gt_box = np.load(scene_dir / gt_box, allow_pickle=True)['gt_box'] # GT 박스 배열 로드

        # TrackLoss용 per-pixel instance id (vehicle 클래스 픽셀만 채움). (H, W), 기본 -1(배경).
        instance_id_map = np.full((self.bev_H, self.bev_W), -1, dtype=np.int64)

        # Phase 3c: multi-class query head GT (가변 길이).
        gt_box_list: List[List[float]] = []    # 박스별 11차원 벡터 누적
        gt_label_list: List[int] = []          # 박스별 클래스 라벨 누적

        buf = np.zeros((self.bev_H, self.bev_W), dtype=np.uint8)   # fillPoly용 임시 버퍼 (H,W)

        for box_data in gt_box:                # GT 박스 하나씩 순회
            if len(box_data) == 0:
                continue                       # 빈 항목 스킵

            class_idx = int(box_data[7])       # nuScenes detection 클래스 id (0~9)

            # class_filter 화이트리스트 적용 — 차량/특정 class만 학습할 때.
            if self.class_filter is not None and class_idx not in self.class_filter:
                continue

            # Box geometry.
            translation = [box_data[0], box_data[1], box_data[4]]   # 중심 (x, y, z)
            size = [box_data[2], box_data[3], box_data[5]]          # 크기 (w, l, h)
            yaw = -box_data[6] - np.pi / 2     # 저장된 yaw → BEV 규약 yaw 로 변환

            # BEV augmentation의 z축 회전을 yaw에도 적용 (position/velocity와 동일한 inverse R).
            bev_aug_np = bev_augm.cpu().numpy() if isinstance(bev_augm, torch.Tensor) else bev_augm
            bev_yaw_delta = float(np.arctan2(bev_aug_np[1, 0], bev_aug_np[0, 0]))  # augmentation 회전각
            yaw = yaw - bev_yaw_delta          # yaw에서 augmentation 회전만큼 보정

            # Velocity (BEV-aug 회전 적용).
            if self.load_velocity and len(box_data) > 10:
                vx = box_data[9]               # GT 속도 vx
                vy = box_data[10]              # GT 속도 vy
                if np.isnan(vx) or np.isnan(vy):
                    vx, vy = 0.0, 0.0          # NaN이면 0으로
                vx_aug, vy_aug = self._apply_bev_aug_to_velocity(bev_augm, vx, vy)  # augmentation 회전 적용
            else:
                vx_aug, vy_aug = 0.0, 0.0

            # === Multi-class GT (모든 nuScenes detection 클래스) ===
            aug_homog = np.ones((4, 1))                         # 중심점 동차좌표 (4,1)
            aug_homog[:3, 0] = [box_data[0], box_data[1], box_data[4]]
            aug_metric = self._prepare_augmented_boxes(bev_augm, aug_homog).reshape(3)  # augmentation 적용된 중심
            cx_m, cy_m, cz_m = float(aug_metric[0]), float(aug_metric[1]), float(aug_metric[2])  # 미터 단위 중심

            # BEV 범위 밖 객체는 query head가 절대 예측 못 하므로 학습에서 제외.
            if not (self.bev_x_min <= cx_m <= self.bev_x_max and
                    self.bev_y_min <= cy_m <= self.bev_y_max):
                continue

            inst_id = int(box_data[11]) if (self.load_instance_id and len(box_data) > 11) else -1  # 추적 instance id
            # gt_box 한 줄 = [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy, inst_id] (11차원).
            gt_box_list.append([
                cx_m, cy_m, cz_m,
                float(size[1]), float(size[0]), float(size[2]),  # w, l, h
                float(np.sin(yaw)), float(np.cos(yaw)),          # yaw를 sin/cos로 (불연속 회피)
                float(vx_aug), float(vy_aug),
                float(inst_id),
            ])
            local_label = self.class_filter[class_idx] if self.class_filter is not None else class_idx  # remap된 라벨
            gt_label_list.append(local_label)

            # === instance_id_map: vehicle 클래스 픽셀만 (TrackLoss) ===
            if class_idx not in _VEHICLE_CLASS_IDS:
                continue                       # 비차량(보행자/콘/배리어)은 instance map에서 제외
            if not (self.load_instance_id and len(box_data) > 11):
                continue                       # instance id 없으면 제외

            # Project box footprint → BEV pixel mask.
            box = Box(translation, size, sincos2quaternion(np.sin(yaw), np.cos(yaw)))  # nuScenes Box 구성
            points = box.bottom_corners()      # 박스 바닥면 4코너 (3, 4)
            homog_points = np.ones((4, 4))     # 동차좌표 (4, 4)
            homog_points[:3, :] = points
            homog_points[-1, :] = 1
            points = self._prepare_augmented_boxes(bev_augm, homog_points)  # augmentation 적용 (3,4)
            points[2] = 1                      # z행을 1로 → view 행렬 적용 위한 동차화
            points = (view @ points)[:2]       # 미터→BEV 픽셀 좌표 (2, 4)

            buf.fill(0)                                                   # 버퍼 초기화
            cv2.fillPoly(buf, [points.round().astype(np.int32).T], 1, INTERPOLATION)  # 박스 footprint 채움
            mask = buf > 0                                               # 채워진 픽셀 마스크
            instance_id_map[mask] = int(box_data[11])                    # 해당 픽셀에 instance id 기록

        instance_id_map = torch.from_numpy(instance_id_map)   # (H, W) long 텐서
        if len(gt_box_list) > 0:
            gt_boxes_tensor = torch.tensor(gt_box_list, dtype=torch.float32)    # (N, 11)
            gt_labels_tensor = torch.tensor(gt_label_list, dtype=torch.long)    # (N,)
        else:
            gt_boxes_tensor = torch.zeros((0, 11), dtype=torch.float32)   # 박스 없으면 빈 텐서
            gt_labels_tensor = torch.zeros((0,), dtype=torch.long)

        # 출력 키: {mode}_instance_id (H,W) / {mode}_gt_boxes (N,11) / {mode}_gt_labels (N,).
        data_dict.update({
            f"{self.mode}_instance_id": instance_id_map,
            f"{self.mode}_gt_boxes": gt_boxes_tensor,
            f"{self.mode}_gt_labels": gt_labels_tensor,
        })
        return data_dict

    def _apply_bev_aug_to_velocity(self, bev_aug, vx, vy):
        # 속도 벡터에 위치와 동일한 inverse 회전(R^T)을 적용 (평행이동은 속도에 무영향).
        if isinstance(bev_aug, torch.Tensor):
            bev_aug = bev_aug.cpu().numpy()
        R = bev_aug[:3, :3].T               # inverse 회전
        v = np.array([vx, vy, 0.0])         # 속도 벡터(z=0)
        v_aug = R @ v                       # 회전 적용
        return float(v_aug[0]), float(v_aug[1])   # augmentation 적용된 vx, vy

    def _prepare_augmented_boxes(self, bev_aug, points, inverse=True):
        # 박스 코너/중심에 BEV augmentation 적용 (RadarDataLoader._augment_boxes 와 동일 로직).
        points_in = np.copy(points)
        Rquery = np.zeros((3, 3))
        if isinstance(bev_aug, torch.Tensor):
            bev_aug = bev_aug.cpu().numpy()
        if inverse:
            Rquery[:3, :3] = bev_aug[:3, :3].T            # 역회전
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]  # x,y 이동 부호 반전
            tquery = tquery[:, None]
            points_out = (Rquery @ (points_in[:3, :] + tquery))   # 이동 후 역회전
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]              # 정방향 회전
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]
            points_out = ((Rquery @ points_in[:3, :]) + tquery)   # 회전 후 이동
        return points_out


# HD맵 레이어를 ego 주변 BEV 마스크로 렌더링하는 transform (map segmentation 태스크용).
class LoadMapLabels:
    nusc_map_name: List[str] = [        # nuScenes 4개 도시 맵 이름
        'boston-seaport',
        'singapore-onenorth',
        'singapore-hollandvillage',
        'singapore-queenstown'
    ]
    map_layers: List[str] = [           # 추출할 맵 레이어(각각 별도 마스크로 출력)
        'lane',
        "road_segment",
        "road_divider",
        "lane_divider",
        "ped_crossing",
        "walkway",
        "carpark_area",
    ]

    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = dataset_dir
        self.to_tensor = ToTensor()
        self.nusc_map = {}              # 도시별 맵 객체 캐시
        for map_name in self.nusc_map_name:
            self.nusc_map[map_name] = FixedNuScenesMap(dataroot=self.dataset_dir, map_name=map_name)  # Shapely 호환 맵 로드

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert "map_name" in data_dict     # 도시 이름 필수
        assert "pose" in data_dict         # ego pose 필수
        assert "view" in data_dict
        assert "bev_augm" in data_dict
        bev_augm = data_dict["bev_augm"]
        map_name = data_dict["map_name"]
        pose = data_dict["pose"] @ bev_augm   # augmentation 반영된 글로벌 ego pose
        view = np.array(data_dict["view"])
        H, W = 200, 200                       # BEV 마스크 해상도

        S = np.array([                        # z축 제거(BEV 투영용) 선택행렬
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        lidar2global = (view @ S @ np.linalg.inv(pose))   # 글로벌→BEV 픽셀 변환
        rotation_lidar2global = lidar2global[:3, :3]      # 회전부만 추출
        v = np.dot(rotation_lidar2global, np.array([1, 0, 0]))  # x축 방향 벡터 변환
        yaw = np.arctan2(v[1], v[0])                      # ego heading 추출
        angle = (yaw / np.pi * 180)                       # 라디안 → 도

        # ego 중심 100x100m 패치를 yaw만큼 회전시켜 각 레이어 마스크 (len(layers), H, W) 생성.
        map_mask = self.nusc_map[map_name].get_map_mask(
            (pose[0][-1], pose[1][-1], 100, 100), angle, self.map_layers, (H, W)
        )
        for i, m in enumerate(map_mask):
            # 레이어별로 상하반전 후 텐서화 (BEV row=-x 규약 정렬) → data_dict[레이어명].
            data_dict[self.map_layers[i]] = self.to_tensor(255 * np.flipud(m)[..., None])
        return data_dict


# ============================================================================
# dataset_refactor.py
# ============================================================================

def relative_transform(
    T_a: np.ndarray, T_b: np.ndarray
) -> np.ndarray:
    """
    Compute relative transform T_a→b using numpy.

    Args:
        T_a: (4,4) pose matrix of frame a in world coords.
        T_b: (4,4) pose matrix of frame b in world coords.

    Returns:
        (4,4) relative transform matrix.
    """
    if T_a.shape != (4, 4) or T_b.shape != (4, 4):
        raise ValueError("Inputs must be 4x4 transformation matrices.")   # 입력 검증

    # Inverse of T_a
    T_a_inv = np.linalg.inv(T_a)        # 프레임 a의 역변환

    # Relative transform
    T_rel = T_a_inv @ T_b              # a→b 상대변환 (b 좌표의 점을 a 좌표로 보냄)

    # # Extract rotation and translation
    # R_rel = T_rel[:3, :3]
    # t_rel = T_rel[:3, 3]

    return T_rel


class NuScenesDataset(Dataset):
    def __init__(self,
        scene_name: str,
        labels_dir: str,
        transforms: Optional[List[Callable]] = None,
        return_prev: bool = False,
        independent_augmentation: bool = True,
        # [multi-frame 8/21] 반환할 이전 프레임 개수 K. 1이면 기존 동작과 완전히 동일
        #   (키 prev_*, ego_motion, bev_warp). K≥2면 prev2_*/ego_motion2/bev_warp2 … 추가.
        #   RCTrans/StreamPETR식 다중프레임 temporal fusion용 — GS 인코딩은 프레임 단위라 불변.
        num_prev_frames: int = 1,
    ) -> None:
        scene_path = Path(labels_dir) / f"{scene_name}.json"   # scene별 샘플 목록 JSON 경로
        self.samples = json.loads(scene_path.read_text())      # 프레임 raw dict 리스트 로드
        self.transforms = transforms                           # 순차 적용할 transform 파이프라인
        self.return_prev = return_prev                         # True면 prev 프레임도 함께 반환(temporal)
        # 기하 ego-warp 파이프라인 제약:
        #   independent_augmentation=True  → 프레임마다 augmentation을 따로 샘플 (rotation-invariance 유도).
        #     이 경우 augmented BEV 간 워프는 단일 행렬이 아니라 bev_augm/prev_bev_augm 합성이 필요.
        #   independent_augmentation=False → prev 프레임이 현재 프레임의 augmentation(bev_augm·ida_mat)을
        #     공유 → ego_motion 한 장으로 prev BEV를 현재 프레임에 기하적으로 워프 가능.
        self.independent_augmentation = independent_augmentation
        self.num_prev_frames = max(1, int(num_prev_frames))

    def __len__(self):
        return len(self.samples)        # 이 scene의 프레임 수

    def _extract_data(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and converts necessary data fields from a raw sample dictionary.

        Args:
            sample_data: The raw dictionary for one frame loaded from the JSON.

        Returns:
            A processed dictionary with tensor/numpy types.
        """
        # Assuming the 'Sample' class from the original code was mainly for
        # dictionary-like unpacking and access. We extract fields directly.

        # raw JSON 한 프레임을 transform 파이프라인이 쓰는 키들로 변환.
        data_dict: Dict[str, Any] = {
            'view': torch.tensor(sample_data['view']),          # 미터→BEV픽셀 view 행렬 (3,3)
            'token': sample_data['token'],                      # 샘플 토큰(devkit 조회용)
            'map_name': sample_data['map_name'],                # 도시 맵 이름
            'pose': np.float32(sample_data['pose']),            # 글로벌 ego pose (4,4)
            'pose_inverse': np.float32(sample_data['pose_inverse']),  # 그 역행렬 (4,4)
            'images': sample_data['images'],                    # 카메라별 이미지 상대경로 리스트
            'intrinsics': sample_data['intrinsics'],            # 카메라별 내부행렬 리스트
            'extrinsics': sample_data['extrinsics'],            # 카메라별 외부행렬 리스트
            'scene': sample_data['scene'],                      # scene 이름(라벨 디렉터리)
            'gt_box': sample_data['gt_box'],                    # GT npz 파일명
        }
        return data_dict

    def _load_one(
        self, idx: int, fixed_augm: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """raw JSON sample → extract → transform pipeline.

        fixed_augm: 미리 주입할 augmentation 행렬(bev_augm·ida_mat). 주어지면 augmentation
        transform이 새로 샘플하지 않고 이 값을 그대로 사용 → 프레임 간 augmentation 공유.
        """
        raw_sample = self.samples[idx]               # idx번째 raw 프레임
        data_dict = self._extract_data(raw_sample)   # 키 정리
        if fixed_augm:
            data_dict.update(fixed_augm)             # 미리 주입된 augmentation이 있으면 덮어씀(공유)
        for transform in self.transforms:
            data_dict = transform(data_dict)         # transform 순차 적용 (image→bev_aug→radar→labels)
        return data_dict

    def __getitem__(self, idx):
        # 현재 프레임 1개를 dict로 로드 (image/radar/lidar2img/GT 등 포함).
        result: Dict[str, Any] = dict(self._load_one(idx))

        # Phase 3b: prev frame을 같이 반환. idx==0이면 자기 자신을 prev로 사용 (degenerate).
        # [multi-frame] K개 이전 프레임(t-1 … t-K)을 같은 규칙으로 반환.
        #   k=1은 기존 키(prev_*, ego_motion, bev_warp)를 그대로 써서 하위호환 유지.
        if self.return_prev:
            # 기하 ego-warp 제약: 공유 augmentation 모드면 현재 프레임의 augmentation을
            # prev에 넘겨 동일하게 적용한다. 그래야 ego_motion 한 장으로 워프가 성립.
            fixed_augm = None
            if not self.independent_augmentation:
                fixed_augm = {
                    k: result[k] for k in ("bev_augm", "ida_mat") if k in result
                }
            bev_augm_cur = np.asarray(result["bev_augm"], dtype=np.float64)

            for k in range(1, self.num_prev_frames + 1):
                prev_idx = max(0, idx - k)          # scene 시작부는 자기 자신으로 클램프
                prefix = "prev_" if k == 1 else f"prev{k}_"
                mkey = "ego_motion" if k == 1 else f"ego_motion{k}"
                wkey = "bev_warp" if k == 1 else f"bev_warp{k}"

                prev = self._load_one(prev_idx, fixed_augm=fixed_augm)
                for kk, v in prev.items():
                    result[f"{prefix}{kk}"] = v

                # ego_motion: prev lidar 프레임 → 현재 lidar 프레임 상대 변환 (4x4).
                #   ego_motion = pose_cur^{-1} @ pose_prev  ⇒ prev 좌표의 점을 현재 좌표로 보냄.
                ego_motion = relative_transform(result["pose"], prev["pose"])
                result[mkey] = torch.from_numpy(ego_motion).float()

                # bev_warp: 모델이 prev BEV feature를 현재 프레임으로 grid_sample 하기 위한
                # "augmented metric" 변환. 현재-aug 좌표 → prev-aug 좌표로 보낸다 (affine_grid는
                # output(현재) 좌표를 input(prev) 좌표로 매핑하므로 이 방향이 맞다):
                #   a_prev = bev_augm_prev^{-1} @ ego_motion^{-1} @ bev_augm_cur @ a_cur
                # 렌더 BEV는 augmented lidar 프레임에 있으므로(extrinsics @ bev_augm) augmentation을
                # 함께 합성해야 한다. 공유 augmentation이면 bev_augm == prev_bev_augm으로 단순해진다.
                bev_augm_prev = np.asarray(prev["bev_augm"], dtype=np.float64)
                bev_warp = (
                    np.linalg.inv(bev_augm_prev)
                    @ np.linalg.inv(ego_motion.astype(np.float64))
                    @ bev_augm_cur
                )
                result[wkey] = torch.from_numpy(bev_warp.astype(np.float32))

        return result


# ============================================================================
# nuscenes_dataset_generated.py
# ============================================================================

# nuscense 경로와 Label 경로에서 데이터셋을 불러옴

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    image=None,                         # image config
    **dataset_kwargs
):
    out = []                            # scene별 NuScenesDataset 누적
    dataset_dir = Path(dataset_dir)     # nuScenes 원본 데이터 루트
    labels_dir = Path(labels_dir)       # 전처리된 라벨/JSON 루트

    # Override augment if not training
    training = True if split == 'train' else False   # train split만 augmentation ON

    # Arrange transformations.
    nusc = NuScenes(version=version, dataroot=str(dataset_dir), verbose=False)  # devkit 핸들(레이더/lidar 로딩용)
    transforms = list()
    # transform 적용 순서가 중요: image-aug/bev-aug 먼저 → 그 행렬을 image/radar 로더가 소비.
    transforms += [
        RandomTransformImage(                          # 1) 카메라 ida_mat 샘플
            dataset_kwargs["img_params"],
            training=training,
            orig_img_size=(
                dataset_kwargs["img_params"]["W"],
                dataset_kwargs["img_params"]["H"],
            ),
        ),
        RandomTransformBev(dataset_kwargs["bev_aug_conf"], training=training),  # 2) BEV bev_augm 샘플
        ImageDataLoader(dataset_kwargs["img_params"], str(dataset_dir)),        # 3) 이미지 로드+투영행렬
        RadarDataLoader(nusc, num_sweeps=7),                                    # 4) 레이더 7-sweep 로드
        # LidarDataLoader(nusc, num_sweeps=1),                                  #    (lidar 로더는 현재 비활성)
    ]

    if dataset_kwargs.get("vehicle", False):
        # 5) detection/track GT 렌더 transform 추가 (vehicle 모드).
        transforms.append(LoadSegmentationLabels(
            labels_dir,
            mode="vehicle",
            class_filter=dataset_kwargs.get("class_filter"),
        ))


    if dataset_kwargs.get("map_layers", False):
        transforms.append(LoadMapLabels(dataset_dir))   # 6) (옵션) HD맵 레이어 GT 추가

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split   # mini 버전이면 split명에 prefix
    split_scenes = get_split(split, 'nuscenes')                    # 해당 split의 scene 목록

    return_prev = bool(dataset_kwargs.get("return_prev", False))   # temporal(이전 프레임) 반환 여부
    num_prev_frames = int(dataset_kwargs.get("num_prev_frames", 1) or 1)  # [multi-frame] K
    # 기하 ego-warp 제약: temporal일 때 prev/현재 프레임의 augmentation 공유 여부.
    independent_augmentation = bool(dataset_kwargs.get("independent_augmentation", True))
    for s in split_scenes:
        # scene 하나당 dataset 객체 1개 (DataModule에서 ConcatDataset으로 합침).
        tmp_dataset = NuScenesDataset(
            s, labels_dir, transforms=transforms, return_prev=return_prev,
            independent_augmentation=independent_augmentation,
            num_prev_frames=num_prev_frames,
        )
        out.append(tmp_dataset)
    return out                          # scene별 dataset 리스트


# ============================================================================
# data_module.py
# ============================================================================

# radar point used as key
def collate_fn_with_list(batch):
    # variable-length point clouds는 stack 못함 → list 유지.
    # 레이더/lidar 포인트는 샘플마다 개수가 달라 (N_i, F) 모양이 제각각 → torch.stack 불가.
    # 따라서 배치로 묶지 않고 list[Tensor]로 두고, 모델 내부에서 per-sample 처리한다.
    # Phase 3b: prev_radar_points도 마찬가지.
    # [multi-frame 8/21] prev2_/prev3_/… 접두사까지 포함하도록 접미사 매칭으로 일반화.
    #   (고정 리스트였을 때 K=4 첫 실행이 죽었음: bs=1이면 torch.stack([t])가 "성공"해
    #    prev2_radar_points가 [1,N,F] 텐서가 되고, radar_encoder의 torch.cat(list,0)이 TypeError.)
    def _is_point_key(k: str) -> bool:
        return k.endswith("radar_points") or k.endswith("lidar_points")

    # First, collect all keys from the batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())     # 배치 내 모든 키 수집(샘플마다 키가 다를 수 있어 합집합)

    result = {}

    # Process each key
    for key in all_keys:
        if _is_point_key(key):
            # Keep as list  — 가변 길이라 stack 안 하고 리스트 유지.
            result[key] = [item.get(key) for item in batch]
        else:
            # Batch the tensors
            values = [item.get(key) for item in batch if key in item]   # 이 키를 가진 샘플 값들
            if len(values) > 0 and isinstance(values[0], torch.Tensor):
                try:
                    result[key] = torch.stack(values)   # 같은 shape 텐서면 배치 차원으로 stack
                except:
                    result[key] = values                # shape 안 맞으면 list로 폴백
            else:
                result[key] = values                    # 텐서가 아니면(문자열 등) 그대로 list

    return result


class DataModule(L.LightningDataModule): # Lighting Data module
    # Lightning DataModule: train/val dataset 구성과 DataLoader 생성을 캡슐화.
    def __init__(
        self,
        dataset: str,                  # 데이터셋 모듈 이름(레지스트리 키)
        debug_mode: bool = False,      # True면 batch1/단일프로세스 디버그 설정
        data_config: dict = None,      # get_data에 넘길 데이터 설정
        loader_config: dict = None,    # DataLoader 설정
        model_config: dict = None,
        img_params: dict = None,       # 모델 측 이미지 파라미터(있으면 data_config에 주입)
    ) -> None:
        super().__init__()
        self.get_data = get_dataset_module_by_name(dataset).get_data   # 위 get_data 함수 바인딩
        self.data_config = data_config         # dataset_dir/labels_dir/version/transform 설정
        self.loader_config = loader_config     # batch_size/num_workers/prefetch 등 DataLoader 설정
        self.model_config = model_config
        if img_params is not None:
            self.data_config['img_params'] = img_params   # 모델 쪽 img_params를 데이터 설정에 주입

        if debug_mode:
            # 디버그 모드: 단일 프로세스·배치1로 강제 (재현성/디버깅 용이).
            self.loader_config['batch_size'] = 1
            self.loader_config['num_workers'] = 0
            self.loader_config['prefetch_factor'] = None
            # num_workers=0이면 persistent_workers=True가 DataLoader에서 에러 → 함께 끔.
            self.loader_config['persistent_workers'] = False

    def setup(self, stage: str) -> None:
        # Lightning이 fit/validate 시작 시 호출 — split별 dataset 구성.
        if stage == "fit":
            self.train_data = self.get_data(split='train', **self.data_config)   # scene별 dataset 리스트
            self.train_dataset = torch.utils.data.ConcatDataset(self.train_data) # 전 scene 합침
            # [B①] CBGS 클래스 밸런싱 리샘플링 (train만). trailer AP 0.000/CV 0.065 대응:
            #   클래스별로 "그 클래스를 포함한 프레임" 목록에서 total/10개씩 복원추출해
            #   epoch을 재구성 → 희소 클래스 프레임이 크게 오버샘플된다 (CBGS, Zhu et al. 2019).
            if self.data_config.get('cbgs', False):
                indices = self._build_cbgs_indices()
                orig = len(self.train_dataset)
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
                print(f"[CBGS] resampled epoch: {orig} → {len(indices)} frames")
        if stage in ["fit", "validate"]:
            self.val_data = self.get_data(split='val', **self.data_config)
            self.val_dataset = torch.utils.data.ConcatDataset(self.val_data)

    def _build_cbgs_indices(self, num_classes: int = 10, seed: int = 42):
        """[B①] ConcatDataset 전역 인덱스 기준 CBGS 리샘플 인덱스 생성 (결정적, DDP-안전).

        프레임별 클래스 존재는 라벨 npz(box_data[7])에서 읽고, BEV 범위(±50m) 밖 박스는
        LoadSegmentationLabels와 동일하게 제외한다. 스캔 결과는 labels_dir에 캐시(1회 비용).
        """
        labels_dir = Path(self.data_config['labels_dir'])
        cache_path = labels_dir / "cbgs_presence_v1.json"
        cache = {}
        try:
            if cache_path.exists():
                cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

        presence = []           # concat 전역 인덱스 순서와 동일하게 프레임별 클래스 리스트 누적
        cache_dirty = False
        for ds in self.train_data:                       # get_data가 만든 scene 순서 그대로
            scene = ds.samples[0]['scene']
            if scene in cache and len(cache[scene]) == len(ds.samples):
                presence.extend([set(f) for f in cache[scene]])
                continue
            scene_frames = []
            for s in ds.samples:
                cls_present = set()
                try:
                    gb = np.load(labels_dir / scene / s['gt_box'], allow_pickle=True)['gt_box']
                    for box_data in gb:
                        if len(box_data) == 0:
                            continue
                        # LoadSegmentationLabels와 동일한 BEV 범위 필터 (augmentation 전 좌표 기준 근사)
                        if not (-50.0 <= float(box_data[0]) <= 50.0 and -50.0 <= float(box_data[1]) <= 50.0):
                            continue
                        c = int(box_data[7])
                        if 0 <= c < num_classes:
                            cls_present.add(c)
                except Exception:
                    pass
                scene_frames.append(sorted(cls_present))
            cache[scene] = scene_frames
            cache_dirty = True
            presence.extend([set(f) for f in scene_frames])

        if cache_dirty:
            try:   # 원자적 쓰기(tmp→rename) — DDP 프로세스 간 race에도 안전. 실패해도 무시(캐시일 뿐).
                tmp = cache_path.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(cache))
                tmp.rename(cache_path)
            except Exception:
                pass

        total = len(presence)
        class_frames = {c: [] for c in range(num_classes)}
        for i, cls_set in enumerate(presence):
            for c in cls_set:
                class_frames[c].append(i)
        rng = np.random.default_rng(seed)                # 고정 seed → 모든 DDP rank 동일 인덱스
        per_cls = int(round(total / num_classes))
        indices = []
        for c in range(num_classes):
            frames = class_frames[c]
            if not frames:
                continue
            indices.extend(rng.choice(frames, size=per_cls, replace=True).tolist())
        cls_counts = {c: len(class_frames[c]) for c in range(num_classes)}
        print(f"[CBGS] frames-with-class: {cls_counts}")
        return indices

    def train_dataloader(self):
        # 학습 로더: shuffle ON, 가변 길이 포인트 처리 위해 collate_fn_with_list 사용.
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn_with_list,
            **self.loader_config
        )

    def val_dataloader(self):
        # 검증 로더: shuffle OFF (순서 고정), collate는 동일.
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=collate_fn_with_list,
            **self.loader_config
        )


# ============================================================================
# dataset module registry shim (was: __init__.py + nuscenes_dataset_generated)
# ============================================================================

# get_data is now defined in this same module, so get_dataset_module_by_name
# returns this module itself; `.get_data` then resolves to the function above.
def get_dataset_module_by_name(name):
    # 과거엔 name으로 별도 모듈을 import했으나, 지금은 get_data가 이 파일 안에 있어
    # 자기 자신 모듈을 반환한다 → 호출부의 `.get_data`가 위 함수로 해소됨.
    return sys.modules[__name__]
