"""
Detection + Tracking Evaluation Script.

학습된 checkpoint로 다음 항목을 한 번에 평가:
  - Main Detection (mAP, NDS, mAVE, mATE, mASE, mAOE, mAAE)
  - Main Tracking  (AMOTA, AMOTP, MOTA, MOTP, IDS, FRAG)
  - Runtime Analysis (Model, Decoder, Tracker, Total)
  - Ablation 1: Score Threshold
  - Ablation 2: Distance Range
  - Ablation 3: Weather (Day/Night/Rain)
  - Ablation 4: Track Match Threshold
  - BEV Visualization (scene별 frame PNG)

Usage:
    python tools/evaluate.py \\
        checkpoint_path=/workspace/logs/checkpoints/.../best.ckpt

NOTE: This is a single self-contained file. The previously-separate helper
modules (gaussianmot/evaluation/{predictions,ablations,report}.py and
gaussianmot/utils/{visualization_full,visualization,color_utils}.py) have been
inlined here verbatim. The original modules are kept on disk as a fallback.
"""
import colorsys     # HSV→RGB 변환(track ID별 색상 생성)
import json          # 결과 metrics_summary.json 저장
import logging       # 진행 로그
import math          # atan2 등(GT yaw 복원)
import os            # 디렉토리 생성/경로 (fig6 출력 등)
import time          # 단계별 runtime 측정(perf_counter)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2           # BEV/카메라 시각화 렌더링(OpenCV, BGR 기준)
import hydra         # config 기반 datamodule/module instantiate
import lightning as L  # LightningModule 타입(ckpt 로드 대상)
import numpy as np
import rootutils     # .project-root 기준 sys.path 설정
import torch
from nuscenes.nuscenes import NuScenes  # sample_token→scene name 조회용 공식 SDK
from omegaconf import DictConfig, OmegaConf
from rich.console import Console        # 결과 표 출력(컬러 콘솔)
from rich.table import Table
from sklearn.decomposition import PCA   # fig6: Gaussian feature를 3채널로 투영
from sklearn.preprocessing import minmax_scale
from tqdm.auto import tqdm

# __file__ 기준 .project-root를 찾아 cwd/pythonpath 설정 → gaussianmot import 가능
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)


# (inlined from gaussianmot/utils/config.py)
# config의 ${mult:..}, ${last_token:..} 보간(interpolation)을 위한 커스텀 resolver 등록.
def register_new_resolvers():

    CUSTOM_RESOLVERS = {
        "mult": lambda x, y: x * y,                  # 두 값 곱(예: bev_h * scale)
        "last_token": lambda x: x.split(".")[-1],    # 점 구분 문자열의 마지막 토큰
    }

    # 중복 등록(같은 프로세스에서 두 번 호출) 방지
    for name, func in CUSTOM_RESOLVERS.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, func)


# (inlined from gaussianmot/utils/pca.py)
# fig6 전용: Gaussian feature map [B,C,H,W]을 PCA로 3채널(RGB)로 압축해 시각화.
def extract_pca_features(
    features: torch.Tensor,            # [B, C, H, W] (예: features_cam [1,128,200,200])
    n_components: int = 3,             # 결과 채널 수(RGB)
) -> torch.Tensor:
    # Ensure the input is on CPU and converted to numpy
    B, C, H, W = features.shape

    # Reshape features to (Batch, Channels, H*W)
    projected_heatmap = features.view(B, C, -1).cpu().numpy()  # [B, C, H*W]

    # Initialize output list
    output = []

    # Process each batch item
    for heatmap in projected_heatmap:                # heatmap: [C, H*W]
        # Transpose to make it (H*W, Channels) for PCA
        heatmap_transposed = heatmap.T               # [H*W, C] (픽셀=샘플, 채널=feature)

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(heatmap_transposed)  # [H*W, 3]

        # Normalize to 0-255 range
        pca_features = minmax_scale(pca_features, feature_range=(0, 255)).astype(np.uint8)

        # Reshape back to (n_components, H, W)
        pca_features = pca_features.T.reshape(n_components, H, W)  # [3, H, W]

        # Convert to torch tensor
        pca_features = torch.from_numpy(pca_features)
        output.append(pca_features)

    # Stack the batch
    pca_features = torch.stack(output, dim=0)        # [B, 3, H, W]

    return pca_features


# (inlined from gaussianmot/utils/consts.py)
# https://github.com/robot-learning-freiburg/BEVCar/blob/29cacda3bc5416d47428c1d0f017527acad34f90/custom_nuscenes_splits.py#L157
# Weather ablation(Ablation 3)용 val scene 분류: day/night/rain 별 scene name 리스트.
# 각 prediction의 scene_name이 어느 그룹에 속하는지로 필터링한다.
VALIDATION_DRN_SPLITS = {
    "day": [
        "scene-0003",
        "scene-0012",
        "scene-0013",
        "scene-0014",
        "scene-0015",
        "scene-0016",
        "scene-0017",
        "scene-0018",
        "scene-0035",
        "scene-0036",
        "scene-0038",
        "scene-0039",
        "scene-0092",
        "scene-0093",
        "scene-0094",
        "scene-0095",
        "scene-0096",
        "scene-0097",
        "scene-0098",
        "scene-0099",
        "scene-0100",
        "scene-0101",
        "scene-0102",
        "scene-0103",
        "scene-0104",
        "scene-0105",
        "scene-0106",
        "scene-0107",
        "scene-0108",
        "scene-0109",
        "scene-0110",
        "scene-0221",
        "scene-0268",
        "scene-0269",
        "scene-0270",
        "scene-0271",
        "scene-0272",
        "scene-0273",
        "scene-0274",
        "scene-0275",
        "scene-0276",
        "scene-0277",
        "scene-0278",
        "scene-0329",
        "scene-0330",
        "scene-0331",
        "scene-0332",
        "scene-0344",
        "scene-0345",
        "scene-0346",
        "scene-0519",
        "scene-0520",
        "scene-0521",
        "scene-0522",
        "scene-0523",
        "scene-0524",
        "scene-0552",
        "scene-0553",
        "scene-0554",
        "scene-0555",
        "scene-0556",
        "scene-0557",
        "scene-0558",
        "scene-0559",
        "scene-0560",
        "scene-0561",
        "scene-0562",
        "scene-0563",
        "scene-0564",
        "scene-0565",
        "scene-0770",
        "scene-0771",
        "scene-0775",
        "scene-0777",
        "scene-0778",
        "scene-0780",
        "scene-0781",
        "scene-0782",
        "scene-0783",
        "scene-0784",
        "scene-0794",
        "scene-0795",
        "scene-0796",
        "scene-0797",
        "scene-0798",
        "scene-0799",
        "scene-0800",
        "scene-0802",
        "scene-0916",
        "scene-0917",
        "scene-0919",
        "scene-0920",
        "scene-0921",
        "scene-0922",
        "scene-0923",
        "scene-0924",
        "scene-0925",
        "scene-0926",
        "scene-0927",
        "scene-0928",
        "scene-0929",
        "scene-0930",
        "scene-0931",
        "scene-0962",
        "scene-0963",
        "scene-0966",
        "scene-0967",
        "scene-0968",
        "scene-0969",
        "scene-0971",
        "scene-0972",
    ],
    "night": [
        "scene-1059",
        "scene-1060",
        "scene-1061",
        "scene-1062",
        "scene-1063",
        "scene-1064",
        "scene-1065",
        "scene-1066",
        "scene-1067",
        "scene-1068",
        "scene-1069",
        "scene-1070",
        "scene-1071",
        "scene-1072",
        "scene-1073",
    ],
    "rain": [
        "scene-0625",
        "scene-0626",
        "scene-0627",
        "scene-0629",
        "scene-0630",
        "scene-0632",
        "scene-0633",
        "scene-0634",
        "scene-0635",
        "scene-0636",
        "scene-0637",
        "scene-0638",
        "scene-0904",
        "scene-0905",
        "scene-0906",
        "scene-0907",
        "scene-0908",
        "scene-0909",
        "scene-0910",
        "scene-0911",
        "scene-0912",
        "scene-0913",
        "scene-0914",
        "scene-0915",
    ],
}


# ###########################################################################
# color_utils.py  (ID 기반 색상 매핑 유틸리티, OpenCV BGR 기준)
#   - 역할: track ID/class ID를 시각화용 고유 색으로 변환. 평가 metric과는 무관, 그림용.
# ###########################################################################
def track_id_to_color(track_id: int, brightness: float = 0.9, saturation: float = 0.7):
    """
    Track ID → 고유 BGR 색상 (deterministic, OpenCV 호환).

    Args:
        track_id: 정수 ID (음수면 회색 반환)
        brightness: HSV value (0~1)
        saturation: HSV saturation (0~1)

    Returns:
        (b, g, r) tuple in [0, 255]
    """
    if track_id < 0:
        return (128, 128, 128)   # 미매칭(-1) → 회색

    golden_ratio = 0.618033988749895    # 황금비로 hue를 분산시켜 인접 ID끼리 색 충돌 최소화
    hue = (track_id * golden_ratio) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (int(b * 255), int(g * 255), int(r * 255))   # OpenCV는 BGR이므로 b,g,r 순으로 반환


# BGR 기준 클래스별 색상
CLASS_COLORS = {
    0: (0, 255, 0),       # vehicle: green
    1: (0, 165, 255),     # pedestrian: orange
    2: (0, 255, 255),     # bicycle: yellow
}


def class_to_color(class_id: int):
    """클래스 ID → BGR 색상."""
    return CLASS_COLORS.get(int(class_id), (200, 200, 200))   # 미정의 클래스는 회색


# ###########################################################################
# Visualization  (BEV primitive + 카메라/composite 통합 섹션)
#   원래 visualization.py / visualization_full.py 두 모듈이었으나 하나로 통합.
#   구성은 두 레이어:
#     [1] BEV primitive (아래) : lidar metric ↔ BEV 픽셀 변환 + BEV 평면에 박스/궤적 그리기.
#         좌표계: nuScenes 컨벤션(lidar +X forward=위, +Y left=왼쪽). metric/labels와 일관.
#     [2] 카메라 3D 투영 + composite (뒤 "─ [2] ─" 구분선 이후): 6카메라에 3D 박스를 투영하고
#         [1]의 BEV primitive를 재사용해 4종 BEV 패널과 합쳐 한 프레임 통합 시각화를 만든다.
#   실제 eval은 [2]의 render_composite_frame을 프레임마다 저장한다([1]은 [2]가 가져다 씀).
# ###########################################################################
def world_to_bev_pixel(
    x: float, y: float,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    bev_w: int, bev_h: int,
) -> Tuple[int, int]:
    """
    LiDAR(metric) coord → BEV pixel coord.

    nuScenes 컨벤션 (labels.py의 view 매트릭스와 일관):
      - lidar +X (forward) → BEV 이미지 위쪽 (py 작음)
      - lidar +Y (left)    → BEV 이미지 왼쪽 (px 작음)

    매핑:
      px = (y_max - y) / (y_max - y_min) * bev_w
      py = (x_max - x) / (x_max - x_min) * bev_h
    """
    px = int((y_max - y) / (y_max - y_min) * bev_w)   # +Y(왼쪽)일수록 px 작음 → 이미지 왼쪽
    py = int((x_max - x) / (x_max - x_min) * bev_h)   # +X(전방)일수록 py 작음 → 이미지 위쪽
    # 경계 클램핑(범위 밖 점도 가장자리에 찍힘)
    px = max(0, min(bev_w - 1, px))
    py = max(0, min(bev_h - 1, py))
    return px, py


def get_box_corners_2d(
    cx: float, cy: float, w: float, l: float, yaw: float
) -> np.ndarray:
    """
    2D bbox (BEV plane) corners.

    Args:
        cx, cy: center
        w: width (y axis)
        l: length (x axis)
        yaw: rotation (rad)

    Returns:
        [4, 2] corners (counterclockwise from front-left)
    """
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])   # 2D 회전행렬(yaw)

    # Local corners (length=x, width=y)
    half_l, half_w = l / 2, w / 2
    corners_local = np.array([                # 중심 기준 로컬 4모서리(반시계, front-left부터)
        [ half_l,  half_w],   # front-left
        [ half_l, -half_w],   # front-right
        [-half_l, -half_w],   # rear-right
        [-half_l,  half_w],   # rear-left
    ])

    # Rotate + translate
    corners = (R @ corners_local.T).T + np.array([cx, cy])   # 회전 후 중심으로 평행이동 → [4,2]
    return corners


def create_bev_canvas(
    bev_h: int = 200, bev_w: int = 200,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    기본 BEV 캔버스 생성 (어두운 배경 + ego + grid).

    컨벤션: 차량 forward(+X)가 이미지 위쪽, left(+Y)가 이미지 왼쪽.
    """
    canvas = np.full((bev_h, bev_w, 3), bg_color, dtype=np.uint8)   # [H,W,3] 어두운 배경

    # Ego vehicle (중앙) + 정면 화살표 (위쪽)
    ego_x, ego_y = bev_w // 2, bev_h // 2     # ego는 항상 BEV 중앙(lidar 원점)
    cv2.circle(canvas, (ego_x, ego_y), 4, (255, 255, 255), -1)
    cv2.line(canvas, (ego_x, ego_y), (ego_x, ego_y - 12), (255, 255, 255), 2)        # 전방(위) 화살표 몸통
    cv2.line(canvas, (ego_x, ego_y - 12), (ego_x - 4, ego_y - 8), (255, 255, 255), 2)  # 화살촉 왼쪽
    cv2.line(canvas, (ego_x, ego_y - 12), (ego_x + 4, ego_y - 8), (255, 255, 255), 2)  # 화살촉 오른쪽

    # Grid (20m 간격) — ±100m 정규화 기준 격자선
    for offset in [-40, -20, 20, 40]:
        x_off_pix = int(offset / 100.0 * bev_w + bev_w / 2)
        y_off_pix = int(offset / 100.0 * bev_h + bev_h / 2)
        cv2.line(canvas, (x_off_pix, 0), (x_off_pix, bev_h), (60, 60, 60), 1)
        cv2.line(canvas, (0, y_off_pix), (bev_w, y_off_pix), (60, 60, 60), 1)

    return canvas


def draw_bev_boxes(
    canvas: np.ndarray,
    boxes_3d: np.ndarray,            # [N, 9] (x,y,z,w,l,h,yaw,vx,vy)
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    x_min: float = -50, x_max: float = 50,
    y_min: float = -50, y_max: float = 50,
    show_score: bool = True,
    show_heading: bool = True,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Detection bbox를 BEV에 그림.

    Args:
        canvas: [H, W, 3] BEV 이미지
        boxes_3d: [N, 9]
        scores, labels: optional

    Returns:
        bbox 그려진 canvas
    """
    bev_h, bev_w = canvas.shape[:2]

    if len(boxes_3d) == 0:          # 검출 0개면 빈 캔버스 그대로 반환
        return canvas

    if isinstance(boxes_3d, torch.Tensor):     # tensor면 numpy로
        boxes_3d = boxes_3d.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    for i, box in enumerate(boxes_3d):
        cx, cy, _, w, l, _, yaw = box[:7]      # [N,9]에서 x,y,(z생략),w,l,(h생략),yaw_bev
        label = int(labels[i]) if labels is not None else 0
        color = class_to_color(label)          # 클래스별 색

        # 4 corners (metric) → BEV pixel
        corners_metric = get_box_corners_2d(cx, cy, w, l, yaw)
        corners_pix = np.array([
            world_to_bev_pixel(c[0], c[1], x_min, x_max, y_min, y_max, bev_w, bev_h)
            for c in corners_metric
        ], dtype=np.int32)

        # Polygon — 박스 외곽 그리기
        cv2.polylines(canvas, [corners_pix], isClosed=True,
                      color=color, thickness=line_thickness)

        # Heading (front edge 강조) — 중심→전면중점 선으로 진행방향 표시
        if show_heading:
            front_mid = ((corners_pix[0] + corners_pix[1]) / 2).astype(np.int32)  # front-left/right 중점
            center_pix = world_to_bev_pixel(cx, cy, x_min, x_max, y_min, y_max, bev_w, bev_h)
            cv2.line(canvas, tuple(center_pix), tuple(front_mid), color, line_thickness + 1)

        # Score 라벨 — 박스 위에 confidence 표시
        if show_score and scores is not None:
            text_pos = (corners_pix[3][0], corners_pix[3][1] - 2)
            cv2.putText(canvas, f"{scores[i]:.2f}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return canvas


def draw_bev_tracks(
    canvas: np.ndarray,
    boxes_3d: np.ndarray,
    track_ids: np.ndarray,
    track_history: Dict[int, List[Tuple[float, float]]],
    x_min: float = -50, x_max: float = 50,
    y_min: float = -50, y_max: float = 50,
    show_id: bool = True,
    trail_thickness: int = 1,
    box_thickness: int = 1,
    max_trail_length: int = 10,
) -> np.ndarray:
    """
    Tracking 결과를 BEV에 그림 (bbox + ID + trail).

    Args:
        canvas: [H, W, 3]
        boxes_3d: [N, 9]
        track_ids: [N] (-1이면 unmatched)
        track_history: {id: [(x, y), ...]} - 과거 위치
    """
    bev_h, bev_w = canvas.shape[:2]

    if isinstance(boxes_3d, torch.Tensor):
        boxes_3d = boxes_3d.cpu().numpy()
    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.cpu().numpy()

    # 1. Trail 먼저 (bbox 뒤로 가도록) — ID별 과거 위치 궤적
    for tid, positions in track_history.items():
        if len(positions) < 2:        # 점 1개면 선 못 그림
            continue
        color = track_id_to_color(int(tid))     # 같은 ID는 항상 같은 색
        # 최근 max_trail_length만 사용(오래된 궤적 잘라냄)
        positions = positions[-max_trail_length:]
        pts = np.array([
            world_to_bev_pixel(x, y, x_min, x_max, y_min, y_max, bev_w, bev_h)
            for x, y in positions
        ], dtype=np.int32)
        for j in range(1, len(pts)):
            # 옛날 점일수록 얇게(최근일수록 굵게) 그려 진행 방향 강조
            alpha = j / len(pts)
            t = max(1, int(trail_thickness * alpha))
            cv2.line(canvas, tuple(pts[j-1]), tuple(pts[j]), color, t)

    # 2. Current bbox + ID — 현재 프레임의 추적 박스
    for i, box in enumerate(boxes_3d):
        cx, cy, _, w, l, _, yaw = box[:7]    # [N,9]에서 박스 파라미터
        tid = int(track_ids[i])              # 이 박스의 track ID(-1=미매칭)
        color = track_id_to_color(tid)

        corners_metric = get_box_corners_2d(cx, cy, w, l, yaw)
        corners_pix = np.array([
            world_to_bev_pixel(c[0], c[1], x_min, x_max, y_min, y_max, bev_w, bev_h)
            for c in corners_metric
        ], dtype=np.int32)

        cv2.polylines(canvas, [corners_pix], isClosed=True,
                      color=color, thickness=box_thickness)

        # Heading
        front_mid = ((corners_pix[0] + corners_pix[1]) / 2).astype(np.int32)
        center_pix = world_to_bev_pixel(cx, cy, x_min, x_max, y_min, y_max, bev_w, bev_h)
        cv2.line(canvas, tuple(center_pix), tuple(front_mid), color, box_thickness + 1)

        # ID 라벨
        if show_id and tid >= 0:
            text_pos = (corners_pix[3][0], corners_pix[3][1] - 2)
            cv2.putText(canvas, f"ID:{tid}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────
# ─ [2] 카메라 3D 투영 + composite ─  (위 [1] BEV primitive 위에 쌓는 상위 레이어)
#   6개 카메라 이미지에 3D 박스를 투영(lidar2img)하고, 4종 BEV 패널([1] 재사용)과 합쳐
#   한 프레임짜리 통합 정성 시각화 이미지를 만든다. eval이 프레임마다 저장하는 게 이쪽.
# ─────────────────────────────────────────────────────────────────────────
# ImageNet normalization (역변환용) — 입력 이미지는 학습 시 정규화돼 있어 시각화 전 복원 필요
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# nuScenes camera 순서 (보통) — 6개 카메라(이미지 텐서 채널 순서와 일치)
CAMERA_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


# ---------------------------------------------------------------------------
# 이미지 역정규화
# ---------------------------------------------------------------------------
def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    [3, H, W] normalized tensor → [H, W, 3] uint8 BGR (OpenCV)
    """
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [3,H,W]→[H,W,3] (CHW→HWC)
    img = img * IMAGENET_STD + IMAGENET_MEAN     # 정규화 역변환(표준화 복원)
    img = (img * 255).clip(0, 255).astype(np.uint8)   # [0,1]→[0,255] uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # RGB→BGR(OpenCV)
    return img


# ---------------------------------------------------------------------------
# 3D bbox → 2D image projection
# ---------------------------------------------------------------------------
def get_3d_box_corners(cx, cy, cz, w, l, h, yaw):
    """
    3D bbox 8 corners (lidar frame).
    Returns: [8, 3] in order: bottom 4 + top 4
    """
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])   # z축 회전(yaw)

    half_l, half_w, half_h = l / 2, w / 2, h / 2
    corners_local = np.array([      # 중심 기준 로컬 8모서리(아래 4 + 위 4)
        # bottom
        [ half_l,  half_w, -half_h],
        [ half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h],
        [-half_l,  half_w, -half_h],
        # top
        [ half_l,  half_w,  half_h],
        [ half_l, -half_w,  half_h],
        [-half_l, -half_w,  half_h],
        [-half_l,  half_w,  half_h],
    ])

    corners = (R @ corners_local.T).T + np.array([cx, cy, cz])   # 회전+이동 → lidar frame [8,3]
    return corners  # [8, 3]


def project_3d_to_2d(corners_3d: np.ndarray, lidar2img: np.ndarray) -> Optional[np.ndarray]:
    """
    [8, 3] corners → [8, 2] pixel coords.
    Z<=0 (camera 뒤) corners 있으면 None 반환 (전체 box skip).
    """
    N = corners_3d.shape[0]
    homog = np.concatenate([corners_3d, np.ones((N, 1))], axis=1)  # [8, 4] 동차좌표
    img_pts = lidar2img @ homog.T  # [4,4]@[4,8] → [4, 8] 이미지 동차좌표

    # depth check — 한 모서리라도 카메라 뒤(z<=0.1)면 박스 전체 skip
    z = img_pts[2, :]
    if np.any(z <= 0.1):
        return None

    pts_2d = img_pts[:2, :] / z[None, :]   # 원근 분할(homogeneous divide) → 픽셀
    return pts_2d.T  # [8, 2]


def draw_3d_box_on_image(
    img: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: Optional[str] = None,
):
    """
    8 corners → image에 wireframe 3D bbox 그리기.
    box가 화면에 거의 안 보이면 skip.
    """
    H, W = img.shape[:2]

    # 박스의 화면 가시성 체크: 적어도 corner 한두 개가 이미지 내부에 있어야 함
    in_bounds = (
        (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < W) &
        (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < H)
    )
    if in_bounds.sum() < 2:     # 화면 내 모서리 2개 미만이면 거의 안 보임
        return  # 거의 안 보이면 skip (label도 안 그림)

    corners_2d = corners_2d.astype(np.int32)

    # 바닥 4개 + 천장 4개 + 수직 4개 = 12 edges (3D wireframe 박스)
    edges = [
        # bottom
        (0,1), (1,2), (2,3), (3,0),
        # top
        (4,5), (5,6), (6,7), (7,4),
        # vertical
        (0,4), (1,5), (2,6), (3,7),
    ]

    for i, j in edges:
        p1, p2 = tuple(corners_2d[i]), tuple(corners_2d[j])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # Label (top-front-left corner 위에)
    if label is not None:
        x, y = corners_2d[4]
        if 0 <= x < W and 0 <= y < H:
            cv2.putText(img, label, (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# GT bbox 색상 (BGR)
GT_COLOR = (255, 150, 50)  # 파란계열


def render_camera_view(
    img_tensor: torch.Tensor,
    lidar2img: np.ndarray,
    boxes_3d: np.ndarray,         # [N, 9] (BEV plane yaw로 들어옴)
    track_ids: Optional[np.ndarray] = None,  # None이면 detection mode
    scores: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,   # [M, 12] (lidar frame)
    cam_name: str = "",
) -> np.ndarray:
    """
    카메라 1개에 3D bbox projection.
    GT (파랑) + Pred (초록 또는 track id 색)을 함께 그림.
    """
    img = denormalize_image(img_tensor)    # [3,H,W] tensor → [H,W,3] uint8 BGR
    H, W = img.shape[:2]

    # ============================================================
    # 1. GT boxes 먼저 그림 (Pred 위에 안 가리도록 뒤로)
    # ============================================================
    if gt_boxes is not None and len(gt_boxes) > 0:
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()

        for box in gt_boxes:        # gt_box: 12-field [cx,cy,l,w,cz,h,yaw_lidar,class,vis,vx,vy,inst_id]
            if len(box) < 9:
                continue
            cx, cy, l, w, cz, h, yaw_lidar = box[0:7]   # 주의: GT는 (l,w) 순, yaw는 lidar frame
            instance_id = int(box[11]) if len(box) > 11 else -1

            # GT는 이미 lidar frame yaw → BEV로 안 바꿔도 됨
            # gt_box[6]이 lidar frame yaw임 (nuscenes_dataset.py에서 직접 저장)
            corners_3d = get_3d_box_corners(cx, cy, cz, w, l, h, yaw_lidar)
            corners_2d = project_3d_to_2d(corners_3d, lidar2img)   # lidar→이미지 투영 [8,2]

            if corners_2d is None:      # 카메라 뒤 → skip
                continue

            label = f"GT:{instance_id}" if instance_id >= 0 else None
            draw_3d_box_on_image(img, corners_2d, GT_COLOR, thickness=2, label=label)  # GT=파랑

    # ============================================================
    # 2. Prediction boxes (Detection / Tracking)
    # ============================================================
    if len(boxes_3d) > 0:
        if isinstance(boxes_3d, torch.Tensor):
            boxes_3d = boxes_3d.cpu().numpy()
        if track_ids is not None and isinstance(track_ids, torch.Tensor):
            track_ids = track_ids.cpu().numpy()

        # 학습 부족 모델 대응: dimensions 너무 작으면 viz용 최소값 보장
        MIN_W, MIN_L, MIN_H = 1.6, 4.0, 1.5

        for i, box in enumerate(boxes_3d):
            cx, cy, cz, w, l, h, yaw_bev, _, _ = box[:9]   # pred [N,9]: yaw는 BEV plane yaw

            w_viz = max(abs(w), MIN_W)     # 박스 크기 하한(작은 박스 시각화 보정)
            l_viz = max(abs(l), MIN_L)
            h_viz = max(abs(h), MIN_H)

            # BEV yaw → lidar yaw (labels.py와 일관): yaw_lidar = -yaw_bev - π/2
            yaw_lidar = -yaw_bev - np.pi / 2

            corners_3d = get_3d_box_corners(cx, cy, cz, w_viz, l_viz, h_viz, yaw_lidar)
            corners_2d = project_3d_to_2d(corners_3d, lidar2img)

            if corners_2d is None:
                continue

            if track_ids is not None and track_ids[i] >= 0:   # tracking 모드: ID 색/라벨
                color = track_id_to_color(int(track_ids[i]))
                label = f"ID:{int(track_ids[i])}"
            else:                                              # detection 모드: 클래스 색 + score
                color = class_to_color(0)  # vehicle
                label = f"{scores[i]:.2f}" if scores is not None else None

            draw_3d_box_on_image(img, corners_2d, color, thickness=2, label=label)

    # Camera name
    if cam_name:
        cv2.putText(img, cam_name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


def render_multi_view(
    images: torch.Tensor,         # [6, 3, H, W]
    lidar2imgs: np.ndarray,        # [6, 4, 4]
    boxes_3d: np.ndarray,
    track_ids: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,
    cam_names: List[str] = None,
    grid_h: int = 2,
    grid_w: int = 3,
    target_w: int = 480,
) -> np.ndarray:
    """
    6 cameras → 2x3 grid 이미지.
    GT (파랑) + Pred (초록/track 색) 동시 표시.
    """
    if cam_names is None:
        cam_names = CAMERA_NAMES

    cam_imgs = []
    for c in range(6):              # 카메라 6개 각각에 GT+Pred 박스 투영
        img = render_camera_view(
            images[c], lidar2imgs[c],    # images[6,3,H,W]의 c번째, lidar2imgs[6,4,4]의 c번째
            boxes_3d, track_ids, scores,
            gt_boxes=gt_boxes,
            cam_name=cam_names[c],
        )
        cam_imgs.append(img)

    # 통일된 크기로 resize
    H, W = cam_imgs[0].shape[:2]
    target_h = int(H * target_w / W)
    cam_imgs = [cv2.resize(img, (target_w, target_h)) for img in cam_imgs]

    # 2x3 grid — 카메라 6개를 2행 3열로 배치
    rows = []
    for r in range(grid_h):
        row = np.hstack(cam_imgs[r * grid_w : (r + 1) * grid_w])   # 가로 3개 결합
        rows.append(row)
    grid = np.vstack(rows)      # 세로 2행 결합

    # Legend (좌상단에 작게)
    legend_y = grid.shape[0] - 50
    legend_x = 10
    # GT 박스 색
    cv2.rectangle(grid, (legend_x, legend_y), (legend_x + 20, legend_y + 12), GT_COLOR, -1)
    cv2.putText(grid, "GT", (legend_x + 25, legend_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Pred (Det) 색
    cv2.rectangle(grid, (legend_x + 60, legend_y), (legend_x + 80, legend_y + 12), (0, 255, 0), -1)
    cv2.putText(grid, "Pred(Det)", (legend_x + 85, legend_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Tracking 색
    cv2.rectangle(grid, (legend_x + 175, legend_y), (legend_x + 195, legend_y + 12), (255, 0, 200), -1)
    cv2.putText(grid, "Pred(Track ID)", (legend_x + 200, legend_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return grid


# ---------------------------------------------------------------------------
# Radar BEV
# ---------------------------------------------------------------------------
def render_radar_bev(
    radar_points: np.ndarray,    # [N, F] (xyz가 처음 3채널)
    bev_h: int = 400, bev_w: int = 400,
    bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
    point_size: int = 2,
    point_color: Tuple[int, int, int] = (0, 200, 255),  # cyan
) -> np.ndarray:
    """Radar points을 BEV에 점으로 표시."""
    canvas = create_bev_canvas(bev_h, bev_w)

    if isinstance(radar_points, torch.Tensor):
        radar_points = radar_points.cpu().numpy()

    if len(radar_points) == 0:
        return canvas

    x_min, x_max, y_min, y_max = bev_range

    for pt in radar_points:        # radar point [N,F]: 앞 2채널이 x,y(lidar metric)
        x, y = pt[0], pt[1]
        if not (x_min <= x < x_max and y_min <= y < y_max):   # BEV 범위 밖 점 제외
            continue
        px, py = world_to_bev_pixel(x, y, x_min, x_max, y_min, y_max, bev_w, bev_h)
        cv2.circle(canvas, (px, py), point_size, point_color, -1)   # cyan 점

    cv2.putText(canvas, "Radar BEV", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


# ---------------------------------------------------------------------------
# GT BEV (gt_box 기반)
# ---------------------------------------------------------------------------
def render_gt_bev(
    gt_boxes: np.ndarray,           # [N, 12] from gt_box.npz
    bev_h: int = 400, bev_w: int = 400,
    bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
    show_id: bool = True,
) -> np.ndarray:
    """
    GT bounding box를 BEV에 그림.
    gt_box 12 fields: [cx, cy, l, w, cz, h, yaw_lidar, class, vis, vx, vy, instance_id]

    주의: gt_box[6]는 lidar frame yaw (BEV가 아님).
    BEV plane으로 변환: yaw_bev = -yaw_lidar - π/2
    """
    canvas = create_bev_canvas(bev_h, bev_w)

    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()

    if len(gt_boxes) == 0:
        cv2.putText(canvas, "GT BEV", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    x_min, x_max, y_min, y_max = bev_range

    for box in gt_boxes:        # gt_box 12-field: [cx,cy,l,w,cz,h,yaw_lidar,class,vis,vx,vy,inst_id]
        if len(box) < 9:
            continue
        cx, cy, l, w, cz, h, yaw_lidar = box[0:7]
        class_id = int(box[7]) if len(box) > 7 else 0       # field[7]=class
        instance_id = int(box[11]) if len(box) > 11 else -1  # field[11]=instance_id

        # GT의 yaw는 lidar frame → BEV plane으로 변환(yaw_lidar=-yaw_bev-π/2의 역)
        yaw_bev = -yaw_lidar - np.pi / 2

        color = class_to_color(class_id)

        # 4 corners (BEV)
        corners = get_box_corners_2d(cx, cy, w, l, yaw_bev)
        corners_pix = np.array([
            world_to_bev_pixel(c[0], c[1], x_min, x_max, y_min, y_max, bev_w, bev_h)
            for c in corners
        ], dtype=np.int32)

        cv2.polylines(canvas, [corners_pix], isClosed=True, color=color, thickness=2)

        # Heading
        front_mid = ((corners_pix[0] + corners_pix[1]) / 2).astype(np.int32)
        center_pix = world_to_bev_pixel(cx, cy, x_min, x_max, y_min, y_max, bev_w, bev_h)
        cv2.line(canvas, tuple(center_pix), tuple(front_mid), color, 2)

        # Instance ID
        if show_id and instance_id >= 0:
            cv2.putText(canvas, f"GT:{instance_id}",
                        (corners_pix[3][0], corners_pix[3][1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv2.putText(canvas, "GT BEV", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


# ---------------------------------------------------------------------------
# Detection BEV (bbox + score)
# ---------------------------------------------------------------------------
def render_detection_bev(
    boxes_3d: np.ndarray,
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    bev_h: int = 400, bev_w: int = 400,
    bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
) -> np.ndarray:
    """Detection 결과 BEV (score 표시)."""
    canvas = create_bev_canvas(bev_h, bev_w)
    canvas = draw_bev_boxes(
        canvas, boxes_3d, scores, labels,
        x_min=bev_range[0], x_max=bev_range[1],
        y_min=bev_range[2], y_max=bev_range[3],
        show_score=True, show_heading=True, line_thickness=2,
    )
    cv2.putText(canvas, "Detection BEV", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


# ---------------------------------------------------------------------------
# Tracking BEV (track ID + trail)
# ---------------------------------------------------------------------------
def render_tracking_bev(
    boxes_3d: np.ndarray,
    track_ids: np.ndarray,
    track_history: Dict[int, List[Tuple[float, float]]],
    bev_h: int = 400, bev_w: int = 400,
    bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
    max_trail_length: int = 10,
) -> np.ndarray:
    """Tracking 결과 BEV (ID + trail)."""
    canvas = create_bev_canvas(bev_h, bev_w)
    canvas = draw_bev_tracks(
        canvas, boxes_3d, track_ids, track_history,
        x_min=bev_range[0], x_max=bev_range[1],
        y_min=bev_range[2], y_max=bev_range[3],
        max_trail_length=max_trail_length,
    )
    cv2.putText(canvas, "Tracking BEV", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


# ---------------------------------------------------------------------------
# Composite layout
# ---------------------------------------------------------------------------
def render_composite_frame(
    images: torch.Tensor,           # [6, 3, H, W]
    lidar2imgs: np.ndarray,          # [6, 4, 4]
    radar_points: np.ndarray,        # [N, F]
    gt_boxes: np.ndarray,            # [N, 12]
    pred_boxes_3d: np.ndarray,       # [M, 9]
    pred_scores: np.ndarray,         # [M]
    pred_labels: np.ndarray,         # [M]
    track_ids: np.ndarray,           # [M]
    track_history: Dict[int, List],
    bev_h: int = 300, bev_w: int = 300,
    cam_target_w: int = 320,
    bev_range: Tuple[float, float, float, float] = (-50, 50, -50, 50),
    max_trail_length: int = 10,
) -> np.ndarray:
    """
    한 frame의 전체 통합 시각화.

    Layout:
      Row 1: [Multi-view 6 cameras grid (2x3)]
      Row 2: [Radar BEV] [GT BEV] [Det BEV] [Track BEV]
    """
    # 상단: 6 cameras (GT + Pred) → 2x3 카메라 그리드
    cam_grid = render_multi_view(
        images, lidar2imgs,
        pred_boxes_3d, track_ids, pred_scores,
        gt_boxes=gt_boxes,
        target_w=cam_target_w,
    )

    # 하단: 4 BEV panels (Radar / GT / Detection / Tracking)
    radar_bev = render_radar_bev(radar_points, bev_h, bev_w, bev_range)
    gt_bev = render_gt_bev(gt_boxes, bev_h, bev_w, bev_range)
    det_bev = render_detection_bev(
        pred_boxes_3d, pred_scores, pred_labels,
        bev_h, bev_w, bev_range,
    )
    track_bev = render_tracking_bev(
        pred_boxes_3d, track_ids, track_history,
        bev_h, bev_w, bev_range, max_trail_length,
    )

    # BEV panels 사이에 흰색 경계선 추가
    border_color = (255, 255, 255)
    border_thickness = 2

    # 각 panel 우측에 경계선 그리기 (마지막은 제외)
    for panel in [radar_bev, gt_bev, det_bev]:
        cv2.line(panel, (panel.shape[1] - 1, 0),
                 (panel.shape[1] - 1, panel.shape[0]),
                 border_color, border_thickness)

    bev_row = np.hstack([radar_bev, gt_bev, det_bev, track_bev])

    # cam grid와 BEV row 사이 수평 경계선
    cv2.line(bev_row, (0, 0), (bev_row.shape[1], 0),
             border_color, border_thickness)

    # 폭 맞추기 (cam_grid의 width와 bev_row의 width 동일하게)
    cam_w = cam_grid.shape[1]
    bev_w_total = bev_row.shape[1]

    if cam_w != bev_w_total:
        # 더 좁은 쪽을 늘림
        if cam_w < bev_w_total:
            new_h = int(cam_grid.shape[0] * bev_w_total / cam_w)
            cam_grid = cv2.resize(cam_grid, (bev_w_total, new_h))
        else:
            new_h = int(bev_row.shape[0] * cam_w / bev_w_total)
            bev_row = cv2.resize(bev_row, (cam_w, new_h))

    # 수직 결합
    composite = np.vstack([cam_grid, bev_row])
    return composite


# ###########################################################################
# predictions.py  (Prediction collector with runtime measurement)
#   - 역할: val 전체를 한 번 순회하며 model→decoder→tracker를 돌려 per-sample 예측을
#           수집하고, 각 단계 runtime(ms)을 측정한다. eval 파이프라인의 핵심.
# ###########################################################################
def get_scene_name(nusc: NuScenes, sample_token: str) -> str:
    """sample_token → scene name. (tracker reset/weather ablation 경계 판단용)"""
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    return scene["name"]


# 차량류 클래스 id (yaw-속도 정렬 대상; cone/barrier/ped 제외)
_DIAG_VEHICLE_CLS = (0, 1, 2, 3, 4, 6, 7)


def _diag_postprocess_boxes(result, radar_pts, vel_from_radar=False, yaw_from_velocity=False,
                            radar_vel_cols=(15, 16), radar_xy_cols=(0, 1)):
    """[진단 전용] 상한(headroom) 측정 후처리 — 논문 방법 아님, 재학습 설계 근거용.

    ① vel_from_radar: 박스 반경 내 radar 보정속도(compensated, ego frame = 박스와 동일 프레임)가
       있으면 예측 속도를 그 평균으로 치환 → "radar 측정 직접 사용" 시 mAVE 하한 측정.
    ② yaw_from_velocity: |v|>1.5m/s 차량류 박스의 yaw를 속도 방향으로 정렬 → 이동 물체의
       yaw-속도 정렬은 물리적 사실이므로 mAOE 개선 상한 측정. boxes_3d의 yaw는 BEV 규약이라
       lidar 방향각 atan2(vy,vx)를 bev로 변환(-x-π/2, involution)해 대입.
    """
    boxes = result["boxes_3d"]          # [K,9] (x,y,z,w,l,h,yaw_bev,vx,vy) — torch.Tensor
    labels = result["labels"]
    if boxes.shape[0] == 0:
        return result
    boxes = boxes.clone()

    if vel_from_radar and radar_pts is not None and radar_pts.numel() > 0:
        pts = radar_pts.to(boxes.device).float()
        px, py = pts[:, radar_xy_cols[0]], pts[:, radar_xy_cols[1]]
        pvx, pvy = pts[:, radar_vel_cols[0]], pts[:, radar_vel_cols[1]]
        project = (vel_from_radar == "project")   # True/"copy"면 단순 복사, "project"면 heading 재구성
        for i in range(boxes.shape[0]):
            r = float(max(boxes[i, 3], boxes[i, 4]) / 2.0 + 0.5)   # 박스 외접원 반경 + 여유
            d2 = (px - boxes[i, 0]) ** 2 + (py - boxes[i, 1]) ** 2
            m = d2 < r * r
            if not bool(m.any()):
                continue
            if project:
                # [진단2] radial→full 재구성: doppler는 시선방향 성분만 측정하므로
                #   v_full = (s / ĥ·r̂) ĥ, s = v_comp·r̂ (점별), ĥ = 예측 heading.
                #   ĥ 180° flip에 불변(부호 상쇄). |ĥ·r̂|<0.3(측면 통과)이면 발산 → 복사 fallback.
                lidar_yaw = -float(boxes[i, 6]) - np.pi / 2         # BEV→lidar yaw
                hx, hy = np.cos(lidar_yaw), np.sin(lidar_yaw)
                pxm, pym = px[m], py[m]
                rn = (pxm ** 2 + pym ** 2).sqrt().clamp(min=1e-3)
                rx, ry = pxm / rn, pym / rn                          # 점별 시선방향 단위벡터
                s = pvx[m] * rx + pvy[m] * ry                        # 점별 radial 속도(부호 있음)
                hdotr = hx * rx + hy * ry
                ok = hdotr.abs() > 0.3
                if bool(ok.any()):
                    scale = (s[ok] / hdotr[ok]).median()             # 점별 재구성 후 median(강건)
                    boxes[i, 7] = float(scale) * hx
                    boxes[i, 8] = float(scale) * hy
                else:
                    boxes[i, 7] = pvx[m].mean()                      # 기하 불량 → 복사 fallback
                    boxes[i, 8] = pvy[m].mean()
            else:
                boxes[i, 7] = pvx[m].mean()
                boxes[i, 8] = pvy[m].mean()

    if yaw_from_velocity:
        vx, vy = boxes[:, 7], boxes[:, 8]
        speed = (vx ** 2 + vy ** 2).sqrt()
        is_vehicle = torch.zeros_like(speed, dtype=torch.bool)
        for c in _DIAG_VEHICLE_CLS:
            is_vehicle |= labels == c
        m = is_vehicle & (speed > 1.5)
        if bool(m.any()):
            lidar_heading = torch.atan2(vy[m], vx[m])
            boxes[m, 6] = -lidar_heading - np.pi / 2   # lidar→BEV yaw (labels.py 규약, involution)

    out = dict(result)
    out["boxes_3d"] = boxes
    return out


class PredictionCollector:
    """
    Val dataset 전체 순회 → prediction 수집 + 단계별 runtime 측정.

    수집 항목 (per sample):
      {
        "token":      str,
        "scene_name": str,
        "pose":       [4,4] np.ndarray,
        "boxes_3d":   [N, 9] np.ndarray  (lidar frame),
        "scores":     [N] np.ndarray,
        "labels":     [N] np.ndarray,
        "embeds":     [N, D] np.ndarray,
        "track_ids":  [N] np.ndarray,
      }

    Timing 항목:
      {
        "model_ms":   List[float],
        "decoder_ms": List[float],
        "tracker_ms": List[float],
        "total_ms":   List[float],
      }
    """
    def __init__(
        self,
        model,
        decoder,
        tracker,
        device: str,
        nusc: NuScenes,
        viz_enabled: bool = True,
        viz_dir: Optional[str] = None,
        viz_bev_h: int = 400,
        viz_bev_w: int = 400,
        viz_max_trail: int = 10,
        viz_score_threshold: float = 0.1,
    ):
        self.model = model          # LightningModule(ckpt 로드됨); forward는 model(batch)
        self.decoder = decoder      # QueryBBoxDecoder: dense head 출력 → 박스 디코딩
        self.tracker = tracker      # CC3DTPPTracker(기본) 또는 legacy Tracker: ID 부여
        self.device = device
        self.nusc = nusc
        self.viz_enabled = viz_enabled
        self.viz_dir = Path(viz_dir) if viz_dir else None
        self.viz_bev_h = viz_bev_h
        self.viz_bev_w = viz_bev_w
        self.viz_max_trail = viz_max_trail
        self.viz_score_threshold = viz_score_threshold   # viz용 박스 필터 임계값(metric과 별개)

        if self.viz_enabled and self.viz_dir is not None:
            self.viz_dir.mkdir(parents=True, exist_ok=True)

        # [진단 후처리] 상한 측정 플래그 — main()에서 cfg.diag.*로 설정됨(기본 off)
        self.diag_vel_from_radar = False
        self.diag_yaw_from_velocity = False

    @staticmethod
    def _boxes_to_global(boxes_3d: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Transform lidar-flat boxes -> global frame for tracker association.

        Tracking association + KF run across frames, so they must use a
        temporally-consistent (global) frame. Detection/track OUTPUT boxes stay
        in lidar-flat (converted per-frame to global downstream); only the copy
        fed to the tracker is moved to global here.

        boxes_3d: [N, 9/10] (cx, cy, cz, w, l, h, yaw|sin,cos, vx, vy) lidar-flat.
        """
        pose = np.asarray(pose)
        if pose.ndim == 3:          # [1,4,4]로 들어오면 첫 배치 추출
            pose = pose[0]
        out = boxes_3d.copy().astype(np.float64)
        R, t = pose[:3, :3], pose[:3, 3]    # pose=lidar→global 변환(회전 R[3,3], 평행이동 t[3])
        # position: lidar 좌표 → global 좌표 (xyz @ R.T + t)
        xyz = np.stack([boxes_3d[:, 0], boxes_3d[:, 1], boxes_3d[:, 2]], axis=1)  # [N,3]
        g = xyz @ R.T + t           # [N,3] global 위치
        out[:, 0], out[:, 1], out[:, 2] = g[:, 0], g[:, 1], g[:, 2]
        # yaw: add pose yaw (only meaningful for 9-d yaw boxes; 10-d sin/cos used
        # by tracker only for output smoothing, harmless to leave local)
        if boxes_3d.shape[1] == 9:          # [N,9] 박스: yaw는 스칼라(index 6), vel은 7,8
            pose_yaw = np.arctan2(R[1, 0], R[0, 0])   # pose의 z축 회전각
            out[:, 6] = boxes_3d[:, 6] + pose_yaw     # local yaw + pose yaw = global yaw
            vidx = (7, 8)
        else:                               # [N,10] 박스: sin/cos yaw, vel은 8,9
            vidx = (8, 9)
        # velocity (rotation only) — 속도는 방향만 회전(평행이동 없음)
        v = np.stack([boxes_3d[:, vidx[0]], boxes_3d[:, vidx[1]], np.zeros(len(boxes_3d))], axis=1)
        vg = v @ R.T                # global frame 속도
        out[:, vidx[0]], out[:, vidx[1]] = vg[:, 0], vg[:, 1]
        return out

    def _to_device(self, batch):
        # batch의 tensor(및 tensor 리스트)만 GPU로 이동, 나머지(token/scene 등)는 그대로
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                out[k] = [item.to(self.device) for item in v]
            else:
                out[k] = v
        return out

    def _extract_scene_name(self, batch, b: int) -> str:
        # 배치 b번째 sample의 token → scene name (scene 경계마다 tracker reset)
        tokens = batch["token"]
        token = tokens[b] if isinstance(tokens, (list, tuple)) else tokens
        return get_scene_name(self.nusc, token)

    def _save_viz_frame(
        self,
        scene_name: str,
        frame_idx: int,
        batch: Dict[str, Any],
        b: int,
        boxes_3d: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        track_ids: np.ndarray,
        history: Dict[int, List],
        gt_boxes: np.ndarray,
    ):
        """통합 시각화: 6 cameras + Radar/GT/Det/Track BEV. scene별 디렉토리에 frame PNG 저장."""
        scene_dir = self.viz_dir / scene_name      # scene_name/0000.png 형태로 저장
        scene_dir.mkdir(parents=True, exist_ok=True)

        # batch에서 시각화 입력 추출 (b=배치 인덱스)
        images = batch["image"][b]                              # [6, 3, H, W]
        lidar2imgs = batch["lidar2img"][b]                       # [6, 4, 4]
        if torch.is_tensor(lidar2imgs):
            lidar2imgs = lidar2imgs.cpu().numpy()

        radar_points = batch["radar_points"][b]
        if torch.is_tensor(radar_points):
            radar_points = radar_points.cpu().numpy()
        # Visualize the keyframe (sample) sweep only — drop accumulated past sweeps.
        # Encoded radar feature layout puts the time-delta at column 52 (0 = keyframe).
        if radar_points.ndim == 2 and radar_points.shape[1] > 52:
            radar_points = radar_points[np.abs(radar_points[:, 52]) < 0.05]  # col 52=time-delta, 0=keyframe만

        try:
            img = render_composite_frame(            # 6캠 + 4 BEV 통합 이미지 생성
                images=images,
                lidar2imgs=lidar2imgs,
                radar_points=radar_points,
                gt_boxes=gt_boxes,
                pred_boxes_3d=boxes_3d,
                pred_scores=scores,
                pred_labels=labels,
                track_ids=track_ids,
                track_history=history,
                bev_h=self.viz_bev_h, bev_w=self.viz_bev_w,
                bev_range=(-50, 50, -50, 50),
                max_trail_length=self.viz_max_trail,
            )
            cv2.imwrite(str(scene_dir / f"{frame_idx:04d}.png"), img)
        except Exception as e:
            import traceback
            print(f"[viz error @ {scene_name}/{frame_idx:04d}] {e}")
            traceback.print_exc()
    def _load_gt_boxes(self, batch, b):
        """batch의 gt_box 경로(.npz)를 읽어 12-field array로 로드 (viz용 GT 박스)."""
        gt_box_path = batch.get("gt_box", [None])[b] if isinstance(batch.get("gt_box"), (list, tuple)) else None
        scene = batch["scene"][b] if isinstance(batch["scene"], (list, tuple)) else batch["scene"]

        if gt_box_path is None:
            return np.zeros((0, 12), dtype=np.float32)   # 경로 없으면 빈 [0,12]

        # labels_dir 추론 (data config에서)
        # 안전하게 환경 경로 탐색 (컨테이너별 마운트 경로 후보)
        from pathlib import Path
        candidates = [
            Path("/root/data/nuscenes/nuscenes/original/full/labels"),
            Path("/data/nuscenes/labels"),
        ]

        for labels_root in candidates:
            full_path = labels_root / scene / gt_box_path
            if full_path.exists():
                try:
                    return np.load(full_path, allow_pickle=True)["gt_box"]  # [M,12]
                except Exception:
                    pass
        return np.zeros((0, 12), dtype=np.float32)       # 어느 후보도 없으면 빈 array


    def run(self, val_loader) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        Val 전체 순회.

        Returns:
            predictions: List[Dict] - per-sample 예측
            timings: Dict[str, List[float]] - 단계별 ms
        """
        predictions = []
        current_scene = None        # 직전 처리 scene(바뀌면 tracker reset)
        scene_frame_idx = 0         # scene 내 프레임 번호(viz 파일명)

        # Timing 누적 — 단계별 ms 리스트
        timings = {
            "model_ms":   [],
            "decoder_ms": [],
            "tracker_ms": [],
            "total_ms":   [],
        }

        self.model.eval()           # BN/dropout eval 모드

        # GPU sync helper
        use_cuda = (
            isinstance(self.device, str) and self.device.startswith("cuda")
        )

        def _now():
            # CUDA면 비동기 커널 끝까지 동기화 후 시각 측정(정확한 runtime)
            if use_cuda:
                torch.cuda.synchronize()
            return time.perf_counter()

        pbar = tqdm(val_loader, desc="Collecting predictions", colour="cyan")
        for batch_idx, batch in enumerate(pbar):    # val 전체 순회(eval B=2)
            batch = self._to_device(batch)          # tensor를 GPU로

            # ---------- Model forward (timed) ----------
            t_model_start = _now()
            with torch.no_grad():
                outputs = self.model(batch)         # GS 렌더 + dense head; outputs["output"]에 head 출력
            t_model_end = _now()

            # ---------- Decoder (timed) ----------
            t_decoder_start = _now()
            pred_results = self.decoder(outputs["output"])   # dense → 박스 디코딩, List[B] of dict
            t_decoder_end = _now()

            # [진단 후처리] velocity/yaw 상한 측정 (cfg.diag.* 플래그, 기본 off)
            if getattr(self, "diag_vel_from_radar", False) or getattr(self, "diag_yaw_from_velocity", False):
                radar_pts = batch.get("radar_points")
                for b in range(len(pred_results)):
                    pts_b = radar_pts[b] if radar_pts is not None else None
                    pred_results[b] = _diag_postprocess_boxes(
                        pred_results[b], pts_b,
                        vel_from_radar=getattr(self, "diag_vel_from_radar", False),
                        yaw_from_velocity=getattr(self, "diag_yaw_from_velocity", False),
                    )

            # per-batch 처리 (배치 안의 각 sample을 개별 처리)
            B = len(pred_results)               # 보통 2
            for b in range(B):
                # Scene 식별 + tracker reset (scene이 바뀌면 추적 상태 초기화)
                scene_name = self._extract_scene_name(batch, b)

                if scene_name != current_scene:
                    self.tracker.reset()        # 새 scene → KF/track 상태 비움
                    current_scene = scene_name
                    scene_frame_idx = 0

                # Numpy 변환 (pred_results[b]: {boxes_3d[N,9], scores[N], labels[N], embeds[N,D]})
                boxes_3d = pred_results[b]["boxes_3d"].cpu().numpy()   # [N,9] lidar-flat
                scores = pred_results[b]["scores"].cpu().numpy()       # [N]
                labels = pred_results[b]["labels"].cpu().numpy()       # [N]
                embeds = pred_results[b].get("embeds", None)           # [N,D] appearance feature(tracker용)
                if embeds is not None:
                    embeds = embeds.cpu().numpy()

                # Pose (lidar-flat -> global). Needed BEFORE the tracker: association
                # and the KF motion model must run in a temporally-consistent (global)
                # frame, otherwise ego motion (~2.5m/frame) corrupts the distance cost.
                poses = batch["pose"]               # lidar→global 변환행렬 [4,4]
                if torch.is_tensor(poses):
                    pose = (
                        poses[b].cpu().numpy() if poses.ndim == 3
                        else poses.cpu().numpy()
                    )
                else:
                    pose = (
                        np.array(poses[b]) if isinstance(poses, (list, tuple))
                        else np.array(poses)
                    )

                # ---------- Tracker (timed) ----------
                t_tracker_start = _now()
                if embeds is not None and len(boxes_3d) > 0:
                    # tracker 입력만 global frame으로 변환(ego motion ~2.5m/frame 보정)
                    boxes_track = self._boxes_to_global(boxes_3d, pose)
                    # New CC3DTPPTracker accepts `labels`; older Tracker signature uses positional args only.
                    try:
                        tids = self.tracker.update(boxes_track, scores, embeds, labels=labels)  # [N] track ID
                    except TypeError:
                        tids = self.tracker.update(boxes_track, scores, embeds)   # legacy Tracker(라벨 인자 없음)
                else:
                    tids = np.full(len(boxes_3d), -1, dtype=np.int64)   # 검출 0 or embed 없음 → 전부 -1
                t_tracker_end = _now()

                # Timing 누적 (tracker는 sample마다, model/decoder는 배치당 1회→첫 sample에서만)
                tracker_ms = (t_tracker_end - t_tracker_start) * 1000
                timings["tracker_ms"].append(tracker_ms)

                if b == 0:      # model/decoder는 배치 전체에 1번 forward → 중복 카운트 방지
                    model_ms = (t_model_end - t_model_start) * 1000
                    decoder_ms = (t_decoder_end - t_decoder_start) * 1000
                    timings["model_ms"].append(model_ms)
                    timings["decoder_ms"].append(decoder_ms)
                    timings["total_ms"].append(model_ms + decoder_ms + tracker_ms)

                # Token
                tokens = batch["token"]
                token = tokens[b] if isinstance(tokens, (list, tuple)) else tokens

                # Viz 저장 (통합) - viz용 score threshold + top-K fallback
                if self.viz_enabled and self.viz_dir is not None:
                    gt_boxes = self._load_gt_boxes(batch, b)   # [M,12] GT 박스

                    # 1차: score threshold 적용 (저신뢰 박스 제거)
                    viz_mask = scores >= self.viz_score_threshold

                    # 2차 fallback: threshold 통과한 박스가 너무 적으면 Top-K 사용(빈 그림 방지)
                    if viz_mask.sum() < 5 and len(scores) > 0:
                        # 상위 min(20, len) 개 박스 사용
                        K = min(20, len(scores))
                        topk_idx = np.argsort(scores)[-K:]     # score 상위 K
                        viz_mask = np.zeros_like(scores, dtype=bool)
                        viz_mask[topk_idx] = True

                    viz_boxes = boxes_3d[viz_mask]     # 시각화는 lidar-flat 박스 사용(global 아님)
                    viz_scores = scores[viz_mask]
                    viz_labels = labels[viz_mask]
                    viz_tids = tids[viz_mask]

                    self._save_viz_frame(
                        scene_name=scene_name,
                        frame_idx=scene_frame_idx,
                        batch=batch,
                        b=b,
                        boxes_3d=viz_boxes,
                        scores=viz_scores,
                        labels=viz_labels,
                        track_ids=viz_tids,
                        # Only draw the CURRENT frame's tracked boxes — no accumulated
                        # trails. The tracker associates in the global frame, so its
                        # history is in global coords and would scatter across the
                        # local BEV. Pass empty history to suppress trails.
                        history={},
                        gt_boxes=gt_boxes,
                    )

                # 수집 — per-sample 예측(metric/ablation에서 재사용; 모델 재forward 없음)
                predictions.append({
                    "token":      token,        # sample token(metric의 nuScenes 매칭 키)
                    "scene_name": scene_name,   # weather ablation/scene 경계용
                    "pose":       pose,         # [4,4] lidar→global(metric에서 global 변환)
                    "boxes_3d":   boxes_3d,     # [N,9] lidar-flat
                    "scores":     scores,       # [N]
                    "labels":     labels,       # [N]
                    "embeds":     embeds if embeds is not None else np.zeros((len(boxes_3d), 0)),  # [N,D]
                    "track_ids":  tids,         # [N] ID(-1=미매칭)
                })

                scene_frame_idx += 1

            pbar.set_postfix_str(f"scene={current_scene}, frame={scene_frame_idx}")

        return predictions, timings   # 전체 sample 예측 + 단계별 timing


# ###########################################################################
# ablations.py  (4종 Ablation 구현)
#   - 역할: 이미 수집된 predictions를 재사용(모델 재forward 없이)해 조건별로 필터링/
#           tracker 재실행 후 metric을 다시 계산한다. score/distance/weather/match_thr.
# ###########################################################################
# ===========================================================================
# 공통 헬퍼
# ===========================================================================
def _build_pred_results_for_sample(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    """PredictionCollector 출력(numpy) → Metric.update() 입력 형식(List[1] of tensor dict)."""
    return [{
        "boxes_3d":  torch.from_numpy(pred["boxes_3d"]),   # [N,9]
        "scores":    torch.from_numpy(pred["scores"]),     # [N]
        "labels":    torch.from_numpy(pred["labels"]),     # [N]
        "embeds":    torch.from_numpy(pred["embeds"]),     # [N,D]
        "track_ids": pred["track_ids"],                    # [N]
    }]


def _build_batch_for_sample(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Metric.update가 batch에서 참조하는 필드(token/pose)만 최소 구성."""
    return {
        "token": [pred["token"]],                  # batch 형식(리스트)으로 감쌈
        "pose":  np.expand_dims(pred["pose"], 0),  # [4,4]→[1,4,4]
    }


def _filter_by_score(pred: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Ablation 1: score threshold로 박스 필터링(낮은 score 제거)."""
    mask = pred["scores"] >= threshold
    out = {}
    for k, v in pred.items():
        # N차원(박스 수) 배열만 마스킹; token/pose/scene_name 등은 그대로
        if isinstance(v, np.ndarray) and v.shape[0] == len(pred["scores"]):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def _filter_by_distance(pred: Dict[str, Any], r_min: float, r_max: float) -> Dict[str, Any]:
    """Ablation 2: BEV center 거리(ego로부터 m)로 박스 필터링."""
    if len(pred["boxes_3d"]) == 0:
        return pred
    xy = pred["boxes_3d"][:, :2]            # 박스 중심 x,y
    dist = np.linalg.norm(xy, axis=1)      # ego(원점)로부터 거리
    mask = (dist >= r_min) & (dist < r_max)  # [r_min, r_max) 범위만 유지
    out = {}
    for k, v in pred.items():
        if isinstance(v, np.ndarray) and v.shape[0] == len(pred["scores"]):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def _empty_pred_dict(template: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation 3: 동일 token/pose 유지하면서 detection 비우기 (해당 weather가 아닌 scene용)."""
    embed_dim = template["embeds"].shape[1] if template["embeds"].ndim == 2 else 0
    return {
        **template,                                       # token/pose/scene_name 유지
        "boxes_3d":  np.zeros((0, 9), dtype=np.float32),  # 검출 0개 → GT만 있는 sample(전부 FN)
        "scores":    np.zeros(0, dtype=np.float32),
        "labels":    np.zeros(0, dtype=np.int64),
        "embeds":    np.zeros((0, embed_dim), dtype=np.float32),
        "track_ids": np.zeros(0, dtype=np.int64),
    }


def _compute_det_metric(
    predictions: List[Dict[str, Any]],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
    score_threshold: float = 0.1,
) -> Dict[str, float]:
    """nuScenes 공식 Detection metric(mAP/NDS/TP errors) 계산."""
    from gaussianmot.metrics import DetectionNuScenesMetric

    metric = DetectionNuScenesMetric(
        nusc_dataroot=nusc_dataroot,
        version=version,
        eval_set=eval_set,
        output_dir=output_dir,
        score_threshold=score_threshold,
    )

    for pred in predictions:            # 모든 sample을 metric에 누적
        if len(pred["boxes_3d"]) == 0:  # 빈 예측도 token/pose는 등록해야 nuScenes 집계가 맞음
            empty_pred = [{
                "boxes_3d": torch.zeros(0, 9),
                "scores":   torch.zeros(0),
                "labels":   torch.zeros(0, dtype=torch.long),
                "embeds":   torch.zeros(0, 0),
            }]
            metric.update(empty_pred, _build_batch_for_sample(pred))
            continue

        pred_results = _build_pred_results_for_sample(pred)   # numpy→tensor dict
        batch = _build_batch_for_sample(pred)                 # token/pose
        metric.update(pred_results, batch)

    try:
        return metric.compute()         # 전체 누적 후 mAP/NDS 등 산출
    except Exception as e:
        import traceback
        log.warning(f"Detection metric compute failed: {e}")
        log.warning(traceback.format_exc())
        return {
            "mAP": 0.0, "NDS": 0.0,
            "mATE": 1.0, "mASE": 1.0, "mAOE": 1.0,
            "mAVE": 1.0, "mAAE": 1.0,
        }


def _compute_track_metric(
    predictions: List[Dict[str, Any]],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
    score_threshold: float = 0.1,
) -> Dict[str, float]:
    """nuScenes 공식 Tracking metric(AMOTA/AMOTP/MOTA/IDS/FRAG) 계산. track_ids 필요."""
    from gaussianmot.metrics import TrackingNuScenesMetric

    metric = TrackingNuScenesMetric(
        nusc_dataroot=nusc_dataroot,
        version=version,
        eval_set=eval_set,
        output_dir=output_dir,
        score_threshold=score_threshold,
    )

    for pred in predictions:
        if len(pred["boxes_3d"]) == 0:
            empty_pred = [{
                "boxes_3d":  torch.zeros(0, 9),
                "scores":    torch.zeros(0),
                "labels":    torch.zeros(0, dtype=torch.long),
                "embeds":    torch.zeros(0, 0),
                "track_ids": np.zeros(0, dtype=np.int64),
            }]
            metric.update(empty_pred, _build_batch_for_sample(pred))
            continue

        pred_results = _build_pred_results_for_sample(pred)
        batch = _build_batch_for_sample(pred)
        metric.update(pred_results, batch)

    try:
        return metric.compute()
    except Exception as e:
        log.warning(f"Tracking metric compute failed: {e}")
        return {
            "AMOTA": 0.0, "AMOTP": 1.0,
            "MOTA": 0.0, "MOTP": 1.0,
            "IDS": 0, "FRAG": 0, "FP": 0, "FN": 0,
        }


# ===========================================================================
# 1. Score Threshold
# ===========================================================================
def ablation_score_threshold(
    predictions: List[Dict[str, Any]],
    thresholds: List[float],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
) -> Dict[float, Dict[str, Any]]:
    """Ablation 1: score threshold를 바꿔가며 Det + Track 성능 변화 측정."""
    results = {}
    for thr in tqdm(thresholds, desc="Ablation: score_threshold", colour="yellow"):
        filtered = [_filter_by_score(p, thr) for p in predictions]   # thr 이상 박스만

        det = _compute_det_metric(
            filtered, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/score_{thr}",
            score_threshold=thr,
        )
        track = _compute_track_metric(
            filtered, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/score_{thr}",
            score_threshold=thr,
        )
        results[thr] = {"det": det, "track": track}

    return results


# ===========================================================================
# 2. Distance Range (Det만)
# ===========================================================================
def ablation_distance_range(
    predictions: List[Dict[str, Any]],
    ranges: List[Tuple[float, float]],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Ablation 2: ego로부터의 거리 범위별 Detection 성능(원거리 검출 약점 분석)."""
    results = {}
    for r_min, r_max in tqdm(ranges, desc="Ablation: distance_range", colour="yellow"):
        key = f"{int(r_min)}-{int(r_max)}m"        # 예: "0-20m"
        filtered = [_filter_by_distance(p, r_min, r_max) for p in predictions]

        det = _compute_det_metric(
            filtered, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/range_{key}",
        )
        results[key] = {"det": det}

    return results


# ===========================================================================
# 3. Weather (Det + Track)
# ===========================================================================
def ablation_weather(
    predictions: List[Dict[str, Any]],
    splits: Dict[str, List[str]],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Ablation 3: Day/Night/Rain 별 Det + Track 성능(악천후 강건성 분석)."""
    results = {}

    for weather, scene_list in tqdm(splits.items(), desc="Ablation: weather", colour="yellow"):
        scene_set = set(scene_list)        # 해당 weather에 속하는 scene name 집합

        # 해당 weather scene이면 예측 유지, 아니면 빈 예측으로(=평가에서 제외 효과)
        filtered = [
            p if p["scene_name"] in scene_set else _empty_pred_dict(p)
            for p in predictions
        ]

        det = _compute_det_metric(
            filtered, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/weather_{weather}",
        )
        track = _compute_track_metric(
            filtered, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/weather_{weather}",
        )
        results[weather] = {"det": det, "track": track}

    return results


# ===========================================================================
# 4. Track Match Threshold (Tracker 재실행)
# ===========================================================================
def ablation_track_match(
    predictions: List[Dict[str, Any]],
    match_thresholds: List[float],
    tracker_cfg: Dict[str, Any],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
) -> Dict[float, Dict[str, Any]]:
    """Ablation 4: match threshold를 바꿔 tracker만 재실행 → Tracking 성능(association 민감도)."""
    from gaussianmot.modeling.components.cc3dt_tracker import CC3DTPPTracker   # 기본 tracker로 재실행

    results = {}
    for match_thr in tqdm(match_thresholds, desc="Ablation: track_match", colour="yellow"):
        new_cfg = dict(tracker_cfg)
        new_cfg["match_score_thr"] = match_thr   # CC3DTPP의 매칭 affinity 임계값만 교체
        tracker = CC3DTPPTracker(**new_cfg)

        # Tracker 재실행 (모델/디코더는 그대로, 박스에 ID만 다시 부여)
        current_scene = None
        new_predictions = []
        for pred in predictions:
            if pred["scene_name"] != current_scene:    # scene 경계 reset
                tracker.reset()
                current_scene = pred["scene_name"]

            if len(pred["boxes_3d"]) > 0 and pred["embeds"].shape[1] > 0:
                # CC3DTPP는 글로벌 프레임에서 association → 박스를 글로벌로 변환 후 업데이트.
                boxes_g = PredictionCollector._boxes_to_global(pred["boxes_3d"], pred["pose"])
                try:
                    tids = tracker.update(boxes_g, pred["scores"], pred["embeds"], labels=pred["labels"])
                except TypeError:
                    tids = tracker.update(boxes_g, pred["scores"], pred["embeds"])
            else:
                tids = np.full(len(pred["boxes_3d"]), -1, dtype=np.int64)

            new_predictions.append({**pred, "track_ids": tids})   # ID만 갱신한 사본

        track = _compute_track_metric(
            new_predictions, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/match_{match_thr}",
        )
        results[match_thr] = {"track": track}

    return results


# ===========================================================================
# 5. Affinity Weights (Tracker 재실행) — appearance/location/velocity 기여 분리
# ===========================================================================
def ablation_affinity(
    predictions: List[Dict[str, Any]],
    weight_configs: List[List[float]],
    tracker_cfg: Dict[str, Any],
    nusc_dataroot: str,
    version: str,
    eval_set: str,
    output_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Ablation 5: affinity 가중치 (w_app, w_loc, w_vel)를 바꿔 tracker만 재실행.

    appearance 임베딩이 association(IDS/AMOTA)에 도움인지 해인지 분리 진단용.
    w_app=0 → 순수 motion/location tracking. 재forward 없이 캐시 예측 재사용.
    """
    from gaussianmot.modeling.components.cc3dt_tracker import CC3DTPPTracker

    results = {}
    for wcfg in tqdm(weight_configs, desc="Ablation: affinity", colour="yellow"):
        w_app, w_loc, w_vel = [float(x) for x in wcfg]
        new_cfg = dict(tracker_cfg)
        new_cfg["w_appearance"] = w_app
        new_cfg["w_location"] = w_loc
        new_cfg["w_velocity"] = w_vel
        tracker = CC3DTPPTracker(**new_cfg)

        current_scene = None
        new_predictions = []
        for pred in predictions:
            if pred["scene_name"] != current_scene:
                tracker.reset()
                current_scene = pred["scene_name"]

            if len(pred["boxes_3d"]) > 0 and pred["embeds"].shape[1] > 0:
                boxes_g = PredictionCollector._boxes_to_global(pred["boxes_3d"], pred["pose"])
                try:
                    tids = tracker.update(boxes_g, pred["scores"], pred["embeds"], labels=pred["labels"])
                except TypeError:
                    tids = tracker.update(boxes_g, pred["scores"], pred["embeds"])
            else:
                tids = np.full(len(pred["boxes_3d"]), -1, dtype=np.int64)

            new_predictions.append({**pred, "track_ids": tids})

        key = f"app{w_app:.1f}_loc{w_loc:.1f}_vel{w_vel:.1f}"
        track = _compute_track_metric(
            new_predictions, nusc_dataroot, version, eval_set,
            output_dir=f"{output_dir}/affinity_{key}",
        )
        results[key] = {"track": track}

    return results


# ###########################################################################
# report.py  (Rich Table 출력 + JSON 결과 저장)
#   - 역할: 계산된 metric/timing/ablation 결과를 Rich 표로 콘솔 출력하고 JSON으로 저장.
# ###########################################################################
def print_main_results(det: Dict[str, float], track: Dict[str, float], console: Console):
    """Main Results - Detection + Tracking 종합 표(좌=검출, 우=추적)."""
    t = Table(title="Main Results", show_lines=True)
    t.add_column("Metric", style="cyan")
    t.add_column("Detection", justify="right", style="magenta")
    t.add_column("Tracking", justify="right", style="green")

    t.add_row("mAP / AMOTA", f"{det['mAP']:.4f}", f"{track['AMOTA']:.4f}")
    t.add_row("NDS / AMOTP", f"{det['NDS']:.4f}", f"{track['AMOTP']:.4f}")
    t.add_row("mATE / MOTA", f"{det['mATE']:.4f}", f"{track['MOTA']:.4f}")
    t.add_row("mAVE / MOTP", f"{det['mAVE']:.4f}", f"{track['MOTP']:.4f}")
    t.add_row("mAOE / IDS",  f"{det['mAOE']:.4f}", f"{int(track['IDS'])}")
    t.add_row("mASE / FRAG", f"{det['mASE']:.4f}", f"{int(track['FRAG'])}")

    console.print(t)


def print_runtime_analysis(timings: Dict[str, list], console: Console):
    """Runtime 통계 출력 (논문 Table IV): 단계별 mean/std/min/max ms + FPS."""
    t = Table(title="Runtime Analysis", show_lines=True)
    t.add_column("Stage", style="cyan")
    t.add_column("Mean (ms)", justify="right", style="magenta")
    t.add_column("Std (ms)",  justify="right", style="magenta")
    t.add_column("Min (ms)",  justify="right", style="magenta")
    t.add_column("Max (ms)",  justify="right", style="magenta")
    t.add_column("FPS",       justify="right", style="green")

    for stage, label in [
        ("model_ms",   "Model"),
        ("decoder_ms", "Decoder"),
        ("tracker_ms", "Tracker"),
        ("total_ms",   "Total"),
    ]:
        vals = timings.get(stage, [])
        if not vals:
            continue
        arr = np.array(vals)
        mean, std = arr.mean(), arr.std()
        mn, mx = arr.min(), arr.max()
        fps = 1000.0 / mean if mean > 0 else 0.0   # ms→FPS 환산

        t.add_row(
            label,
            f"{mean:.2f}", f"{std:.2f}",
            f"{mn:.2f}", f"{mx:.2f}",
            f"{fps:.2f}",
        )

    console.print(t)


def print_score_threshold_ablation(results: Dict[float, Dict], console: Console):
    t = Table(title="Ablation 1: Score Threshold", show_lines=True)
    t.add_column("Threshold", style="cyan")
    t.add_column("mAP",   justify="right", style="magenta")
    t.add_column("NDS",   justify="right", style="magenta")
    t.add_column("mAVE",  justify="right", style="magenta")
    t.add_column("AMOTA", justify="right", style="green")
    t.add_column("IDS",   justify="right", style="green")

    for thr, res in sorted(results.items()):
        t.add_row(
            f"{thr:.2f}",
            f"{res['det']['mAP']:.4f}",
            f"{res['det']['NDS']:.4f}",
            f"{res['det']['mAVE']:.4f}",
            f"{res['track']['AMOTA']:.4f}",
            f"{int(res['track']['IDS'])}",
        )
    console.print(t)


def print_distance_range_ablation(results: Dict[str, Dict], console: Console):
    t = Table(title="Ablation 2: Distance Range (Detection)", show_lines=True)
    t.add_column("Range", style="cyan")
    t.add_column("mAP",  justify="right", style="magenta")
    t.add_column("NDS",  justify="right", style="magenta")
    t.add_column("mATE", justify="right", style="magenta")
    t.add_column("mAVE", justify="right", style="magenta")

    for key, res in results.items():
        t.add_row(
            key,
            f"{res['det']['mAP']:.4f}",
            f"{res['det']['NDS']:.4f}",
            f"{res['det']['mATE']:.4f}",
            f"{res['det']['mAVE']:.4f}",
        )
    console.print(t)


def print_weather_ablation(results: Dict[str, Dict], console: Console):
    t = Table(title="Ablation 3: Weather Condition", show_lines=True)
    t.add_column("Weather", style="cyan")
    t.add_column("mAP",   justify="right", style="magenta")
    t.add_column("NDS",   justify="right", style="magenta")
    t.add_column("AMOTA", justify="right", style="green")
    t.add_column("IDS",   justify="right", style="green")

    for weather, res in results.items():
        t.add_row(
            weather,
            f"{res['det']['mAP']:.4f}",
            f"{res['det']['NDS']:.4f}",
            f"{res['track']['AMOTA']:.4f}",
            f"{int(res['track']['IDS'])}",
        )
    console.print(t)


def print_track_match_ablation(results: Dict[float, Dict], console: Console):
    t = Table(title="Ablation 4: Track Match Threshold", show_lines=True)
    t.add_column("Match Thr", style="cyan")
    t.add_column("AMOTA", justify="right", style="green")
    t.add_column("AMOTP", justify="right", style="green")
    t.add_column("IDS",   justify="right", style="green")
    t.add_column("FRAG",  justify="right", style="green")

    for thr, res in sorted(results.items()):
        t.add_row(
            f"{thr:.2f}",
            f"{res['track']['AMOTA']:.4f}",
            f"{res['track']['AMOTP']:.4f}",
            f"{int(res['track']['IDS'])}",
            f"{int(res['track']['FRAG'])}",
        )
    console.print(t)


def save_all_results(
    det_main: Dict,
    track_main: Dict,
    ablations: Dict[str, Any],
    output_dir: str,
    timings: Dict = None,
):
    """전체 결과(main det/track + ablations + runtime)를 metrics_summary.json으로 저장."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_py(obj):
        # numpy 스칼라/배열을 JSON 직렬화 가능한 파이썬 기본형으로 재귀 변환
        if isinstance(obj, dict):
            return {str(k): to_py(v) for k, v in obj.items()}   # dict key도 str로(float key 대응)
        elif isinstance(obj, list):
            return [to_py(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {
        "main": {
            "detection": to_py(det_main),
            "tracking":  to_py(track_main),
        },
        "ablations": to_py(ablations),
    }

    if timings is not None:
        runtime_summary = {}
        for stage, vals in timings.items():
            if not vals:
                continue
            arr = np.array(vals)
            mean = float(arr.mean())
            runtime_summary[stage] = {
                "mean_ms": mean,
                "std_ms":  float(arr.std()),
                "min_ms":  float(arr.min()),
                "max_ms":  float(arr.max()),
                "fps":     float(1000.0 / mean) if mean > 0 else 0.0,
                "num_samples": len(vals),
            }
        summary["runtime"] = runtime_summary

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ###########################################################################
# Main entrypoint  (original tools/evaluate.py)
#   - 역할: config 로드 → datamodule/module/decoder/tracker 구성 → collector.run →
#           main metric → 4종 ablation → 표/JSON 저장 → (옵션) fig6. 전체 오케스트레이션.
# ###########################################################################
def load_from_checkpoint(
    module: L.LightningModule,
    checkpoint_path: str,
    device: str,
) -> L.LightningModule:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["state_dict"]
    # 평가 모듈은 EMA teacher 없음 → ckpt의 teacher.* 키 제거
    state = {k: v for k, v in state.items() if not k.startswith("teacher.")}
    # strict=False: warm-start류 ckpt(새 모듈 키가 없는 baseline 등)도 평가 가능.
    # 누락/잉여 키를 카운트해 의도치 않은 큰 불일치를 드러낸다.
    incompat = module.load_state_dict(state, strict=False)   # 키 불일치 허용(warm-start류)
    n_miss, n_unexp = len(incompat.missing_keys), len(incompat.unexpected_keys)
    if n_miss or n_unexp:    # 의도치 않은 큰 불일치를 로그로 드러냄(조용한 미로드 방지)
        print(f"[load_from_checkpoint] strict=False: missing={n_miss} unexpected={n_unexp}")
        if n_miss:
            print(f"  e.g. missing: {incompat.missing_keys[:5]}")
        if n_unexp:
            print(f"  e.g. unexpected: {incompat.unexpected_keys[:5]}")
    module.to(device)
    return module


# ---------------------------------------------------------------------------
# Fig.6-style qualitative figure (merged from tools/viz_fig6.py)
#
# Per sample, produces a horizontal strip:
#   [ multi-view camera canvas | PCA(camera Gaussian BEV) | PCA(radar Gaussian BEV) | detection error map ]
#
# Error map color code (GaussianMOT Fig.6 analog: correct / missing / incorrect):
#   - GREEN  : correct  (TP — predicted box matched to a GT)
#   - RED    : incorrect(FP — predicted box with no GT match)
#   - YELLOW : missing  (FN — GT box not detected)
#   - 역할: GaussianMOT Fig.6 재현용 정성 그림(카메라 + camera/radar Gaussian feature PCA
#           + 검출 오류맵)을 n_samples개 생성. main eval과 독립적인 옵션 기능.
# ---------------------------------------------------------------------------
MATCH_DIST = 2.0      # TP 판정 중심거리(m): pred↔GT center가 이 이하면 매칭
SCORE_THR = 0.2       # 오류맵에 그릴 pred score 임계값
BEV_PX = 600          # fig6 BEV 패널 픽셀 크기
BEV_RANGE = (-50.0, 50.0, -50.0, 50.0)  # x_min,x_max,y_min,y_max


def denormalize_imagenet(x):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    return (x * std) + mean


def make_canvas(x, denorm=True):
    """GaussianMOT helper: stitch multi-view images (6 -> 2x3, rear flipped)."""
    if x.ndim == 5:          # [B,6,3,H,W]면 첫 배치
        x = x[0]
    num_imgs, c, h, w = x.shape    # [6,3,H,W]
    if num_imgs == 6:
        rows, cols = 2, 3    # 6캠 → 2x3
    elif num_imgs == 4:
        rows, cols = 1, 4
    else:
        raise ValueError(f"Unsupported number of cameras: {num_imgs}")
    if denorm:
        x = denormalize_imagenet(x)
    x = x.clamp(0, 1)
    imgs = [(x[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8) for i in range(num_imgs)]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, cols)
        if num_imgs == 6 and idx in [3, 4, 5]:
            img = np.fliplr(img)
        canvas[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
    return canvas  # RGB


def pca_to_bgr(feat):  # feat [1,C,H,W] -> HxWx3 BGR, upscaled
    pca = extract_pca_features(feat, n_components=3)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # H,W,3 RGB
    pca = cv2.cvtColor(pca, cv2.COLOR_RGB2BGR)
    return cv2.resize(pca, (BEV_PX, BEV_PX), interpolation=cv2.INTER_NEAREST)


def _bev_canvas():
    # fig6용 빈 BEV(중앙 십자선 격자)
    c = np.full((BEV_PX, BEV_PX, 3), 20, dtype=np.uint8)
    cv2.line(c, (BEV_PX // 2, 0), (BEV_PX // 2, BEV_PX), (60, 60, 60), 1)
    cv2.line(c, (0, BEV_PX // 2), (BEV_PX, BEV_PX // 2), (60, 60, 60), 1)
    return c


def _draw_box(canvas, cx, cy, w, l, yaw, color, thick=2):
    # fig6 오류맵용 박스 1개 그리기(metric 좌표→픽셀)
    xmin, xmax, ymin, ymax = BEV_RANGE
    corners = get_box_corners_2d(cx, cy, w, l, yaw)
    pix = np.array([world_to_bev_pixel(c[0], c[1], xmin, xmax, ymin, ymax, BEV_PX, BEV_PX) for c in corners], np.int32)
    cv2.polylines(canvas, [pix], True, color, thick, cv2.LINE_AA)


def error_map(pred_boxes, pred_scores, gt_boxes):
    """Color: GREEN=TP(correct), RED=FP(incorrect), YELLOW=FN(missing). BGR."""
    canvas = _bev_canvas()
    keep = pred_scores >= SCORE_THR     # 임계값 이상 pred만
    pb = pred_boxes[keep]
    g = gt_boxes
    gt_matched = np.zeros(g.shape[0], dtype=bool)   # GT가 매칭됐는지 추적(중복 매칭 방지)
    # match each pred to nearest unmatched GT center (greedy nearest-center 매칭)
    for j in range(pb.shape[0]):
        pcx, pcy, _, pw, pl, _, pyaw = pb[j, 0], pb[j, 1], pb[j, 2], pb[j, 3], pb[j, 4], pb[j, 5], pb[j, 6]
        if g.shape[0] > 0:
            d = np.linalg.norm(g[:, :2] - np.array([pcx, pcy]), axis=1)   # 모든 GT와의 중심거리
            d[gt_matched] = 1e9         # 이미 매칭된 GT는 제외
            k = int(d.argmin())         # 가장 가까운 미매칭 GT
            tp = d[k] <= MATCH_DIST     # 2m 이내면 TP
        else:
            tp = False
        if tp:
            gt_matched[k] = True
            _draw_box(canvas, pcx, pcy, pw, pl, pyaw, (0, 220, 0))      # green TP
        else:
            _draw_box(canvas, pcx, pcy, pw, pl, pyaw, (0, 0, 230))      # red FP
    # unmatched GT -> FN (yellow), GT yaw = atan2(sin,cos)
    for i in range(g.shape[0]):
        if gt_matched[i]:               # 매칭된 GT는 skip
            continue
        gyaw = math.atan2(g[i, 6], g[i, 7])   # 여기 gt_boxes는 sin/cos yaw 포맷(task_gt_boxes)
        _draw_box(canvas, g[i, 0], g[i, 1], g[i, 3], g[i, 4], gyaw, (0, 230, 230), 2)  # yellow FN
    # legend
    for k, (txt, col) in enumerate([("correct(TP)", (0, 220, 0)), ("incorrect(FP)", (0, 0, 230)), ("missing(FN)", (0, 230, 230))]):
        cv2.putText(canvas, txt, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return canvas


def generate_fig6(cfg: DictConfig) -> None:
    register_new_resolvers()
    device = cfg.device
    key = cfg.task.key
    n_samples = int(cfg.get("n_samples", 8))     # 생성할 그림 개수
    out_dir = "/workspace/outputs/viz_fig6"
    os.makedirs(out_dir, exist_ok=True)

    dm = hydra.utils.instantiate(cfg.data); dm.setup("fit")   # datamodule + val loader
    val_loader = dm.val_dataloader()
    module = hydra.utils.instantiate(cfg.module, cfg=cfg)
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(ck.get("state_dict", ck), strict=False)   # 키 불일치 허용
    module.to(device).eval()

    from gaussianmot.modeling.components.query_bbox_decoder import QueryBBoxDecoder
    r = cfg.data.data_config.render
    # 박스 디코더(이름은 query 계열이나 dense head 출력을 박스로 디코딩)
    decoder = QueryBBoxDecoder(key=key, x_min=r.x_min, x_max=r.x_max, y_min=r.y_min, y_max=r.y_max,
                               bev_h=r.bev_h, bev_w=r.bev_w, max_num=cfg.decoder.max_num, score_threshold=SCORE_THR).to(device)

    def to_dev(b):
        o = {}
        for k, v in b.items():
            if torch.is_tensor(v): o[k] = v.to(device)
            elif isinstance(v, list) and len(v) and torch.is_tensor(v[0]): o[k] = [x.to(device) for x in v]
            else: o[k] = v
        return o

    it = iter(val_loader); n = 0
    with torch.no_grad():
        while n < n_samples:
            try: batch = next(it)
            except StopIteration: break
            batch = to_dev(batch)
            out = module.model(batch)              # GS feature(features_cam/radar) + head 출력
            preds = decoder(out["output"])[0]      # 첫 sample 박스 디코딩
            pb = preds["boxes_3d"].cpu().numpy(); ps = preds["scores"].cpu().numpy()  # [N,9],[N]
            gtb = batch.get(f"{key}_gt_boxes")     # task별 GT(sin/cos yaw 포맷)
            gtb = (gtb[0] if isinstance(gtb, (list, tuple)) else (gtb[0] if gtb.ndim == 3 else gtb)).cpu().numpy()

            cam = cv2.cvtColor(make_canvas(batch["image"][0]), cv2.COLOR_RGB2BGR)   # 6캠 2x3 그리드
            cam = cv2.resize(cam, (BEV_PX * 2, BEV_PX), interpolation=cv2.INTER_AREA)
            pca_c = pca_to_bgr(out["features_cam"])      # camera Gaussian BEV의 PCA 시각화
            pca_r = pca_to_bgr(out["features_radar"])    # radar Gaussian BEV의 PCA 시각화
            emap = error_map(pb, ps, gtb)                # TP/FP/FN 오류맵
            # labels
            for img, t in [(pca_c, "PCA camera feat"), (pca_r, "PCA radar feat"), (emap, "detection error map")]:
                cv2.putText(img, t, (8, BEV_PX - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            strip = np.hstack([cam, pca_c, pca_r, emap])   # 4패널 가로 결합 → 한 줄짜리 그림
            cv2.imwrite(os.path.join(out_dir, f"fig6_{n:02d}.png"), strip)
            print(f"saved fig6_{n:02d}.png")
            n += 1
    print(f"done: {n} figures in {out_dir}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate.yaml")
def main(cfg: DictConfig) -> None:
    register_new_resolvers()
    console = Console()

    # =======================================================================
    # 1. Setup — datamodule(val loader) + module(ckpt 로드) + nuScenes SDK
    # =======================================================================
    log.info("Loading datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")                       # val split 준비
    val_loader = datamodule.val_dataloader()      # eval B=2

    log.info(f"Loading module with weights from <{cfg.checkpoint_path}>...")
    module = hydra.utils.instantiate(cfg.module, cfg=cfg)
    module = load_from_checkpoint(module, cfg.checkpoint_path, cfg.device)   # strict=False 로드
    module.eval()

    # 추론 결정성: densification의 clone jitter가 torch.randn을 쓰므로 forward가
    # 확률적이다. 시드를 고정하지 않으면 동일 ckpt·동일 설정에서도 실행마다
    # 결과가 달라진다(실측: IDS ±69, AMOTA ±0.004). 보고 수치의 재현성을 위해 고정.
    _seed = int(cfg.get("eval_seed", 0))
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    np.random.seed(_seed)
    log.info(f"Inference RNG seeded with {_seed} (densification jitter is stochastic)")

    # 추론 전용 혼합정밀도(카메라 분기 한정). radar 인코더(PTv3)는 spconv에 half 커널이
    # 없어 전역 autocast가 불가하므로, Pixels-to-Gaussians에만 autocast를 걸고 결과를
    # fp32로 되돌린다. cfg에 없으면 비활성(기존 동작과 동일).
    _amp = cfg.get("camera_amp", None)
    if _amp:
        _dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[str(_amp)]
        _orig_cam = module.model.forward_features_camera

        def _cam_amp(batch, _o=_orig_cam, _d=_dtype):
            with torch.autocast("cuda", dtype=_d):
                out = _o(batch)
            return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                    for k, v in out.items()}

        module.model.forward_features_camera = _cam_amp
        log.info(f"Camera-branch autocast enabled: {_amp} (radar branch stays fp32)")

    log.info("Loading NuScenes...")
    nusc = NuScenes(                              # scene name 조회용(metric 아님)
        version=cfg.version,
        dataroot=cfg.nusc_dataroot,
        verbose=False,
    )

    # =======================================================================
    # 2. Decoder + Tracker (query-based multi-class only)
    #    decoder: dense head 출력 → 박스([N,9]) 디코딩. tracker: 박스에 ID 부여.
    # =======================================================================
    from gaussianmot.modeling.components.cc3dt_tracker import CC3DTPPTracker  # 기본(유일) tracker
    from gaussianmot.modeling.components.query_bbox_decoder import QueryBBoxDecoder
    decoder = QueryBBoxDecoder(
        key=cfg.task.key,
        x_min=cfg.data.data_config.render.x_min,
        x_max=cfg.data.data_config.render.x_max,
        y_min=cfg.data.data_config.render.y_min,
        y_max=cfg.data.data_config.render.y_max,
        bev_h=cfg.data.data_config.render.bev_h,
        bev_w=cfg.data.data_config.render.bev_w,
        max_num=cfg.decoder.max_num,
        score_threshold=cfg.decoder.score_threshold,
        # [NMS fix] decode 후 클래스별 circle-NMS 반경 (None/미설정 = 기존 동작)
        nms_circle_radii=cfg.decoder.get("nms_circle_radii", None),
        # [중심보정 8/26] heatmap centroid 중심 재추정 (추론 전용, 재학습 불필요)
        centroid_refine=cfg.decoder.get("centroid_refine", False),
        centroid_min_ratio=cfg.decoder.get("centroid_min_ratio", 0.3),
        centroid_max_radius=cfg.decoder.get("centroid_max_radius", 12),
    )
    decoder.to(cfg.device)

    # CC-3DT++ tracker (KF3D + 3-way affinity: appearance/location/velocity). 글로벌 프레임 association.
    # tracker_cfg를 dict로 만들어 그대로 인스턴스화 + track_match ablation이 재사용(match_score_thr 가변).
    tracker_cfg = dict(
        match_score_thr=0.5,            # 매칭 허용 affinity 임계값 (w_app=0 스윕 최적: 0.5 → AMOTA 0.264)
        init_score_thr=0.4,             # 새 track 생성 score 임계값
        obj_score_thr=0.1,              # 추적 대상 최소 score
        max_miss=5,                     # 연속 미검출 허용 프레임(이후 track 제거)
        # [2026-07-17 ablation] hw3.0 ckpt에서 appearance 임베딩=노이즈 판명
        # (app-only AMOTA 0.120/IDS 19418; w_app=0 → 0.258 vs 기본 0.206) → 기본값을 motion/location으로 전환.
        # 임베딩 감독 복원(SupCon 3.0) 재학습 후 재평가 시 w_appearance 재도입 검토.
        w_appearance=0.0,               # appearance(embed) affinity 가중치 (노이즈 임베딩 → 0)
        w_location=0.5,                 # 위치(글로벌 거리) affinity 가중치
        w_velocity=0.5,                 # 속도 affinity 가중치
        r_loc=4.0,                      # 위치 거리 정규화 반경(m)
        r_vel=3.0,                      # 속도 차이 정규화 반경
        with_cats=True,                 # 같은 클래스끼리만 매칭
        embed_momentum=0.9,             # track embed EMA 갱신 계수
        dt=cfg.tracker.dt,              # 프레임 간 시간(KF motion model)
    )
    tracker = CC3DTPPTracker(**tracker_cfg)

    # =======================================================================
    # 3. Prediction 수집 (val 전체 1회 순회: model→decoder→tracker + runtime 측정)
    # =======================================================================
    log.info("Collecting predictions on val set (with runtime)...")
    viz_dir = f"{cfg.output_dir}/viz" if cfg.viz_enabled else None

    collector = PredictionCollector(
        model=module,
        decoder=decoder,
        tracker=tracker,
        device=cfg.device,
        nusc=nusc,
        viz_enabled=cfg.viz_enabled,
        viz_dir=viz_dir,
        viz_bev_h=cfg.viz_bev_h,
        viz_bev_w=cfg.viz_bev_w,
        viz_max_trail=cfg.viz_max_trail,
        viz_score_threshold=cfg.get("viz_score_threshold", 0.1),
    )
    # [진단 후처리] cfg.diag.* 플래그 전달 (기본 off — evaluate.yaml 참조)
    diag_cfg = cfg.get("diag", {}) or {}
    # 주의: "project" 같은 모드 문자열을 보존해야 하므로 bool() 캐스팅 금지
    collector.diag_vel_from_radar = diag_cfg.get("vel_from_radar", False)
    collector.diag_yaw_from_velocity = bool(diag_cfg.get("yaw_from_velocity", False))
    if collector.diag_vel_from_radar or collector.diag_yaw_from_velocity:
        log.warning(f"[DIAG] postprocess ON: vel_from_radar={collector.diag_vel_from_radar}, "
                    f"yaw_from_velocity={collector.diag_yaw_from_velocity} — 상한측정용, 논문수치 아님")
    predictions, timings = collector.run(val_loader)   # 여기서 모델 forward는 단 1회(이후 재사용)
    log.info(f"Collected {len(predictions)} predictions.")

    # =======================================================================
    # 4. Main metrics — 전체 val에 대한 공식 Detection/Tracking 수치
    # =======================================================================
    log.info("Computing main metrics...")
    det_main = _compute_det_metric(
        predictions,
        nusc_dataroot=cfg.nusc_dataroot,
        version=cfg.version,
        eval_set=cfg.eval_set,
        output_dir=f"{cfg.output_dir}/main",
    )
    track_main = _compute_track_metric(
        predictions,
        nusc_dataroot=cfg.nusc_dataroot,
        version=cfg.version,
        eval_set=cfg.eval_set,
        output_dir=f"{cfg.output_dir}/main",
    )

    # =======================================================================
    # 5. Ablations — config로 켜진 것만 실행(수집된 predictions 재사용)
    # =======================================================================
    ablations = {}

    if cfg.ablation.score_threshold.enabled:
        log.info("Running ablation 1/4: Score Threshold...")
        ablations["score_threshold"] = ablation_score_threshold(
            predictions,
            thresholds=list(cfg.ablation.score_threshold.thresholds),
            nusc_dataroot=cfg.nusc_dataroot,
            version=cfg.version,
            eval_set=cfg.eval_set,
            output_dir=f"{cfg.output_dir}/ablation",
        )

    if cfg.ablation.distance_range.enabled:
        log.info("Running ablation 2/4: Distance Range...")
        ablations["distance_range"] = ablation_distance_range(
            predictions,
            ranges=[tuple(r) for r in cfg.ablation.distance_range.ranges],
            nusc_dataroot=cfg.nusc_dataroot,
            version=cfg.version,
            eval_set=cfg.eval_set,
            output_dir=f"{cfg.output_dir}/ablation",
        )

    if cfg.ablation.weather.enabled:
        log.info("Running ablation 3/4: Weather...")
        ablations["weather"] = ablation_weather(
            predictions,
            splits=VALIDATION_DRN_SPLITS,
            nusc_dataroot=cfg.nusc_dataroot,
            version=cfg.version,
            eval_set=cfg.eval_set,
            output_dir=f"{cfg.output_dir}/ablation",
        )

    if cfg.ablation.track_match.enabled:
        log.info("Running ablation 4/4: Track Match Threshold...")
        ablations["track_match"] = ablation_track_match(
            predictions,
            match_thresholds=list(cfg.ablation.track_match.thresholds),
            tracker_cfg=tracker_cfg,
            nusc_dataroot=cfg.nusc_dataroot,
            version=cfg.version,
            eval_set=cfg.eval_set,
            output_dir=f"{cfg.output_dir}/ablation",
        )

    if cfg.ablation.get("affinity", {}).get("enabled", False):
        log.info("Running ablation 5: Affinity Weights (app/loc/vel)...")
        ablations["affinity"] = ablation_affinity(
            predictions,
            weight_configs=[list(w) for w in cfg.ablation.affinity.configs],
            tracker_cfg=tracker_cfg,
            nusc_dataroot=cfg.nusc_dataroot,
            version=cfg.version,
            eval_set=cfg.eval_set,
            output_dir=f"{cfg.output_dir}/ablation",
        )

    # =======================================================================
    # 6. 결과 출력 + 저장 — Rich 표 → plain-text 요약 → JSON → (옵션) fig6
    # =======================================================================
    console.rule("[bold red]Evaluation Results")
    print_main_results(det_main, track_main, console)
    print_runtime_analysis(timings, console)

    if "score_threshold" in ablations:
        print_score_threshold_ablation(ablations["score_threshold"], console)
    if "distance_range" in ablations:
        print_distance_range_ablation(ablations["distance_range"], console)
    if "weather" in ablations:
        print_weather_ablation(ablations["weather"], console)
    if "track_match" in ablations:
        print_track_match_ablation(ablations["track_match"], console)

    console.rule()

    # Plain-text summary (rich 미지원 환경/파이프 대응)
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("-" * 80)
    print(
        f"DETECTION  | mAP={det_main['mAP']:.4f}  NDS={det_main['NDS']:.4f}  "
        f"mATE={det_main['mATE']:.4f}  mASE={det_main['mASE']:.4f}  "
        f"mAOE={det_main['mAOE']:.4f}  mAVE={det_main['mAVE']:.4f}  "
        f"mAAE={det_main['mAAE']:.4f}"
    )
    print(
        f"TRACKING   | AMOTA={track_main['AMOTA']:.4f}  AMOTP={track_main['AMOTP']:.4f}  "
        f"MOTA={track_main['MOTA']:.4f}  MOTP={track_main['MOTP']:.4f}  "
        f"IDS={int(track_main['IDS'])}  FRAG={int(track_main['FRAG'])}"
    )
    total_ms = timings.get("total_ms", []) if timings else []
    if total_ms:
        total_arr = np.array(total_ms)
        print(
            f"RUNTIME    | total {total_arr.mean():.2f}ms  "
            f"({1000.0 / total_arr.mean():.2f} FPS)"
        )
    print("=" * 80 + "\n")

    save_all_results(                  # metrics_summary.json 저장
        det_main, track_main, ablations, cfg.output_dir,
        timings=timings,
    )
    log.info(f"✓ Results saved: {cfg.output_dir}/metrics_summary.json")

    if cfg.get("fig6", False):         # cfg.fig6=true면 정성 그림 추가 생성
        generate_fig6(cfg)


if __name__ == "__main__":
    main()
