# ============================================================================
# [실행 컨텍스트 / 검증된 런타임 사실]
#   - 이 파일의 metric들(DetectionNuScenesMetric / TrackingNuScenesMetric)은
#     "평가(tools/evaluate.py)"에서만 사용된다.
#     train.yaml·evaluate.yaml 모두 metrics: {} (빈 dict)라서
#     학습 중에는 이 metric들이 호출되지 않는다 (update/compute 미실행).
#   - 평가 공통 차원:
#       * eval batch B=2  (evaluate.yaml trainer.batch_size)
#       * BEV grid 200×200, x,y ∈ [-50,50] (0.5m/cell)
#       * num_classes=10
#   - box 9-dim = (x, y, z, w, l, h, yaw_bev, vx, vy), lidar/BEV-plane frame.
# ============================================================================
import torchmetrics                       # torchmetrics.Metric 기반 지표 클래스(상태 누적·분산 reduce 지원)
import numpy as np                         # 좌표 변환 수식(행렬곱, arctan2 등)은 numpy로 처리
import torch                               # 모델 출력 텐서 처리 및 IoU 누적 상태(torch.zeros 등)

import json                                # nuScenes 제출 JSON(submission) 저장/직렬화
import os                                  # (현재 직접 사용은 없으나 경로 관련 유틸 import)
import tempfile                            # (임시 파일 관련 import; 현 로직에서는 미사용)
from pathlib import Path                   # output_dir 등 경로를 Path 객체로 다룸
from typing import List, Dict, Any, Optional   # 타입 힌트(가독성/정적 검사용)

from pyquaternion import Quaternion        # global yaw 각도를 쿼터니언(w,x,y,z)으로 변환(nuScenes rotation 포맷)


# ============================================================================
# nuScenes detection eval 포맷 변환 유틸리티.
# (inlined from gaussianmot/utils/nuscenes_format.py)
#
# 좌표계:
#   - BBoxDecoder 출력 box: BEV-plane frame
#     - labels.py에서 yaw = -box_data[6] - π/2로 변환된 상태
#     - position은 lidar-ego-flat frame
#   - nuScenes eval 입력: global frame
#     - pose = world_from_egolidarflat (4x4, flat=True → yaw 회전만)
# ============================================================================

# nuScenes 공식 detection class id ↔ name
# DYNAMIC (gaussianmot/data/nuscenes_dataset.py) 순서와 1:1 매칭됨.
CLASS_ID_TO_NAME = {
    # 모델이 출력하는 정수 class_id → nuScenes 공식 detection class 이름.
    # 이 매핑이 어긋나면 eval에서 클래스별 AP가 전부 0이 되므로 dataset 정의와 반드시 일치해야 함.
    0: "car",
    1: "truck",
    2: "bus",
    3: "trailer",
    4: "construction_vehicle",
    5: "pedestrian",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic_cone",
    9: "barrier",
}


# Static obstacles는 nuScenes detection eval에서 attribute가 빈 문자열.
# 아래 표는 속도 정보가 없을 때 쓰는 클래스별 기본(fallback) attribute.
# (실제로는 lidar_box_to_nusc_global에서 속도 기반으로 attribute를 다시 고르며,
#  이 표는 분기에 걸리지 않는 예외 클래스의 fallback으로만 사용됨.)
DEFAULT_ATTR = {
    "car": "vehicle.moving",
    "truck": "vehicle.moving",
    "bus": "vehicle.moving",
    "trailer": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "pedestrian": "pedestrian.moving",
    "motorcycle": "cycle.with_rider",
    "bicycle": "cycle.with_rider",
    "traffic_cone": "",   # cone/barrier는 attribute 개념이 없어 빈 문자열
    "barrier": "",
}


def bev_yaw_to_lidar_yaw(bev_yaw: float) -> float:
    """
    BEV-plane frame yaw → lidar frame yaw 역변환.
    labels.py에서 (GT 생성 시) yaw = -box_data[6] - π/2 로 변환됨.
    따라서 모델이 학습한 BEV-plane yaw를 lidar yaw로 되돌리려면 같은 식을 그대로 적용.
    역변환: lidar_yaw = -bev_yaw - π/2
    """
    return -bev_yaw - np.pi / 2   # BEV yaw 부호 반전 후 -π/2 오프셋 제거(=동일 식 재적용)


def lidar_box_to_nusc_global(
    box_lidar: np.ndarray,   # [9] (x,y,z,w,l,h,yaw_bev,vx,vy)
    pose: np.ndarray,        # [4,4] lidar→global (flat)
    score: float,
    class_id: int,
    sample_token: str,
) -> Optional[Dict[str, Any]]:
    """
    Lidar-frame box → nuScenes detection result dict.
    NaN, Inf, 비정상 size가 있으면 None 반환.

    입력: box_lidar [9]=(x,y,z,w,l,h,yaw_bev,vx,vy), pose [4,4].
    반환 dict: translation [3] / size [3] / rotation [4](w,x,y,z) / velocity [2]
              (+ sample_token, detection_name, detection_score, attribute_name).
    """
    # NaN/Inf가 섞인 박스는 eval에서 예외를 유발하므로 즉시 폐기(None).
    if np.any(np.isnan(box_lidar)) or np.any(np.isinf(box_lidar)):
        return None

    # box_lidar(길이 9) 언팩: 중심(x,y,z) / 크기(w,l,h) / BEV yaw / 속도(vx,vy)
    # box_lidar = [9] (스칼라 9개), 각 변수는 python float 스칼라
    x, y, z, w, l, h, yaw_bev, vx, vy = box_lidar

    # 비정상 size 필터 — 음수/0 크기 박스는 부피가 없어 eval IoU 계산이 깨짐.
    if w <= 0 or l <= 0 or h <= 0:
        return None
    if w > 50 or l > 50 or h > 20:   # 너무 큰 박스 거부(차량 스케일 벗어난 발산 출력 제거)
        return None

    # 0. pose 차원 안전화 ([4,4] 또는 [1,4,4] 가능) — lidar→global 변환행렬을 (4,4)로 정규화
    pose = np.asarray(pose)
    if pose.ndim == 3:
        pose = pose[0]               # 배치 차원이 끼어있으면 첫 원소만 사용
    if pose.shape != (4, 4):
        return None                  # (4,4)가 아니면 변환 불가 → 폐기

    # 0. pose 차원 안전화  (※ 위와 중복된 안전화 블록 — 기존 코드 그대로 유지)
    pose = np.asarray(pose)
    if pose.ndim == 3 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.shape != (4, 4):
        return None

    # 1. BEV yaw → lidar yaw : eval은 lidar 기준 각도를 거쳐 global로 가야 하므로 먼저 역변환
    yaw_lidar = bev_yaw_to_lidar_yaw(float(yaw_bev))

    # 2. Position : lidar frame 중심점을 동차좌표로 만들고 pose(@)로 곱해 global frame 위치를 얻음
    #    (왜 pose @ center인가: pose = world_from_egolidarflat 변환행렬이라 점을 직접 global로 옮김)
    center_lidar_hom = np.array([x, y, z, 1.0])   # 동차좌표 [4] = [x,y,z,1]
    center_global = pose @ center_lidar_hom       # global frame 좌표(동차) [4]
    center_global = center_global[:3]             # 앞 3개만 취해 (x,y,z) 위치 → [3]

    # 3. Rotation : pose의 회전부에서 ego의 global yaw를 추출해 박스 yaw에 더함
    pose_R = pose[:3, :3]                          # pose의 상단 3x3 회전행렬 [3,3]
    pose_yaw = np.arctan2(pose_R[1, 0], pose_R[0, 0])  # 회전행렬에서 z축 yaw 각도 복원
    yaw_global = yaw_lidar + pose_yaw              # lidar yaw + ego global yaw = global yaw
    rot_global = Quaternion(axis=[0, 0, 1], angle=yaw_global)  # z축 회전 쿼터니언(nuScenes rotation 포맷)

    # 4. Velocity: lidar → global (rotation만) — 속도는 방향 벡터라 translation 없이 회전만 적용
    v_lidar = np.array([float(vx), float(vy), 0.0])   # lidar frame 속도(평면, z=0) [3]
    v_global = pose_R @ v_lidar                       # 회전행렬만 곱해 global frame 속도로 변환 [3]

    # 5. Class name + attribute (속도 기반 동적 선택, nuScenes 3-state)
    # vehicle: moving (>0.2) / stopped (0 < v ≤ 0.2) / parked (v=0)
    # nuScenes annotation은 신호 대기 등 일시 정지를 'stopped'로 분류하므로
    # 정확히 0이 아니면 stopped로 보내는 것이 mini_val 분포와 더 잘 맞음.
    class_name = CLASS_ID_TO_NAME.get(int(class_id), "car")   # id→이름(미지정 id는 car로 fallback)
    speed = float(np.linalg.norm(v_global[:2]))   # global 평면 속도 크기(m/s) — attribute 분기 기준
    if class_name in ("car", "truck", "bus", "trailer", "construction_vehicle"):
        # 차량류 3-state: 빠르면 moving / 느리지만 움직이면 stopped / 완전 정지면 parked
        if speed > 0.2:
            attr = "vehicle.moving"
        elif speed > 0.0:
            attr = "vehicle.stopped"   # 신호 대기 등 일시 정지(0은 아님)
        else:
            attr = "vehicle.parked"    # 속도 정확히 0 → 주차로 간주
    elif class_name == "pedestrian":
        # 보행자는 moving/standing 2-state(임계 0.1m/s)
        attr = "pedestrian.moving" if speed > 0.1 else "pedestrian.standing"
    elif class_name in ("motorcycle", "bicycle"):
        # Cycle은 with/without rider 선택. 속도가 있으면 rider 있다고 간주.
        attr = "cycle.with_rider" if speed > 0.2 else "cycle.without_rider"
    elif class_name in ("traffic_cone", "barrier"):
        attr = ""                      # 정적 장애물은 attribute 없음
    else:
        attr = DEFAULT_ATTR.get(class_name, "")   # 위 분기에 없는 클래스는 기본 표 사용

    # nuScenes detection result 1건(dict) 구성 — submission JSON의 results 리스트 원소.
    # 반환 dict: translation [3] / size [3] / rotation [4](쿼터니언 w,x,y,z) / velocity [2]
    return {
        "sample_token": sample_token,                       # 이 박스가 속한 sample(프레임) 토큰
        "translation": [float(c) for c in center_global],   # global frame 중심 [3] = [x,y,z]
        "size": [float(w), float(l), float(h)],             # 박스 크기 [3] = [width,length,height]
        "rotation": [float(rot_global.w), float(rot_global.x),   # global yaw 쿼터니언 [4] = [w,x,y,z]
                     float(rot_global.y), float(rot_global.z)],
        "velocity": [float(v_global[0]), float(v_global[1])],    # global frame 속도 [2] = [vx,vy]
        "detection_name": class_name,                       # eval에서 클래스별 AP 집계용 이름
        "detection_score": float(score),                    # confidence(PR-curve 정렬 기준)
        "attribute_name": attr,                             # mAAE(attribute error) 평가용
    }


def build_nusc_submission(
    all_results: Dict[str, List[Dict[str, Any]]],
    use_camera: bool = True,
    use_radar: bool = True,
    use_lidar: bool = False,
    use_map: bool = False,
    use_external: bool = False,
) -> Dict[str, Any]:
    """
    sample_token → list of detection dicts를 nuScenes submission 형식으로 포장.
    최상위는 meta(사용 센서 플래그)와 results(토큰별 검출 리스트) 두 키로 구성.
    """
    return {
        "meta": {                     # 어떤 입력 모달리티를 썼는지 명시(공식 제출 규약)
            "use_camera": use_camera,
            "use_lidar": use_lidar,
            "use_radar": use_radar,
            "use_map": use_map,
            "use_external": use_external,
        },
        "results": all_results,       # {sample_token: [detection dict, ...]}
    }



# ========================================================================
# Tracking format conversion
# ========================================================================

# nuScenes tracking은 7개 클래스만 허용 (cone/barrier/construction_vehicle 제외).
# tracking submission에 이 외 클래스가 들어가면 eval이 무시하거나 에러를 낼 수 있음.
TRACKING_CLASSES = {
    "car", "truck", "bus", "trailer",
    "pedestrian", "motorcycle", "bicycle",
}


def lidar_track_to_nusc_global(
    box_lidar: np.ndarray,
    pose: np.ndarray,
    score: float,
    class_id: int,
    sample_token: str,
    tracking_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Lidar-frame box + track ID → nuScenes tracking result dict.
    lidar_box_to_nusc_global()과 동일한 좌표 변환 재사용(중복 구현 방지).
    """
    # 먼저 detection 변환을 그대로 수행해 global frame 박스(dict)를 얻음.
    det = lidar_box_to_nusc_global(
        box_lidar=box_lidar,
        pose=pose,
        score=score,
        class_id=class_id,
        sample_token=sample_token,
    )
    if det is None:
        return None                          # invalid 박스면 트랙도 폐기

    # detection dict 필드를 tracking 포맷 키 이름으로 재포장(+ tracking_id 추가).
    return {
        "sample_token": det["sample_token"],
        "translation": det["translation"],   # global 중심 [x,y,z]
        "size": det["size"],                 # 크기 [w,l,h]
        "rotation": det["rotation"],         # global yaw 쿼터니언
        "velocity": det["velocity"],         # global 속도 [vx,vy]
        "tracking_id": f"track_{tracking_id}",   # 시간축 동일 물체 식별자(문자열)
        "tracking_name": det["detection_name"],  # tracking은 detection_name을 그대로 사용
        "tracking_score": float(score),      # AMOTA recall-threshold 정렬 기준
    }


def build_nusc_tracking_submission(
    all_results: Dict[str, List[Dict[str, Any]]],
    use_camera: bool = True,
    use_radar: bool = True,
    use_lidar: bool = False,
    use_map: bool = False,
    use_external: bool = False,
) -> Dict[str, Any]:
    """nuScenes tracking submission 형식 (detection과 동일한 구조)."""
    return {
        "meta": {                     # 사용 센서 플래그(detection submission과 동일)
            "use_camera": use_camera,
            "use_lidar": use_lidar,
            "use_radar": use_radar,
            "use_map": use_map,
            "use_external": use_external,
        },
        "results": all_results,       # {sample_token: [tracking dict, ...]}
    }


# ========================================================================
# Detection Metric (nuScenes 공식 평가)
# ========================================================================


class DetectionNuScenesMetric(torchmetrics.Metric):
    """
    nuScenes 공식 Detection Eval.

    [실행 컨텍스트] 학습 중 미사용(train.yaml/evaluate.yaml 모두 metrics:{}),
    평가(tools/evaluate.py)에서만 update/compute가 호출된다.

    사용법:
      metric = DetectionNuScenesMetric(...)
      for batch in loader:
          pred_results = bbox_decoder(model(batch)["output"])
          metric.update(pred_results, batch)
      scores = metric.compute()
      # scores = {"mAP": ..., "NDS": ..., "mAVE": ..., ...}  ← 전부 python float(스칼라)
    """
    # 상태는 GPU로 올릴 필요 없음 (CPU list에 sample 단위로 누적)
    is_differentiable = False    # 미분 불가(공식 eval 호출 결과라 gradient 없음)
    higher_is_better = True       # mAP/NDS는 클수록 좋음
    full_state_update = False     # update마다 누적만, 전체 상태 재계산 불필요(메모리/속도)

    def __init__(
        self,
        nusc_dataroot: str = "/root/data/nuscenes/nuscenes/original/full",   # nuScenes 원본 데이터 경로
        version: str = "v1.0-trainval",       # nuScenes 버전(mini/trainval)
        eval_set: str = "val",                # 평가 split(val/mini_val 등)
        output_dir: str = "logs/eval",        # submission JSON·eval 결과 저장 위치
        class_names: Optional[List[str]] = None,   # 평가 대상 클래스 이름 목록
        score_threshold: float = 0.1,         # 이 점수 미만 검출은 제출 전 제거
        use_camera: bool = True,              # 입력 모달리티 플래그(meta에 기록)
        use_radar: bool = True,
        use_lidar: bool = False,
    ):
        super().__init__()
        self._box_converter = lidar_box_to_nusc_global   # lidar→global 변환 함수 참조 보관

        self.nusc_dataroot = nusc_dataroot
        self.version = version
        self.eval_set = eval_set
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)   # 출력 디렉터리 미리 생성
        self.class_names = class_names or ["car"]            # 미지정 시 car만
        self.score_threshold = score_threshold
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.use_lidar = use_lidar

        # sample_token → list of detection dicts
        # torchmetrics state로 등록 (nondefault: list)
        self.add_state(
            "sample_results",      # 프레임별 검출 결과를 모으는 리스트 상태
            default=[],
            dist_reduce_fx=None,  # dict 구조이므로 커스텀 병합 필요(자동 reduce 비활성)
        )
    
    def update(
        self,
        pred_results: List[Dict[str, torch.Tensor]],
        batch: Dict[str, Any],
    ):
        """
        Args:
            pred_results: QueryBBoxDecoder 출력. List[dict], 길이 B(=2).
                각 dict: {"boxes_3d": [N,9], "scores": [N], "labels": [N], "embeds": ...}
                N = 그 샘플의 검출 수(score≥threshold 필터 후, 0 가능 → boxes_3d [0,9]).
            batch: data batch
                - "token": str 또는 List[str] (길이 B)
                - "pose":  [B, 4, 4] Tensor (또는 [4,4])
        """
        tokens = batch["token"]              # 배치 내 각 프레임의 sample_token (str 또는 List[str] 길이 B)
        if isinstance(tokens, str):
            tokens = [tokens]                # 단일 토큰이면 리스트로 감싸 일관 처리

        poses = batch["pose"]                # lidar→global 변환행렬 [B,4,4] (또는 [4,4])
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()      # 텐서면 numpy로(변환 수식은 numpy로 처리)
        else:
            poses = np.array(poses)

        B = len(pred_results)                # 배치 크기(=프레임 수), eval에서 B=2
        for b in range(B):
            token = tokens[b] if isinstance(tokens, (list, tuple)) else tokens   # b번째 프레임 토큰(str)
            pose = poses[b] if poses.ndim == 3 else poses                        # b번째 pose [4,4]

            # 모델 출력(텐서)을 numpy로: 박스[N,9] / 점수[N] / 라벨[N] (N=0 가능)
            boxes_3d = pred_results[b]["boxes_3d"].cpu().numpy()   # [N,9]
            scores = pred_results[b]["scores"].cpu().numpy()       # [N]
            labels = pred_results[b]["labels"].cpu().numpy()       # [N]

            # score threshold 필터링 — 저신뢰 검출 제거(제출 크기/precision 관리)
            mask = scores >= self.score_threshold
            boxes_3d = boxes_3d[mask]
            scores = scores[mask]
            labels = labels[mask]

            # 변환 (None 반환 = invalid box, 건너뜀)
            dets = []
            for i in range(len(boxes_3d)):
                det = self._box_converter(           # lidar 박스 → global frame nuScenes dict
                    box_lidar=boxes_3d[i],           # [9] (i번째 박스)
                    pose=pose,                       # [4,4]
                    score=float(scores[i]),
                    class_id=int(labels[i]),
                    sample_token=token,
                )
                if det is not None:
                    dets.append(det)                 # 유효 박스만 수집

            # 누적 (sample별 dict) — compute()에서 토큰 단위로 병합됨
            self.sample_results.append({"token": token, "detections": dets})
    
    def compute(self) -> Dict[str, float]:
        """
        nuScenes DetectionEval 호출 → metrics 반환.
        모든 prediction이 비어있으면 0 반환 (eval 우회).
        """
        # nuScenes 공식 eval 모듈은 무거우므로 compute 시점에만 지연 import.
        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import DetectionEval

        # 1. sample_token → list of detections (dedupe)
        #    같은 토큰이 여러 update에 나뉘어 왔을 수 있으므로 토큰 단위로 합침.
        all_results = {}
        for entry in self.sample_results:
            token = entry["token"]
            dets = entry["detections"]
            if token in all_results:
                all_results[token].extend(dets)   # 기존 토큰이면 검출 추가
            else:
                all_results[token] = dets          # 새 토큰이면 신규 등록

        # 빈 결과 사전 체크 — 검출이 전혀 없으면 eval이 에러나므로 0점으로 우회.
        total_dets = sum(len(v) for v in all_results.values())
        if total_dets == 0:
            print("[DetectionMetric] No detections found; returning zeros.")
            return {
                "mAP": 0.0, "NDS": 0.0,         # 점수류는 0
                "mATE": 1.0, "mASE": 1.0, "mAOE": 1.0,   # error류는 최악값 1.0
                "mAVE": 1.0, "mAAE": 1.0,
            }

        # 2. submission JSON 저장 — 공식 evaluator는 파일 경로를 입력으로 받음
        submission = build_nusc_submission(
            all_results,
            use_camera=self.use_camera,
            use_radar=self.use_radar,
            use_lidar=self.use_lidar,
        )
        result_path = self.output_dir / "results_nusc.json"
        with open(result_path, "w") as f:
            json.dump(submission, f)            # 제출 dict를 JSON으로 직렬화

        # 3. nuScenes DetectionEval 실행
        nusc = NuScenes(                         # nuScenes DB 로드(메타데이터)
            version=self.version,
            dataroot=self.nusc_dataroot,
            verbose=False,
        )
        cfg = config_factory("detection_cvpr_2019")   # 공식 detection 평가 설정(거리 임계 등)

        eval_output_dir = self.output_dir / "nusc_eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        evaluator = DetectionEval(               # 제출 JSON vs GT 비교 평가기
            nusc=nusc,
            config=cfg,
            result_path=str(result_path),
            eval_set=self.eval_set,
            output_dir=str(eval_output_dir),
            verbose=False,
        )
        metrics_summary = evaluator.main(render_curves=False)   # 실제 평가 수행(곡선 렌더는 생략)

        # 4. 요약 값 파싱 — 공식 metric dict에서 핵심 지표만 뽑아 반환
        # 반환: {mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE} — 전부 python float(스칼라)
        return {
            "mAP": float(metrics_summary["mean_ap"]),                       # 평균 정밀도
            "NDS": float(metrics_summary["nd_score"]),                      # nuScenes 종합 점수
            "mATE": float(metrics_summary["tp_errors"]["trans_err"]),       # 위치 오차
            "mASE": float(metrics_summary["tp_errors"]["scale_err"]),       # 크기 오차
            "mAOE": float(metrics_summary["tp_errors"]["orient_err"]),      # 방향(yaw) 오차
            "mAVE": float(metrics_summary["tp_errors"]["vel_err"]),         # 속도 오차
            "mAAE": float(metrics_summary["tp_errors"]["attr_err"]),        # attribute 오차
        }

    def reset(self):
        super().reset()                # torchmetrics 내부 상태 초기화
        self.sample_results = []        # 누적 리스트 비우기(다음 eval 라운드 대비)



class TrackingNuScenesMetric(torchmetrics.Metric):
    """
    nuScenes 공식 Tracking Eval.

    [실행 컨텍스트] 학습 중 미사용(metrics:{}), 평가(tools/evaluate.py)에서만 호출.

    DetectionNuScenesMetric과 동일한 인터페이스 (pred_results: List[dict] 길이 B=2;
    각 dict boxes_3d [N,9]/scores [N]/labels [N]).
    단, pred_results[b]에 track_ids [N] 필드가 포함되어야 함(-1=unmatched, tid≥0만 통과).

    사용법:
      metric = TrackingNuScenesMetric(...)
      
      # 매 frame (Tracker로 track_ids 추가 후)
      for b in range(len(pred_results)):
          pred_results[b]["track_ids"] = tracker.update(...)
      metric.update(pred_results, batch)
      
      # epoch 끝
      scores = metric.compute()
      # scores = {"AMOTA": ..., "AMOTP": ..., "IDS": ..., ...}
      #   AMOTA/AMOTP/MOTA/MOTP = python float(스칼라), IDS/FRAG/FP/FN = int
    """
    is_differentiable = False     # 공식 eval 결과라 미분 불가
    higher_is_better = True        # AMOTA/MOTA는 클수록 좋음
    full_state_update = False      # 누적만, 전체 재계산 불필요

    def __init__(
        self,
        nusc_dataroot: str = "/root/data/nuscenes/nuscenes/original/full",
        version: str = "v1.0-trainval",
        eval_set: str = "val",
        output_dir: str = "logs/eval",
        score_threshold: float = 0.1,
        use_camera: bool = True,
        use_radar: bool = True,
        use_lidar: bool = False,
    ):
        super().__init__()
        self._box_converter = lidar_track_to_nusc_global   # track_id 포함 변환 함수 참조

        self.nusc_dataroot = nusc_dataroot
        self.version = version
        self.eval_set = eval_set
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.score_threshold = score_threshold
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.use_lidar = use_lidar

        self.add_state(
            "sample_results",      # 프레임별 트랙 결과 누적 리스트
            default=[],
            dist_reduce_fx=None,   # dict 구조라 자동 reduce 비활성
        )

    def update(
        self,
        pred_results: List[Dict[str, torch.Tensor]],
        batch: Dict[str, Any],
    ):
        """
        pred_results[b]는 DetectionNuScenesMetric 입력에 추가로
        'track_ids' [N] 필드를 포함해야 함. (Tracker.update()로 부여된 시간축 ID)
        Detection update와 동일 shape: boxes_3d [N,9], scores [N], labels [N], track_ids [N].
        batch["token"]=str/List[str](길이 B=2), batch["pose"]=[B,4,4](또는 [4,4]).
        """
        tokens = batch["token"]              # 프레임별 sample_token (str 또는 List[str] 길이 B)
        if isinstance(tokens, str):
            tokens = [tokens]

        poses = batch["pose"]                # lidar→global 변환행렬 [B,4,4] (또는 [4,4])
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        else:
            poses = np.array(poses)

        B = len(pred_results)                # eval에서 B=2
        for b in range(B):
            token = tokens[b] if isinstance(tokens, (list, tuple)) else tokens   # b번째 프레임 토큰(str)
            pose = poses[b] if poses.ndim == 3 else poses                        # b번째 pose [4,4]

            boxes_3d = pred_results[b]["boxes_3d"].cpu().numpy()   # 박스 [N,9]
            scores = pred_results[b]["scores"].cpu().numpy()       # 점수 [N]
            labels = pred_results[b]["labels"].cpu().numpy()       # 라벨 [N]

            # track_ids 필수 — 추적 평가에는 시간축 식별자가 반드시 있어야 함
            if "track_ids" not in pred_results[b]:
                raise KeyError(
                    f"pred_results[{b}] must contain 'track_ids'. "
                    f"Call tracker.update() before metric.update()."
                )
            tids = pred_results[b]["track_ids"]   # 트랙 ID [N] (-1=unmatched, tid≥0만 통과)
            if isinstance(tids, torch.Tensor):
                tids = tids.cpu().numpy()          # [N]

            # 필터: score threshold + valid track (tid >= 0) — 미할당(-1) 트랙 제외
            mask = (scores >= self.score_threshold) & (tids >= 0)
            boxes_3d = boxes_3d[mask]
            scores = scores[mask]
            labels = labels[mask]
            tids = tids[mask]

            tracks = []
            for i in range(len(boxes_3d)):
                class_name = CLASS_ID_TO_NAME.get(int(labels[i]), "car")
                if class_name not in TRACKING_CLASSES:
                    continue                      # tracking 허용 7클래스가 아니면 스킵

                track = self._box_converter(      # lidar 박스 → global tracking dict
                    box_lidar=boxes_3d[i],        # [9] (i번째 박스)
                    pose=pose,                    # [4,4]
                    score=float(scores[i]),
                    class_id=int(labels[i]),
                    sample_token=token,
                    tracking_id=int(tids[i]),     # 시간축 트랙 ID 전달
                )
                if track is not None:
                    tracks.append(track)          # 유효 트랙만 수집

            self.sample_results.append({"token": token, "tracks": tracks})   # 프레임별 누적
    
    def compute(self) -> Dict[str, float]:
        # 추적 평가기도 무거우므로 compute 시점에만 지연 import
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_config_factory

        # sample_token → tracks 병합 (같은 토큰의 트랙을 한 리스트로)
        all_results = {}
        for entry in self.sample_results:
            token = entry["token"]
            tracks = entry["tracks"]
            if token in all_results:
                all_results[token].extend(tracks)
            else:
                all_results[token] = tracks

        # 빈 결과 사전 체크 — 트랙이 없으면 eval 우회하여 0/최악값 반환
        total = sum(len(v) for v in all_results.values())
        if total == 0:
            print("[TrackingMetric] No tracks found; returning zeros.")
            return {
                "AMOTA": 0.0, "AMOTP": 1.0,    # 정확도 0 / 위치오차 최악
                "MOTA": 0.0, "MOTP": 1.0,
                "IDS": 0, "FRAG": 0, "FP": 0, "FN": 0,   # 카운트류 0
            }

        # submission 저장 — TrackingEval도 파일 경로 입력
        submission = build_nusc_tracking_submission(
            all_results,
            use_camera=self.use_camera,
            use_radar=self.use_radar,
            use_lidar=self.use_lidar,
        )
        result_path = self.output_dir / "results_nusc_tracking.json"
        with open(result_path, "w") as f:
            json.dump(submission, f)            # tracking 제출 JSON 직렬화

        # TrackingEval 실행
        eval_output_dir = self.output_dir / "nusc_tracking_eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        cfg = track_config_factory("tracking_nips_2019")   # 공식 tracking 평가 설정

        evaluator = TrackingEval(               # 제출 트랙 vs GT 트랙 비교 평가기
            config=cfg,
            result_path=str(result_path),
            eval_set=self.eval_set,
            output_dir=str(eval_output_dir),
            nusc_version=self.version,
            nusc_dataroot=self.nusc_dataroot,
            verbose=False,
        )
        metrics_summary = evaluator.main(render_curves=False)   # 실제 추적 평가 수행

        # 핵심 추적 지표 추출(없으면 기본값) — AMOTA가 1차 목표 지표
        # 반환: AMOTA/AMOTP/MOTA/MOTP = python float(스칼라), IDS/FRAG/FP/FN = int
        return {
            "AMOTA": float(metrics_summary.get("amota", 0.0)),   # 평균 MOTA(주요 추적 지표)
            "AMOTP": float(metrics_summary.get("amotp", 0.0)),   # 평균 MOTP(위치 정밀도)
            "MOTA":  float(metrics_summary.get("mota", 0.0)),    # 단일 임계 MOTA
            "MOTP":  float(metrics_summary.get("motp", 0.0)),    # 단일 임계 MOTP
            "IDS":   int(metrics_summary.get("ids", 0)),         # ID 스위치 횟수
            "FRAG":  int(metrics_summary.get("frag", 0)),        # 트랙 끊김(fragmentation) 횟수
            "FP":    int(metrics_summary.get("fp", 0)),          # false positive 수
            "FN":    int(metrics_summary.get("fn", 0)),          # false negative 수
        }

    def reset(self):
        super().reset()                # torchmetrics 상태 초기화
        self.sample_results = []        # 누적 트랙 리스트 비우기

