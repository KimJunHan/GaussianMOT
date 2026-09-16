"""CC-3DT++ -style 3-way association tracker (self-contained port).

Adapted from `vis4d/op/track3d/cc_3dt.py` + CR3DT's velocity-aware association.
This tracker keeps the same `update(boxes_3d, scores, embeds)` signature as
the previous ByteTrack-like Tracker so it can drop in to PredictionCollector.

Affinity decomposition (CR3DT-style):
    sim_total = w_app * cos_sim(embed)
              + w_loc * exp(-d_xy / r_loc)
              + w_vel * exp(-d_vel / r_vel)
Where:
  - w_app + w_loc + w_vel = 1
  - exp kernels keep affinity in [0, 1]

[한글 개요]
이 파일은 GaussMOT(GS 기반 camera-radar 3D 검출+추적, nuScenes) 파이프라인의
'추적(tracking)' 단계를 담당한다. 검출 단계가 매 프레임 3D 박스를 내놓으면,
이 트래커가 프레임들 사이에서 같은 물리적 객체에 일관된 정수 track ID를 붙인다.

핵심 아이디어 (CC-3DT++ 스타일):
  - KF3D(칼만필터 기반 등속도 모션 모델)로 각 track의 다음 위치를 '예측'한다.
  - 검출과 track 사이의 '유사도(affinity)'를 3가지로 분해해서 합산한다:
      1) appearance : track_head per-instance 임베딩의 코사인 유사도
      2) location   : BEV(x,y) 중심 거리 기반 (거리가 가까울수록 1에 가까움)
      3) velocity   : 속도 벡터 차이 기반 (속도가 비슷할수록 1에 가까움)
  - 이 affinity 행렬에 헝가리안(Hungarian) 매칭을 적용해 track↔det을 일대일 연결한다.

좌표계 주의: PredictionCollector는 ego motion(프레임당 약 2.5m 이동)을 상쇄하기 위해
박스를 '글로벌 프레임'으로 변환해 넘겨준다. 즉 이 트래커는 글로벌 프레임에서 동작한다.

호출 패턴: 평가(tools/evaluate.py)의 PredictionCollector가 매 프레임
`tracker.update(boxes_global, scores, embeds, labels)`를 호출하고,
scene(시퀀스)이 바뀌면 `tracker.reset()`을 호출해 내부 상태를 비운다.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np  # 모든 KF 연산/affinity 행렬을 numpy로 처리 (트래커는 CPU에서 동작)
from scipy.optimize import linear_sum_assignment  # 헝가리안 알고리즘 (최적 일대일 할당)


class _KF3DState:
    """Lightweight Kalman state for a single track. Constant-velocity in BEV.

    Tracks (x, y, vx, vy) with Kalman filter; remaining box dims (z, w, l, h, sin, cos)
    are exponentially smoothed (no KF) for simplicity.

    [한글] 단일 track 1개의 칼만필터(KF) 상태.
    - BEV 평면의 (x, y, vx, vy) 4차원만 정식 KF로 추정한다(등속도 모델).
    - 나머지 박스 차원(z, w, l, h, sin, cos)은 KF 없이 EMA(지수이동평균)로만 부드럽게
      갱신한다. 이들은 모션 예측에 핵심이 아니어서 단순화한 것.
    """

    def __init__(self, box: np.ndarray, dt: float = 0.5):
        # box [10]: cx, cy, cz, w, l, h, sin, cos, vx, vy  (요(yaw)는 sin/cos로 분해된 형태)
        self.dt = dt  # 프레임 간 시간 간격(초). nuScenes 키프레임 ≈ 0.5s
        # KF 상태벡터 x [4] = [x, y, vx, vy] : 위치(x,y)와 속도(vx,vy)
        self.x = np.array([box[0], box[1], box[8], box[9]], dtype=np.float64)  # [x, y, vx, vy]
        # Initial covariance
        # 초기 공분산 P [4,4]: 위치는 비교적 확실(1.0), 속도는 불확실(4.0)하게 시작
        self.P = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float64)
        # Process / measurement noise
        # Q [4,4]: 프로세스(모션 모델) 노이즈. 매 predict마다 불확실성을 키운다
        self.Q = np.diag([0.5, 0.5, 1.0, 1.0]).astype(np.float64)
        # R_pos [2,2]: 측정 노이즈. 검출 (x,y) 관측의 신뢰도(작을수록 검출을 신뢰)
        self.R_pos = np.diag([0.5, 0.5]).astype(np.float64)
        # Non-KF dims kept as EMA
        # KF로 추정하지 않는 차원들(EMA로만 갱신): z 높이, wlh 크기, yaw의 sin/cos
        self.z = float(box[2])                                          # z 중심 높이 (스칼라)
        self.wlh = np.array([box[3], box[4], box[5]], dtype=np.float64)  # 박스 크기 [w, l, h]
        self.sin = float(box[6])                                        # yaw 의 sin 성분
        self.cos = float(box[7])                                        # yaw 의 cos 성분

    @property
    def pos(self) -> np.ndarray:
        # 현재 추정 위치 [x, y] (KF 상태의 앞 2개) -> shape [2]
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        # 현재 추정 속도 [vx, vy] (KF 상태의 뒤 2개) -> shape [2]
        return self.x[2:]

    def predict(self) -> None:
        # ===== KF 예측(predict) 단계: 모션 모델로 다음 프레임 상태를 미리 굴린다 =====
        dt = self.dt
        # 상태 전이 행렬 F [4,4] : 등속도 모델. x' = x + vx*dt, y' = y + vy*dt, 속도는 유지
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.x = F @ self.x               # 상태 예측: x_pred = F x   -> shape [4]
        self.P = F @ self.P @ F.T + self.Q  # 공분산 예측: P_pred = F P F^T + Q (불확실성 증가) -> [4,4]

    def update(self, det_box: np.ndarray, alpha_dim: float = 0.7) -> None:
        # ===== KF 보정(update) 단계: 매칭된 검출(det_box)로 상태를 교정한다 =====
        # Measurement = [x, y] from detection
        z = np.array([det_box[0], det_box[1]], dtype=np.float64)  # 관측치 z [2] = 검출의 (x, y)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)  # 관측 행렬 H [2,4]: 상태→관측(위치만)
        y = z - H @ self.x                          # 잔차(innovation) y [2] = 관측 - 예측위치
        S = H @ self.P @ H.T + self.R_pos           # 잔차 공분산 S [2,2]
        K = self.P @ H.T @ np.linalg.inv(S)         # 칼만 이득 K [4,2]
        self.x = self.x + K @ y                     # 상태 보정: x = x + K y  -> [4]
        self.P = (np.eye(4) - K @ H) @ self.P       # 공분산 보정: P = (I - K H) P -> [4,4]
        # Override velocity with detection if provided (CR3DT uses radar velocity directly)
        # CR3DT는 레이더가 직접 측정한 속도를 활용한다. 검출 속도가 유효하면 KF 속도와 혼합.
        det_vx, det_vy = float(det_box[8]), float(det_box[9])  # 검출이 보고한 속도 (radar 기반)
        if abs(det_vx) > 1e-3 or abs(det_vy) > 1e-3:  # 속도가 의미있게 0이 아닐 때만
            # Blend KF velocity with measured detection velocity
            self.x[2] = 0.5 * self.x[2] + 0.5 * det_vx  # vx = KF속도와 측정속도 50:50 블렌드
            self.x[3] = 0.5 * self.x[3] + 0.5 * det_vy  # vy = 동일하게 블렌드
        # EMA-update z / wlh / yaw
        # KF로 추정 안 하는 차원들을 EMA로 갱신(alpha_dim=0.7 -> 기존 70% + 신규 30%)
        self.z = alpha_dim * self.z + (1 - alpha_dim) * float(det_box[2])      # z 높이 EMA
        self.wlh = alpha_dim * self.wlh + (1 - alpha_dim) * det_box[3:6]        # 크기 [w,l,h] EMA
        # Normalize sin/cos before blending
        # yaw 방향(sin,cos)은 단위벡터여야 하므로 블렌드 전에 정규화
        new_sin, new_cos = float(det_box[6]), float(det_box[7])
        n = (new_sin ** 2 + new_cos ** 2) ** 0.5 + 1e-8   # sin/cos 벡터의 노름
        new_sin /= n; new_cos /= n                         # 단위벡터로 정규화
        self.sin = alpha_dim * self.sin + (1 - alpha_dim) * new_sin  # sin EMA
        self.cos = alpha_dim * self.cos + (1 - alpha_dim) * new_cos  # cos EMA


class _Track:
    """A single active track.

    [한글] 활성(active) track 1개를 표현하는 컨테이너.
    - tid: 이 track의 고유 정수 ID (출력으로 나가는 추적 ID)
    - kf:  위 _KF3DState (모션/박스 상태)
    - embed: appearance 임베딩(L2 정규화 유지) — 코사인 유사도 계산에 사용
    - score/age/miss_count/history: 신뢰도/생존 프레임 수/연속 미관측 수/(x,y) 궤적
    __slots__로 인스턴스 속성을 고정해 메모리/속도를 최적화한다.
    """
    __slots__ = ['tid', 'kf', 'cls_id', 'embed', 'score', 'age', 'miss_count', 'history']

    def __init__(self, tid: int, box: np.ndarray, embed: np.ndarray, cls_id: int, score: float):
        self.tid = tid                       # 이 track에 부여된 고유 ID
        self.kf = _KF3DState(box)            # 모션/박스 상태(칼만필터) 초기화
        self.cls_id = int(cls_id)            # 클래스 id (클래스 일관성 매칭에 사용)
        self.embed = embed.astype(np.float64).copy()  # appearance 임베딩 [D] 복사
        # L2 normalize
        n = np.linalg.norm(self.embed) + 1e-8  # 임베딩 노름
        self.embed = self.embed / n            # 코사인 유사도 위해 L2 정규화
        self.score = float(score)            # 최신 검출 신뢰도
        self.age = 1                         # 생존(갱신된) 프레임 수
        self.miss_count = 0                  # 연속으로 매칭 실패한 프레임 수
        self.history: List[Tuple[float, float]] = [(float(box[0]), float(box[1]))]  # (x,y) 궤적 기록

    def predict(self) -> None:
        # 이 track의 KF 예측 단계 위임 (다음 프레임 위치 추정)
        self.kf.predict()

    def update(self, box: np.ndarray, embed: np.ndarray, score: float, embed_momentum: float = 0.9) -> None:
        # ===== 매칭 성공 시 track 갱신 =====
        self.kf.update(box)                      # KF 보정(검출 박스로 위치/속도 교정)
        e = embed.astype(np.float64)             # 새 검출 임베딩 [D]
        e = e / (np.linalg.norm(e) + 1e-8)       # L2 정규화
        # 임베딩 momentum 갱신: 기존 90% + 신규 10% (외형 임베딩을 천천히 적응시켜 jitter 완화)
        self.embed = embed_momentum * self.embed + (1 - embed_momentum) * e
        self.embed = self.embed / (np.linalg.norm(self.embed) + 1e-8)  # 블렌드 후 다시 정규화
        self.score = float(score)                # 신뢰도 갱신
        self.age += 1                            # 생존 프레임 수 +1
        self.miss_count = 0                      # 매칭됐으므로 미관측 카운트 리셋
        self.history.append((float(self.kf.pos[0]), float(self.kf.pos[1])))  # 보정된 (x,y)를 궤적에 추가

    def to_array(self) -> np.ndarray:
        """Pack current state back to [10] box (cx, cy, cz, w, l, h, sin, cos, vx, vy).

        [한글] track의 현재 상태를 다시 [10] 박스 형태로 직렬화한다.
        (KF 위치/속도 + EMA로 유지한 z/크기/yaw 를 합쳐서 반환)
        """
        return np.array([
            self.kf.pos[0], self.kf.pos[1], self.kf.z,        # cx, cy, cz
            self.kf.wlh[0], self.kf.wlh[1], self.kf.wlh[2],   # w, l, h
            self.kf.sin, self.kf.cos,                          # yaw 의 sin, cos
            self.kf.vel[0], self.kf.vel[1],                    # vx, vy
        ], dtype=np.float64)


class CC3DTPPTracker:
    """CC-3DT++ -style 3-way affinity tracker with KF3D motion model.

    Args:
        match_score_thr: minimum total affinity to accept a match.
        init_score_thr: minimum detection score to start a new track.
        obj_score_thr: minimum detection score to participate in matching.
        max_miss: drop track after N consecutive missed frames.
        w_appearance: weight for embedding cosine similarity.
        w_location:   weight for centroid distance affinity.
        w_velocity:   weight for velocity affinity (CR3DT additional).
        r_loc:        spatial decay radius for location affinity (meters).
        r_vel:        velocity decay radius (m/s).
        with_cats:    require class consistency for matches.
        embed_momentum: EMA for embedding update.
        dt:           frame interval (seconds, for KF predict).

    [한글] 평가에서 기본으로 쓰는 트래커. CC-3DT++ 스타일의 3-way affinity 매칭 +
    KF3D 모션 모델을 결합한다. PredictionCollector가 프레임마다 update()를 부르고,
    scene 경계에서 reset()을 부른다.

    [임계값/가중치 의미]
      - match_score_thr : 이 affinity 미만이면 매칭 거부(연결 안 함)
      - init_score_thr  : 이 점수 이상인 미매칭 검출만 새 track으로 생성
      - obj_score_thr   : 이 점수 이상인 검출만 매칭 후보로 사용
      - max_miss        : 연속 N프레임 미관측 시 track 삭제
      - w_app/w_loc/w_vel: 외형/위치/속도 affinity 가중치 (합=1 로 정규화)
      - r_loc/r_vel     : 위치/속도 affinity의 지수 감쇠 반경
      - with_cats       : True면 같은 클래스끼리만 매칭 허용
    """

    def __init__(
        self,
        match_score_thr: float = 0.3,
        init_score_thr: float = 0.4,
        obj_score_thr: float = 0.1,
        max_miss: int = 5,
        w_appearance: float = 0.4,
        w_location: float = 0.3,
        w_velocity: float = 0.3,
        r_loc: float = 4.0,
        r_vel: float = 3.0,
        with_cats: bool = True,
        embed_momentum: float = 0.9,
        dt: float = 0.5,
    ):
        self.match_score_thr = match_score_thr  # 매칭 수락 최소 affinity
        self.init_score_thr = init_score_thr    # 신규 track 생성 최소 점수
        self.obj_score_thr = obj_score_thr      # 매칭 참여 최소 점수
        self.max_miss = max_miss                # 최대 연속 미관측 허용 프레임 수
        # Normalize weights so they sum to 1
        # 세 가중치 합이 1이 되도록 정규화(분모 0 방지로 1e-6 하한)
        s = max(w_appearance + w_location + w_velocity, 1e-6)
        self.w_app = w_appearance / s  # 외형(appearance) affinity 가중치
        self.w_loc = w_location / s    # 위치(location) affinity 가중치
        self.w_vel = w_velocity / s    # 속도(velocity) affinity 가중치
        self.r_loc = r_loc             # 위치 affinity 감쇠 반경 (m)
        self.r_vel = r_vel             # 속도 affinity 감쇠 반경 (m/s)
        self.with_cats = with_cats     # 클래스 일관성 강제 여부
        self.embed_momentum = embed_momentum  # 임베딩 EMA momentum
        self.dt = dt                   # KF predict용 프레임 간격(초)

        self.tracks: List[_Track] = []  # 현재 활성 track 리스트
        self._next_id: int = 0          # 다음에 부여할 track ID (단조 증가)

    def reset(self) -> None:
        # ===== scene(시퀀스) 경계에서 호출: 모든 track 상태와 ID 카운터 초기화 =====
        self.tracks = []     # 활성 track 전부 제거
        self._next_id = 0    # ID 카운터를 0으로 되돌림 (새 시퀀스는 새 ID 공간)

    def get_track_history(self) -> Dict[int, List[Tuple[float, float]]]:
        """Map active track id -> its (x, y) center history (tracker/global frame).

        Used by the visualization path to draw per-track trails. Note the
        coordinates are in the tracker's association frame (global), so they are
        clipped against the local BEV range by the renderer.

        [한글] 활성 track별 (x,y) 궤적을 dict로 반환 (시각화에서 trail 그릴 때 사용).
        좌표는 트래커의 연관 좌표계(글로벌)이므로 렌더러가 로컬 BEV 범위로 클립한다.
        """
        return {t.tid: list(t.history) for t in self.tracks}  # {tid: [(x,y), ...]}

    def update(
        self,
        boxes_3d: np.ndarray,    # [N, 10] (cx, cy, cz, w, l, h, sin, cos, vx, vy) or [N, 9] (legacy yaw)
        scores: np.ndarray,       # [N]
        embeds: np.ndarray,       # [N, D] (per-detection L2-normed)
        labels: Optional[np.ndarray] = None,  # [N] class ids (optional but recommended)
    ) -> np.ndarray:
        """Assign track IDs to current frame detections. Returns [N] int array, -1 = no track.

        [한글] 이번 프레임 검출들에 track ID를 할당해 [N] 정수 배열로 반환한다(-1=ID 없음).
        전체 흐름: (0)입력 정규화 → (1)KF predict → (2)점수 필터 →
                   (3)3-way affinity + 헝가리안 매칭 → (4)미매칭 검출로 신규 track 생성 →
                   (5)오래 미관측된 track 삭제.
        """
        N = len(boxes_3d) if boxes_3d is not None else 0  # 이번 프레임 검출 개수
        if labels is None:
            labels = np.zeros((N,), dtype=np.int64)  # 라벨 없으면 전부 클래스 0으로 (단일 클래스 취급)
        # Coerce 9-d (yaw) to 10-d (sin, cos)
        # legacy 9-d 박스(yaw 단일값)를 10-d(sin,cos 분해)로 변환해 내부 표현을 통일
        if N > 0 and boxes_3d.shape[1] == 9:
            yaw = boxes_3d[:, 6]          # yaw 라디안 [N]
            sin_y = np.sin(yaw)           # sin 성분 [N]
            cos_y = np.cos(yaw)           # cos 성분 [N]
            boxes_10 = np.zeros((N, 10), dtype=boxes_3d.dtype)  # 변환 결과 버퍼 [N,10]
            boxes_10[:, :6] = boxes_3d[:, :6]   # cx,cy,cz,w,l,h 복사
            boxes_10[:, 6] = sin_y              # 인덱스 6 = sin
            boxes_10[:, 7] = cos_y              # 인덱스 7 = cos
            boxes_10[:, 8:] = boxes_3d[:, 7:9]  # vx,vy 복사
            boxes_3d = boxes_10                 # 이후 로직은 모두 10-d 가정

        # 1) KF predict all tracks
        # (1) 모든 활성 track을 한 스텝 예측해서 이번 프레임 예상 위치로 이동시킨다
        for t in self.tracks:
            t.predict()

        # 2) Filter detections by score threshold (for matching)
        # (2) obj_score_thr 이상인 검출만 매칭 후보로 선별
        det_inds_all = np.arange(N)  # 전체 검출 인덱스 [0..N-1]
        if N > 0:
            valid_match = scores >= self.obj_score_thr  # 매칭 참여 가능 마스크 [N] bool
        else:
            valid_match = np.zeros((0,), dtype=bool)    # 검출 0개면 빈 마스크
        det_inds_match = det_inds_all[valid_match]      # 매칭 후보 검출의 전역 인덱스
        out_tids = np.full((N,), -1, dtype=np.int64)    # 출력 ID 배열 [N], 기본 -1(미할당)

        # 3) Hungarian on (track, det) affinity
        # (3) track과 검출 후보가 둘 다 존재할 때만 affinity 매칭 수행
        if len(self.tracks) > 0 and len(det_inds_match) > 0:
            T = len(self.tracks)          # track 개수
            M = len(det_inds_match)       # 매칭 후보 검출 개수
            # 각 track의 상태를 행렬로 적층 (벡터화 affinity 계산용)
            track_pos = np.stack([t.kf.pos for t in self.tracks], axis=0)   # [T, 2] track 위치
            track_vel = np.stack([t.kf.vel for t in self.tracks], axis=0)   # [T, 2] track 속도
            track_emb = np.stack([t.embed for t in self.tracks], axis=0)    # [T, D] track 임베딩
            track_cls = np.array([t.cls_id for t in self.tracks])           # [T] track 클래스

            det_pos = boxes_3d[det_inds_match, :2]                          # [M, 2] 검출 위치(x,y)
            det_vel = boxes_3d[det_inds_match, 8:10]                        # [M, 2] 검출 속도(vx,vy)
            det_emb = embeds[det_inds_match]                                # [M, D] 검출 임베딩
            det_cls = labels[det_inds_match]                               # [M] 검출 클래스

            # Appearance: cosine (assumes both L2-normed)
            # (a) 외형 affinity: 임베딩 코사인 유사도. 안전하게 다시 L2 정규화 후 내적
            det_emb_n = det_emb / (np.linalg.norm(det_emb, axis=-1, keepdims=True) + 1e-8)    # [M,D]
            track_emb_n = track_emb / (np.linalg.norm(track_emb, axis=-1, keepdims=True) + 1e-8)  # [T,D]
            cos_sim = (track_emb_n @ det_emb_n.T).clip(-1.0, 1.0)            # [T, M] in [-1, 1]
            app_aff = 0.5 * (cos_sim + 1.0)                                  # [T, M] in [0, 1] 로 매핑

            # Location: exp(-d / r_loc)
            # (b) 위치 affinity: BEV 거리 d를 exp 커널로 변환 (가까울수록 1)
            d_pos = np.linalg.norm(track_pos[:, None, :] - det_pos[None, :, :], axis=-1)  # [T, M] 거리
            loc_aff = np.exp(-d_pos / self.r_loc)                                          # [T, M] in (0,1]

            # Velocity: exp(-|v_diff| / r_vel)
            # (c) 속도 affinity: 속도 벡터 차이를 exp 커널로 변환 (비슷할수록 1)
            d_vel = np.linalg.norm(track_vel[:, None, :] - det_vel[None, :, :], axis=-1)  # [T, M] 속도차
            vel_aff = np.exp(-d_vel / self.r_vel)                                          # [T, M] in (0,1]

            # 3-way 가중 합산 -> 최종 affinity 행렬 [T, M]
            aff = self.w_app * app_aff + self.w_loc * loc_aff + self.w_vel * vel_aff       # [T, M]

            # Class consistency mask (optional)
            # (선택) 클래스 일관성: 다른 클래스 쌍의 affinity를 0으로 눌러 매칭 차단
            if self.with_cats:
                cls_mask = (track_cls[:, None] == det_cls[None, :])  # [T, M] bool (같은 클래스 True)
                aff = aff * cls_mask.astype(aff.dtype)               # 다른 클래스면 affinity 0

            # Hungarian (maximize affinity → minimize -affinity)
            # 헝가리안은 비용 최소화이므로 -aff를 넣어 affinity 최대화를 푼다
            row_ind, col_ind = linear_sum_assignment(-aff)  # row_ind: track 인덱스, col_ind: det 후보 인덱스
            matched_tracks = set()  # 이번 프레임 매칭된 track 인덱스 집합
            matched_dets = set()    # 이번 프레임 매칭된 검출 전역 인덱스 집합
            for r, c in zip(row_ind, col_ind):
                if aff[r, c] < self.match_score_thr:
                    continue  # affinity가 임계값 미만이면 이 쌍은 매칭으로 인정하지 않음
                det_global_idx = int(det_inds_match[c])  # 후보 인덱스 c -> 전역 검출 인덱스로 환원
                t = self.tracks[r]                       # 매칭된 track 객체
                # 매칭된 검출로 track 갱신(KF 보정 + 임베딩 momentum 갱신)
                t.update(boxes_3d[det_global_idx], embeds[det_global_idx], scores[det_global_idx],
                         embed_momentum=self.embed_momentum)
                out_tids[det_global_idx] = t.tid          # 이 검출의 출력 ID = track의 ID
                matched_tracks.add(r)                     # track r 매칭됨 기록
                matched_dets.add(int(det_global_idx))     # 검출 매칭됨 기록

            # Mark unmatched tracks as missed
            # 이번 프레임에 매칭 못 된 track은 miss_count 증가(곧 삭제 후보)
            for ti, t in enumerate(self.tracks):
                if ti not in matched_tracks:
                    t.miss_count += 1
        else:
            matched_dets = set()  # track이나 검출이 없으면 매칭 단계 생략

        # 4) Create new tracks from high-score unmatched detections
        # (4) 매칭 안 됐고 점수가 충분히 높은 검출은 새 track으로 생성
        for i in range(N):
            if out_tids[i] != -1:
                continue  # 이미 ID 할당된(매칭된) 검출은 건너뜀
            if scores[i] < self.init_score_thr:
                continue  # 신규 생성 임계값 미만이면 새 track 안 만듦
            # 새 track 생성: 다음 ID 부여, 박스/임베딩/클래스/점수로 초기화
            t = _Track(self._next_id, boxes_3d[i], embeds[i], int(labels[i]), float(scores[i]))
            self.tracks.append(t)          # 활성 track 목록에 추가
            out_tids[i] = self._next_id    # 이 검출의 출력 ID 부여
            self._next_id += 1             # 다음 ID 준비

        # 5) Drop tracks that missed too many frames
        # (5) 연속 미관측이 max_miss를 초과한 track은 소멸 처리(목록에서 제거)
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]

        return out_tids  # [N] track ID 배열 반환 (-1 = ID 미할당)
