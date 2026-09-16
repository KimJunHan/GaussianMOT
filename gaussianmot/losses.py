# --- 기본 라이브러리 import (로깅, 수치연산, 파이토치 텐서/신경망/함수형 API) ---
import logging                     # 경고/디버그 로그 출력용
import numpy as np                 # heatmap gaussian 계산 등 CPU 수치연산용
import torch                       # 텐서 연산 핵심 라이브러리
import torch.nn as nn              # nn.Module, nn.Parameter 등 신경망 빌딩블록
import torch.nn.functional as F    # interpolate 등 함수형 연산 (리사이즈 등)

logger = logging.getLogger(__name__)  # 모듈 단위 로거 (현재 파일에서는 직접 사용 안 함)


# ============================================================================
# CenterPoint 스타일 supervision을 위한 Gaussian heatmap 유틸 모음.
# TransFusion / CenterPoint (mmdet3d) 구현을 옮겨온(inline) 것.
# (원래 gaussianmot/utils/heatmap.py 에 있던 코드를 이 파일로 인라인함)
# ============================================================================
def gaussian_2d(shape, sigma: float = 1.0):
    # shape=(높이, 너비)인 2D 가우시안 커널을 만든다. 중심이 1.0, 멀어질수록 0으로 감쇠.
    m, n = [(ss - 1.0) / 2.0 for ss in shape]   # 커널 중심까지의 반경(세로 m, 가로 n). shape는 보통 홀수.
    y, x = np.ogrid[-m : m + 1, -n : n + 1]     # 중심을 원점(0,0)으로 하는 좌표 그리드 생성 (y는 열벡터, x는 행벡터)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))  # 2D 가우시안 식: 중심에서의 거리^2에 비례해 지수적으로 감쇠
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # 수치적으로 무시 가능한(아주 작은) 값은 0으로 잘라 노이즈 제거
    return h                                     # [높이, 너비] 가우시안 커널 반환 (중심=1.0)


def draw_heatmap_gaussian(heatmap: torch.Tensor, center, radius: int, k: float = 1.0):
    """`heatmap` 의 `center` 위치에 반경 `radius`짜리 2D 가우시안을 in-place로 그린다.

    `heatmap`: [H, W]. `center`: (x, y) 정수 좌표. GT 박스 중심마다 가우시안 봉우리를 찍어
    CenterPoint heatmap의 soft target을 만든다.
    """
    diameter = 2 * radius + 1                              # 가우시안 커널 한 변 길이 (반경*2+1, 홀수)
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6.0)  # 지름 크기의 가우시안 커널 생성 (관례상 sigma=지름/6)

    x, y = int(center[0]), int(center[1])                 # 중심 좌표를 정수 픽셀 인덱스로 변환
    height, width = heatmap.shape[0:2]                    # heatmap 전체 크기 [H, W]

    # 중심에서 커널을 붙일 때, heatmap 경계를 넘지 않도록 좌/우/상/하로 잘라낼 범위를 계산
    left, right = min(x, radius), min(width - x, radius + 1)   # 가로 방향으로 실제 사용 가능한 좌/우 폭
    top, bottom = min(y, radius), min(height - y, radius + 1)  # 세로 방향으로 실제 사용 가능한 상/하 폭

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]   # heatmap에서 가우시안을 덮어쓸 부분영역 (뷰)
    masked_gaussian = torch.from_numpy(                                    # 커널에서 그에 대응하는 부분영역만 잘라 텐서로
        gaussian[radius - top : radius + bottom, radius - left : radius + right]
    ).to(heatmap.device, torch.float32)                                   # heatmap과 동일 device/float32로 맞춤
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # 잘린 영역이 비어있지 않을 때만 합성
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)  # element-wise max: 겹치는 객체끼리 더 큰 봉우리 유지
    return heatmap                                        # (in-place 수정된) heatmap 반환


def gaussian_radius(det_size, min_overlap: float = 0.5) -> float:
    """객체 크기(height, width)에 대해 가우시안 반경을 계산한다.

    의미: 예측 박스가 GT 중심에서 반경 r만큼 어긋나도 IoU(겹침)가 min_overlap 이상 유지되도록
    보장하는 r. CornerNet/CenterNet 유도식 그대로. 세 가지 어긋남 경우(둘 다 안쪽 / 둘 다 바깥쪽 /
    한쪽씩)에 대해 r을 각각 풀고, 가장 보수적인(최소) 값을 쓴다.
    """
    height, width = det_size                  # 객체의 (세로=length, 가로=width) 셀 단위 크기
    if isinstance(height, torch.Tensor):      # 텐서로 들어오면 파이썬 스칼라로 변환 (np.sqrt에 넣기 위함)
        height = height.item()
        width = width.item()

    # 경우 1: 예측 박스가 GT 안쪽으로 r만큼 줄어든 경우 → 이차방정식 a1 r^2 - b1 r + c1 = 0의 근
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = float(np.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0)))  # 판별식(음수 방지 위해 0과 max)
    r1 = (b1 + sq1) / 2

    # 경우 2: 예측 박스가 GT 바깥쪽으로 r만큼 커진 경우 → 다른 계수의 이차방정식 근
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = float(np.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0)))
    r2 = (b2 + sq2) / (2 * a2)

    # 경우 3: 한 변은 안쪽, 한 변은 바깥쪽으로 r만큼 어긋난 경우 → 또 다른 이차방정식 근
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = float(np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0)))
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)                     # 세 경우 중 최소 반경 (가장 엄격한 조건)을 채택


class MultipleLoss(torch.nn.ModuleDict):
    """
    여러 개의 loss 함수와 각각의 weight를 하나로 묶어 가중합을 계산하는 컨테이너.

    사용 예:
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)

    즉 dict에 'xxx' = loss모듈, 'xxx_weight' = float 형태로 짝지어 넣으면,
    forward에서 각 loss를 호출해 weight를 곱하고 전부 더한 총손실을 돌려준다.
    """
    def __init__(self, modules_or_weights):
        # 실제 학습 config(configs/train.yaml)는 양수 고정 가중치만 사용한다:
        #   set_weight=1.0(CenterPointLoss), track_weight=0.5(TrackLoss), doppler_weight=0.5(DopplerConsistencyLoss).
        #   → weight=-1(uncertainty weighting) 경로를 쓰는 항이 하나도 없어 learnable_weights는 항상 비어 있음(dead).
        modules = dict()             # 실제 loss 함수(모듈)들을 담을 dict
        weights = dict()             # 각 loss의 가중치(float)를 담을 dict
        learnable_weights = dict()   # uncertainty 기반 '학습 가능한' 가중치를 담을 dict 실제 config엔 weight=-1 항이 없어 항상 빈 dict

        # --- 1단계: weight(float)만 먼저 파싱 ---
        for key, v in modules_or_weights.items():
            if isinstance(v, float):              # 값이 float이면 이건 weight 항목
                k = key.replace('_weight', '')    # 'bce_weight' → 'bce' (대응하는 loss 이름)
                if v == -1:                        # weight=-1 은 "학습 가능한 가중치로 처리하라"는 특수 신호
                    weights[k] =  0.5 if k not in ['visible', 'ped'] else 10.0  # 초기 기본 가중치(특정 키는 10배 강조)
                    learnable_weights[k] = nn.Parameter(torch.tensor(0.0), requires_grad=True)  # 학습되는 log-uncertainty 파라미터(초기 0)
                else:
                    weights[k] = v                 # 일반적인 고정 가중치

        # --- 2단계: loss 함수(모듈)들을 파싱 ---
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):           # float이 아니면 loss 모듈 자체
                modules[key] = v

        super().__init__(modules)                  # ModuleDict 초기화: loss 모듈들을 등록

        self._weights = weights                                       # 고정/초기 가중치 dict 저장
        self.learnable_weights = torch.nn.ParameterDict(learnable_weights)  # 학습 가능한 가중치를 ParameterDict로 등록

    def forward(self, pred, batch):
        outputs = dict()   # 각 loss 이름 → loss 값
        weights = dict()   # 각 loss 이름 → 실제 적용된 가중치 (로깅/모니터링용)

        # --- 각 loss 모듈을 순회하며 loss 값 계산 ---
        for k, v in self.items():

            if k =='learnable_weights':            # ParameterDict 자체는 호출 대상이 아니므로 건너뜀
                continue
            elif k != 'Set':                       # 일반 loss: (pred, batch)를 넣어 스칼라 loss 하나를 얻음
                # 실제 config의 loss 키는 'set'/'track'/'doppler'(소문자)라 모두 이 분기로 흐른다.
                # 대문자 'Set'(DETR set-prediction) 분기는 현재 학습에선 안 탐.
                outputs[k] = v(pred, batch)
            else:                                  # 'Set' 키는 DETR-style set-prediction loss로 dict를 반환
                if 'pred_logits' not in pred:      # set-prediction 예측이 없으면 스킵
                    continue
                out = v(pred, batch)
                for k2, v2 in out.items():         # 반환된 여러 하위 loss를 outputs에 펼쳐 담음
                    outputs[k2] = v2
        # outputs = {k: v(pred, batch) for k, v in self.items()}
        loss = []                                  # 가중치 적용된 개별 loss를 모을 리스트
        for k, o in outputs.items():
            loss_weight = self._weights[k]         # 해당 loss의 기본 가중치 실제: set=1.0, track=0.5, doppler=0.5
            # 실제 config엔 learnable_weights가 비어 있어 아래 if 분기는 항상 거짓 → 항상 고정 가중치 경로
            if k in self.learnable_weights:        # 학습 가능한 가중치(uncertainty weighting, Kendall 2018)인 경우
                # 1/exp(s) 형태로 분산이 클수록 가중치를 자동으로 낮춘다 (s는 학습되는 log-분산)
                loss_weight = (1 / torch.exp(self.learnable_weights[k])) * loss_weight
                weights[k] = loss_weight           # 실제 적용된 가중치 기록
                uncertainty = self.learnable_weights[k] * 0.5  # +0.5*s 정규화 항: weight가 0으로 폭주하는 것을 막는 패널티
            else:
                uncertainty = 0.0                  # 고정 가중치면 정규화 항 없음
            single_loss = loss_weight * o + uncertainty  # 최종 개별 loss = 가중치*loss + (uncertainty 패널티)
            outputs[k] = single_loss               # outputs를 가중치 적용 후 값으로 덮어씀
            loss.append(single_loss)

        return sum(loss), outputs, weights         # (총손실, 항목별 loss dict, 적용 가중치 dict) 반환

# ========================================================================
# Tracking Loss
# ========================================================================

class TrackLoss(nn.Module):
    """
    Tracking loss - cross-frame supervised contrastive (SupCon).

    [의미] BEV 위 각 픽셀이 추적용 임베딩 벡터를 갖는다.
      - positive pair: 같은 instance_id(같은 객체)에 속한 픽셀들 → 임베딩을 서로 가깝게
      - negative pair: 다른 instance에 속한 픽셀들 → 임베딩을 서로 멀게
    같은 객체 픽셀이 임베딩 공간에서 뭉치면, 추론 때 cosine similarity로 프레임 간
    객체 매칭(association=추적)이 쉬워진다. instance_id_map은 labels.py에서 생성.
    임베딩은 채널 방향 L2 정규화되어 있어 두 임베딩의 내적 = cosine similarity.

    [③ projection / association 분리] 이 loss는 contrastive용 projection
    `{key}_track_proj`(있으면)으로 계산하고, 추론/oracle의 association은 `{key}_track_embed`을
    쓴다(track_head.py). loss-공간과 매칭-공간을 분리(SimCLR식).
    [④ cross-frame] 이전 프레임 임베딩(prev)을 같은 SupCon pool에 합쳐 시간축 positive를
    형성한다. module.py가 cross_frame_grad=true면 prev에도 grad를 흘려보냄(양방향).

    [실제 shape 흐름] (학습에서만 계산, B=1)
      prediction['{key}_track_proj'] : [B=1, D=64, 200, 200]  (per-pixel 임베딩맵)
      batch['{key}_instance_id']     : [B=1, 200, 200]        (per-pixel GT 객체 id)
        → _sample_pixels: instance마다 ≤ max_pixels(=16) 픽셀 샘플 → feats [k≤16, 64]
        → (현재 + prev) 풀 합침 → feats [N, 64], labels [N]   (N = 총 샘플 픽셀 수)
        → sim = feats @ featsᵀ / τ(=0.15) → [N, N] (모든 픽셀쌍 cosine/τ)
        → SupCon(log-softmax, positive=같은 id) → 스칼라
      반환: scalar loss (낮을수록 같은 객체 픽셀끼리 더 가까움). 값이 정체하면 추적 표현이
            instance를 못 가르고 있다는 신호(현재 디버깅 대상).
    """
    def __init__(
        self,
        temperature: float = 0.1,           # contrastive 온도: 작을수록 유사도 차이를 더 날카롭게 구분(어려운 negative 강조)
                                            # 실제 config(train.yaml)=0.15 (track 수렴 개선 ②: 0.07→0.15, 기본값 0.1과 다름)
        max_pixels_per_instance: int = 8,   # 객체 하나당 샘플링할 최대 픽셀 수(메모리/계산량 제한 + 클래스 불균형 완화)
                                            # 실제 config(train.yaml)=16 (개선 B: 8→16, 기본값 8과 다름)
        reg_weight: float = 1e-4,           # 임베딩 L2 정규화 항 가중치(collapse 방지용 약한 안전망)
        key: str = 'vehicle',               # 어떤 태스크 키의 임베딩/instance를 쓸지 (예: 'vehicle')
    ):
        super().__init__()
        self.temperature = temperature
        self.max_pixels_per_instance = max_pixels_per_instance
        self.reg_weight = reg_weight
        self.key = key

    def _sample_pixels(
        self,
        embed_b: torch.Tensor,        # [D, H, W] 한 프레임의 픽셀별 임베딩 실제: [64, 200, 200]
        inst_b: torch.Tensor,         # [H, W]    한 프레임의 픽셀별 instance_id 맵 실제: [200, 200]
        device: torch.device,
    ):
        """각 instance_id마다 픽셀을 최대 max_pixels_per_instance개씩 무작위 샘플링한다."""
        feats_list, labels_list = [], []          # 샘플된 임베딩들과 그에 대응하는 instance 라벨들
        ids = torch.unique(inst_b)                # 이 프레임에 존재하는 모든 instance_id
        ids = ids[ids >= 0]                        # 음수 id(배경/무효)는 제외
        for iid in ids.tolist():                  # 객체 id 하나씩 처리
            idx = (inst_b == iid).nonzero(as_tuple=False)  # 해당 객체에 속한 픽셀 좌표들 [n, 2]
            n = idx.shape[0]                       # 그 객체의 픽셀 개수
            if n == 0:
                continue
            k = min(n, self.max_pixels_per_instance)   # 너무 큰 객체도 최대 k개까지만 사용
            sel = idx[torch.randperm(n, device=device)[:k]]  # n개 중 무작위로 k개 픽셀 선택
            feats = embed_b[:, sel[:, 0], sel[:, 1]].T  # 선택 픽셀의 임베딩 추출 후 전치 → [k, D] (행=픽셀, 열=임베딩차원) 실제: [k≤16, 64]
            feats_list.append(feats)
            labels_list.append(torch.full((k,), iid, dtype=torch.long, device=device))  # 이 k개 픽셀의 라벨을 모두 iid로
        return feats_list, labels_list             # (객체별 임베딩 리스트, 객체별 라벨 리스트)

    def _supcon_one_frame(
        self,
        embed_b: torch.Tensor,
        inst_b: torch.Tensor,
        prev_embed_b: torch.Tensor = None,   # [D, H, W] 또는 None (이전 프레임 임베딩) 실제: [64, 200, 200]
        prev_inst_b: torch.Tensor = None,    # [H, W] 또는 None    (이전 프레임 instance 맵) 실제: [200, 200]
    ) -> torch.Tensor:
        """
        embed_b/inst_b: 현재 프레임 (이미 L2-normalized)
        prev_*: 옵션. 주어지면 SupCon pool에 함께 넣어 '시간축(cross-frame) positive'를 형성.

        Phase 3b 핵심: instance_id는 scene 전역에서 일관되게 부여(_instance_token_to_id)되므로,
        현재 프레임 픽셀과 이전 프레임 픽셀을 하나의 통합 pool로 묶기만 해도
        같은 객체(같은 id)는 두 프레임에 걸쳐 자동으로 positive pair가 된다.
        즉 별도 매칭 코드 없이 cross-frame 일관성 학습이 일어난다.
        """
        device = embed_b.device
        feats_list, labels_list = self._sample_pixels(embed_b, inst_b, device)  # 현재 프레임 픽셀 샘플링

        if prev_embed_b is not None and prev_inst_b is not None:
            prev_feats, prev_labels = self._sample_pixels(prev_embed_b, prev_inst_b, device)  # 이전 프레임도 샘플링
            feats_list.extend(prev_feats)     # 현재+이전 임베딩을 같은 pool에 합침
            labels_list.extend(prev_labels)   # 라벨도 합침 → 같은 id면 cross-frame positive로 묶임

        if len(feats_list) < 2:               # 샘플된 객체 그룹이 2개 미만이면 대비할 게 없으므로 스킵
            return None

        # contrast가 의미 있으려면 서로 다른 instance가 최소 2종류 이상 있어야 함(negative가 존재해야 함)
        all_labels = torch.cat(labels_list)
        if torch.unique(all_labels).numel() < 2:
            return None

        feats = torch.cat(feats_list, dim=0)   # 모든 샘플 임베딩을 한 행렬로 [N, D] (N=총 픽셀 수) 실제: [N, 64]
        labels = torch.cat(labels_list, dim=0)  # 대응 라벨 [N]

        # 코사인 유사도 행렬 (임베딩이 이미 정규화되어 있어 내적 = cosine). temperature로 나눠 logit 스케일 조정
        sim = feats @ feats.T / self.temperature  # [N, D] @ [D, N] -> [N, N], 모든 픽셀 쌍의 유사도 실제: [N,64]@[64,N]->[N,N]
        # 자기 자신과의 유사도(대각선)는 분자/분모 양쪽에서 제외해야 함
        self_mask = torch.eye(sim.shape[0], dtype=torch.bool, device=device)  # 대각선 마스크 [N, N]
        sim.masked_fill_(self_mask, float('-inf'))  # 대각선을 -inf로 → softmax/logsumexp에서 0 기여

        # 각 anchor(행)에 대해 log-softmax: log p(j|i) = sim_ij - log(sum_k exp(sim_ik))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [N, N], 분모는 자기 제외 모든 픽셀(=negative 포함)
        # positive 마스크: 같은 라벨끼리 True, 단 자기 자신은 제외
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # [N, N]
        pos_count = pos_mask.sum(dim=1)        # 각 anchor의 positive 개수
        valid = pos_count > 0                  # positive가 하나도 없는 anchor는 SupCon 계산에서 제외
        if not valid.any():
            return None

        # 대각선의 -inf가 (~pos_mask 위치의) 0과 곱해지면 0*-inf=NaN이 되므로, 먼저 0으로 채워 안전하게 합산
        masked_log_prob = log_prob.masked_fill(~pos_mask, 0.0)  # positive가 아닌 위치는 0으로 → 합산에서 빠짐
        # SupCon loss: anchor별 positive들의 평균 log-prob (positive 개수로 정규화)
        mean_log_prob_pos = masked_log_prob.sum(dim=1)[valid] / pos_count[valid].float()
        return -mean_log_prob_pos.mean()       # 음의 평균 → positive가 가까울수록 loss 감소

    def forward(self, prediction, batch):
        # ③ projection head: contrastive는 projection(`_track_proj`)으로 계산(있으면 우선).
        #   없으면 기존 association 임베딩(`_track_embed`)으로 fallback.
        # 실제: key='vehicle' → proj_key='vehicle_track_proj'(③로 우선), 없으면 'vehicle_track_embed'로 fallback
        proj_key = f"{self.key}_track_proj"
        embed_key = proj_key if proj_key in prediction else f"{self.key}_track_embed"
        inst_key = f"{self.key}_instance_id"    # batch에서 instance_id 맵을 꺼낼 키

        if embed_key not in prediction:         # 모델이 track 임베딩을 안 내놓으면(헤드 비활성) loss 0
            device = next(iter(prediction.values())).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        embed = prediction[embed_key]      # [B, D, H, W] 배치별 픽셀 임베딩 실제: [B=1, 64, 200, 200]
        # L2 정규화 항 — 항상 약하게 들어가서 임베딩이 0으로 붕괴(collapse)하는 것을 막고, gradient 흐름을 보장하는 안전망
        reg = self.reg_weight * (embed ** 2).mean()

        if inst_key not in batch:          # GT instance 맵이 없으면 contrastive 학습 불가 → reg 항만 반환
            return reg

        inst = batch[inst_key]             # [B, H, W] GT instance_id 맵 실제: [B=1, 200, 200] (필요시 200으로 nearest resize)
        # BEV 해상도 매칭: track_embed가 다운샘플돼 있으면 instance_id 맵을 nearest로 같은 크기로 리사이즈
        B, _, H, W = embed.shape           # 임베딩 기준 해상도 추출
        if inst.shape[-2:] != (H, W):
            inst = F.interpolate(
                inst.unsqueeze(1).float(), size=(H, W), mode='nearest',  # 라벨이므로 보간 없이 nearest
            ).squeeze(1).long()            # [B,1,H,W]→[B,H,W], 다시 정수 id로

        # Phase 3b: 이전 프레임 정보가 batch에 들어와 있으면 SupCon pool에 추가하여 cross-frame positive 형성
        # ③ projection 일관성: 현재가 proj를 쓰면 prev도 proj(같은 공간)로 맞춘다.
        prev_proj_key = f"prev_{self.key}_track_proj"
        prev_embed_key = prev_proj_key if prev_proj_key in batch else f"prev_{self.key}_track_embed"
        prev_inst_key = f"prev_{self.key}_instance_id"
        prev_embed = batch.get(prev_embed_key)   # 이전 프레임 임베딩 (없으면 None)
        prev_inst = batch.get(prev_inst_key)     # 이전 프레임 instance 맵 (없으면 None)
        if prev_embed is not None and prev_inst is not None:
            if prev_inst.shape[-2:] != (H, W):   # 이전 프레임 instance 맵도 현재 해상도로 맞춤
                prev_inst = F.interpolate(
                    prev_inst.unsqueeze(1).float(), size=(H, W), mode='nearest',
                ).squeeze(1).long()
        else:
            prev_embed, prev_inst = None, None   # 둘 중 하나라도 없으면 cross-frame 비활성

        losses = []
        for b in range(B):                       # 배치의 각 샘플(프레임)에 대해 SupCon loss 계산
            l = self._supcon_one_frame(
                embed[b], inst[b],
                prev_embed_b=(prev_embed[b] if prev_embed is not None else None),  # 있으면 이전 프레임 임베딩 전달
                prev_inst_b=(prev_inst[b] if prev_inst is not None else None),
            )
            if l is not None:                    # 유효한(positive/negative가 충분한) 프레임만 수집
                losses.append(l)

        if not losses:                           # 모든 프레임이 무효였으면 reg 항만 반환
            return reg

        return torch.stack(losses).mean() + reg  # 프레임 평균 SupCon loss + 정규화 항


class TrackOffsetLoss(nn.Module):
    """[AMOTP 리팩토링 ①] CenterTrack식 temporal offset 회귀 loss.

    현재 프레임 각 GT 중심에서 **이전 프레임 같은 객체 중심까지의 변위 (dx,dy)[m]** 를 L1 회귀.
    - 타깃: prev 중심을 `A(bev_augm) @ ego_motion @ A^{-1}` 로 현재 aug-BEV 좌표계로 옮겨
      (prev_mapped - cur) 계산 (같은 instance_id끼리 매칭; prev에 없는 객체는 마스크 0).
    - **floor≈0 → decimal로 학습**되고(SupCon과 달리), 위치 기반이라 추론 association에 직접 사용
      → AMOTP 직결. temporal_warp로 trunk에 prev 컨텍스트가 있어 head가 예측 가능.
    - offset은 velocity×dt와 크기 유사(방향 반대: 과거를 가리킴). offset_scale로 정규화해 decimal.

    [실제 shape] pred `{key}_dense_track_offset` [B,2,H,W](m) / gt_boxes [N,11](…,inst_id=10)
      ego_motion [4,4] (prev lidar→cur lidar) / bev_augm [4,4] (aug 회전·스케일).
    """
    def __init__(self, bev_h: int = 200, bev_w: int = 200,
                 x_min: float = -50.0, x_max: float = 50.0,
                 y_min: float = -50.0, y_max: float = 50.0,
                 offset_scale: float = 3.0,   # 타깃 정규화 스케일[m] (전형적 프레임간 변위) → L1 decimal
                 key: str = "vehicle"):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.res_row = (x_max - x_min) / bev_h      # x→row 해상도 [m/cell]
        self.res_col = (y_max - y_min) / bev_w      # y→col 해상도 [m/cell]
        self.offset_scale = offset_scale
        self.key = key

    @staticmethod
    def _aslist(x, B):
        if x is None: return [None] * B
        if isinstance(x, (list, tuple)): return list(x)
        if torch.is_tensor(x) and x.dim() >= 3: return [x[i] for i in range(x.shape[0])]
        return [x]

    def forward(self, prediction: dict, batch: dict) -> torch.Tensor:
        pred = prediction.get(f"{self.key}_dense_track_offset")   # [B,2,H,W] (m)
        if pred is None:
            ref = next(iter(prediction.values())); return ref.sum() * 0.0
        B, _, H, W = pred.shape; device = pred.device
        cur_l  = self._aslist(batch.get(f"{self.key}_gt_boxes"), B)
        prev_l = self._aslist(batch.get(f"prev_{self.key}_gt_boxes"), B)
        ego_l  = self._aslist(batch.get("ego_motion"), B)
        A_l    = self._aslist(batch.get("bev_augm"), B)
        pA_l   = self._aslist(batch.get("prev_bev_augm"), B)   # 이전 프레임 aug (train은 independent라 다름!)
        tgt  = torch.zeros(B, 2, H, W, device=device)   # offset 타깃 [B,2,H,W] (m)
        mask = torch.zeros(B, 1, H, W, device=device)   # 현재·이전 둘 다 존재하는 객체 중심만 1
        for b in range(B):
            cur, prev, ego, A, pA = cur_l[b], prev_l[b], ego_l[b], A_l[b], pA_l[b]
            if cur is None or prev is None or ego is None or A is None: continue
            if pA is None: pA = A                    # prev_bev_augm 없으면 공유 aug로 폴백
            cur = torch.as_tensor(cur, dtype=torch.float32, device=device)
            prev = torch.as_tensor(prev, dtype=torch.float32, device=device)
            ego = torch.as_tensor(ego, dtype=torch.float32, device=device)
            A = torch.as_tensor(A, dtype=torch.float32, device=device)
            pA = torch.as_tensor(pA, dtype=torch.float32, device=device)
            if cur.numel() == 0 or prev.numel() == 0: continue
            # 올바른 변환: prev-aug 좌표 → 현재-aug 좌표 = A_cur @ ego_motion @ inv(A_prev).
            #   (검증: 정지객체 offset≈0.06m, 움직임 |offset|≈|v|×0.5s(=키프레임 dt), 방향 -v와 2°)
            try: M = A @ ego @ torch.inverse(pA)
            except Exception: continue
            pctr = {int(prev[j, 10].item()): prev[j, :3] for j in range(prev.shape[0])}  # prev instance→중심(aug)
            for i in range(cur.shape[0]):
                iid = int(cur[i, 10].item())
                if iid not in pctr: continue        # 이전 프레임에 없던 객체는 스킵
                cx, cy = float(cur[i, 0]), float(cur[i, 1])
                col = int(self.bev_w / 2.0 - cy / self.res_col)   # CenterPoint와 동일 셀 규칙
                row = int(self.bev_h / 2.0 - cx / self.res_row)
                if not (0 <= col < W and 0 <= row < H): continue
                p = pctr[iid]
                pm = M @ torch.tensor([p[0], p[1], p[2], 1.0], device=device)   # prev 중심 → 현재 좌표
                tgt[b, 0, row, col] = pm[0] - cx     # dx (과거 위치 - 현재)
                tgt[b, 1, row, col] = pm[1] - cy     # dy
                mask[b, 0, row, col] = 1.0
        denom = mask.sum().clamp(min=1.0)
        # 점유 셀에서만 L1, /2(두 채널) /offset_scale(정규화 → decimal)
        return ((pred - tgt).abs() * mask).sum() / denom / 2.0 / self.offset_scale


# ========================================================================
# Phase 3c: DETR-style set-prediction losses (one-to-one matched)
# (CenterPoint 검출 supervision: heatmap focal loss용 보조 함수와 dense loss)
# ========================================================================

def _gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, gamma: float = 4.0, eps: float = 1e-12) -> torch.Tensor:
    """가우시안 타겟용 penalty-reduced focal loss (CornerNet/CenterPoint 스타일).

    중심 픽셀(target==1)은 일반 focal loss로, 그 주변(0<target<1)은 가우시안 거리에 따라
    패널티를 줄여(penalty-reduced) negative로 취급한다. 정확한 중심만 강하게, 근처는 부드럽게.

    Args:
        pred: [B, C, H, W] sigmoid를 거친 heatmap (0~1 확률).
        target: 같은 shape, GT 중심에서 1이고 주변으로 갈수록 줄어드는 가우시안 타겟 [0,1].
    """
    pred = pred.clamp(min=eps, max=1 - eps)        # log(0)/log(음수) 방지를 위해 확률을 (eps, 1-eps)로 클램프
    pos_mask = target.eq(1).float()                # 타겟이 정확히 1인 위치 = positive(객체 중심)
    neg_mask = target.lt(1).float()                # 타겟이 1보다 작은 모든 위치 = negative(주변+배경)
    neg_weights = (1 - target).pow(gamma)          # 중심에 가까운 negative일수록(=target 큼) 패널티를 줄이는 가중(거리 기반)
    # positive loss: 확신 못 할수록(pred 작음) 큰 페널티, (1-pred)^alpha로 쉬운 예제 down-weight
    pos_loss = -(pred.log()) * (1 - pred).pow(alpha) * pos_mask
    # negative loss: 배경인데 높게 예측할수록(pred 큼) 페널티, pred^alpha + 거리가중 neg_weights 적용
    neg_loss = -((1 - pred).log()) * pred.pow(alpha) * neg_weights * neg_mask
    num_pos = pos_mask.sum().clamp(min=1.0)        # positive 개수로 정규화(최소 1로 0나눗셈 방지)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos  # 양/음 loss 합을 positive 수로 평균


def _quality_focal_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    """Quality/Generalized Focal Loss (GFL의 분류 항, QFL).

    CornerNet focal이 pos(=1)/neg(<1)를 나눠 다루는 것과 달리, QFL은 **연속 soft 타겟 y∈[0,1]**
    (여기선 가우시안 heatmap 값)을 그대로 품질 타깃으로 쓰는 BCE에, 예측이 타깃에서 멀수록 커지는
    변조항 ``|y - σ|^β``를 곱한다. → 정확도(품질)와 분류 신뢰도를 하나로 묶어 랭킹을 개선(mAP/NDS ↑).

    Args:
        pred: [B, C, H, W] sigmoid 확률(0~1).
        target: 같은 shape, 가우시안 soft 타깃 [0,1] (중심=1, 주변 감쇠).
        beta: 변조항 지수(focusing). 기본 2.0.
    """
    pred = pred.clamp(min=eps, max=1 - eps)            # log 안정화
    # 품질-인지 BCE: 타깃 y로 양/음을 부드럽게 섞음(y=1 중심 → -log σ, y=0 배경 → -log(1-σ))
    bce = -(target * pred.log() + (1 - target) * (1 - pred).log())
    modulating = (target - pred).abs().pow(beta)      # 예측이 타깃과 멀수록(어려울수록) 가중 ↑
    loss = modulating * bce                            # [B,C,H,W]
    num_pos = target.eq(1).float().sum().clamp(min=1.0)  # gaussian_focal과 동일 정규화(중심 개수)
    return loss.sum() / num_pos


# ---------------- CenterPoint-style dense loss (CR3DT-inspired) ----------------
class CenterPointLoss(nn.Module):
    """CenterHead 출력에 맞춘 dense CenterPoint loss.

    구성 요소:
      - Heatmap: 클래스별 가우시안 splat GT 대비 gaussian focal loss (검출 분류)
      - Regression L1: GT 셀 중심에서만 (offset, height, dim_log, sin, cos, vx, vy) 회귀

    설계는 CR3DT(바닐라 CenterPoint)를 따른다: Hungarian matcher 없음, transformer aux loss 없음.
    총 두 개의 loss 항(heatmap + box)만 사용한다.
    """

    def __init__(
        self,
        num_classes: int = 10,             # 검출 클래스 수 (heatmap 채널 수)
        cls_weight: float = 1.0,           # (현재 forward에서 0과 곱해져 미사용/dead) 분류 항 가중치 자리 — heatmap+box 두 항만 실제 동작
        box_weight: float = 0.25,          # 회귀(box) loss 가중치
                                           # 실제 config(train.yaml)=1.0 (회귀 underfit 수정: 0.25→1.0, 기본값 0.25와 다름)
        heatmap_weight: float = 1.0,       # heatmap focal loss 가중치
        bev_h: int = 200,                  # BEV 격자 세로 셀 수
        bev_w: int = 200,                  # BEV 격자 가로 셀 수
        x_min: float = -50.0,              # BEV가 커버하는 x(전방) 범위 [m]
        x_max: float = 50.0,
        y_min: float = -50.0,              # BEV가 커버하는 y(좌우) 범위 [m]
        y_max: float = 50.0,
        gaussian_overlap: float = 0.1,     # gaussian_radius 계산에 쓰는 최소 IoU(반경 산정 기준)
                                           #   [③] 스칼라 또는 클래스별 리스트(len=num_classes).
                                           #   값이 클수록 반경이 좁아진다 → 대형 객체 peak 확산 억제.
        min_radius: int = 2,               # 가우시안 splat 최소 반경(작은 객체도 최소한의 봉우리 보장)
        max_radius: int = 0,               # [③] 가우시안 반경 상한(0=미적용). 대형 객체 peak 확산 차단.
        # [②'] 클래스별 box 회귀 가중치(len=num_classes). anchor 기반 검출기의 클래스 전용 용량
        #   배분을 구조 변경 없이 흉내낸다. 희소·대형 클래스가 car/pedestrian gradient에 묻히는 것 완화.
        class_reg_weights=None,
        # [①] 클래스별 크기 사전 log(w,l,h) (shape [num_classes,3]). 지정하면 크기 회귀 타깃에서
        #   이 값을 빼 '잔차 회귀'로 바꾼다. 디코드(center_head)에서 동일 값을 더해 복원해야 한다.
        size_prior=None,
        # 10-d code 각 차원 가중치: (offset_x, offset_y, cz, log_w, log_l, log_h, sin, cos, vx, vy)
        code_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),  # 속도(vx,vy)는 0.2로 낮춰 다른 항을 압도 않게
                                                                          # [mAVE fix] config(train.yaml)에서 vx,vy=1.0으로 상향(0.2→1.0)
        # [mAVE fix] 속도 head collapse(예측 median 0.017 vs GT car 1.7m/s) 대응: object-center
        #   속도 L1에 moving-object 가중 w=1+min(|v_gt|,cap)를 곱해 정적 다수(GT median 0.05)에
        #   묻히던 움직이는 물체의 gradient를 강화. cap은 드문 고속(17m/s) 객체가 loss를 폭주시켜
        #   grad-clip을 지배하지 않도록 상한. vel_moving_weight=False면 기존(균일) 동작.
        # [plateau fix] 회귀 타깃 전채널 정규화 divisor(각 채널 std). None이면 정규화 없음(기존).
        #   진단결과 全 채널이 res/std≈1~1.9로 underfit → 채널별 std로 나눠 조건화·gradient 균형.
        #   순서: (off_x,off_y,cz,log_w,log_l,log_h,sin,cos,vx,vy)
        code_norm=None,
        vel_moving_weight: bool = True,    # 속도 L1 moving 가중 사용 여부
        vel_weight_cap: float = 10.0,      # 가중 상한: w = 1 + min(|v_gt|, cap)
        # [mAP fix] heatmap 분류 loss 종류: "gaussian_focal"(CornerNet 기존) | "qfl"(GFL 분류항).
        #   qfl은 품질-인지 BCE로 랭킹을 개선 → mAP/NDS 향상 노림. CenterHead 구조는 그대로 유지.
        heatmap_loss_type: str = "gaussian_focal",
        qfl_beta: float = 2.0,             # QFL 변조항 지수
        # [B②] direction-bin 분류 loss 가중치. >0이면 GT 중심 셀에서 dense_dir(2-class,
        #   bin1 ⇔ cos(yaw)>0)에 cross-entropy를 건다. head의 use_dir_classifier와 함께 사용.
        dir_weight: float = 0.0,
        # [A-yaw] MultiBin yaw (head.use_multibin_yaw와 세트). sin/cos는 10ch L1 스택에서
        #   빠지고(인덱스 6,7 무시) 대신 bin CE + GT-bin 잔차 (sin,cos) L1로 감독.
        multibin_yaw: bool = False,
        yaw_bin_weight: float = 1.0,       # bin 분류 CE 가중치
        yaw_res_weight: float = 2.0,       # bin 잔차 L1 가중치
        # [A-yaw] 이웃 감독 가중치. decode가 heatmap 피크(±1셀 지터) 셀의 회귀값을 읽는데
        #   기존 감독은 정확히 중심 셀 1칸뿐 → 3×3 이웃에도 동일 타깃을 이 가중치로 감독.
        #   (offset 타깃은 셀별로 재계산 → 이웃 셀에서 decode해도 진짜 중심을 가리킴.)
        #   0.0이면 기존(중심만) 동작.
        neighborhood_weight: float = 0.0,
        # [중심보정 8/27] 크기 적응형 offset 감독. 감독 반경 = heatmap 가우시안 반경 × scale.
        #   0이면 비활성(기존 ±1셀 이웃 감독만). 실측 피크오차(중앙값 car 0.7~trailer 3.5셀)를
        #   포괄해 decode가 읽는 셀에서도 offset이 참 중심을 가리키게 한다.
        offset_radius_scale: float = 0.0,
        offset_max_radius: int = 12,
        offset_weight: float = 1.0,      # 이 항의 총 가중치
        key: str = "vehicle",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.heatmap_weight = heatmap_weight
        self.vel_moving_weight = vel_moving_weight
        self.vel_weight_cap = vel_weight_cap
        self.heatmap_loss_type = heatmap_loss_type
        self.qfl_beta = qfl_beta
        self.dir_weight = dir_weight
        self.multibin_yaw = multibin_yaw
        self.yaw_bin_weight = yaw_bin_weight
        self.yaw_res_weight = yaw_res_weight
        self.neighborhood_weight = neighborhood_weight
        self.offset_radius_scale = float(offset_radius_scale)
        self.offset_max_radius = int(offset_max_radius)
        self.offset_weight = float(offset_weight)
        # bin 중심 (head.yaw_bin_centers와 동일해야 함)
        self.register_buffer("yaw_bin_centers",
                             torch.tensor([0.0, 1.5707963, 3.1415927, -1.5707963]))
        self.bev_h, self.bev_w = bev_h, bev_w
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        # 렌더러/`view` 좌표 규칙(CenterHead 참고)과 일치: col = W/2 - y/res_col, row = H/2 - x/res_row.
        # res = 셀 하나가 담는 실제 거리(미터/셀).
        self.res_row = (x_max - x_min) / bev_h   # row(=x 방향) 해상도 [m/cell]
        self.res_col = (y_max - y_min) / bev_w   # col(=y 방향) 해상도 [m/cell]
        # [③] 클래스별 반경 제어
        if isinstance(gaussian_overlap, (list, tuple)):
            assert len(gaussian_overlap) == num_classes, "gaussian_overlap 길이가 num_classes와 달라요"
            self.gaussian_overlap = [float(x) for x in gaussian_overlap]
        else:
            self.gaussian_overlap = [float(gaussian_overlap)] * num_classes
        self.min_radius = min_radius
        self.max_radius = int(max_radius)
        # [②'] 클래스별 box 회귀 가중치
        if class_reg_weights is not None:
            assert len(class_reg_weights) == num_classes, "class_reg_weights 길이가 num_classes와 달라요"
            self.class_reg_weights = [float(x) for x in class_reg_weights]
        else:
            self.class_reg_weights = None
        # [①] 클래스별 크기 사전(log). center_head.size_prior와 반드시 동일해야 한다.
        if size_prior is not None:
            sp = torch.tensor([list(r) for r in size_prior], dtype=torch.float32)
            assert sp.shape == (num_classes, 3), f"size_prior shape {tuple(sp.shape)} != ({num_classes}, 3)"
            self.register_buffer("size_prior", sp)
        else:
            self.size_prior = None
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32))  # 버퍼로 등록(자동 device 이동)
        # [plateau fix] 전채널 정규화 divisor. 지정되면 residual을 채널별 std로 나눠 조건화.
        if code_norm is not None:
            self.register_buffer("code_norm", torch.tensor(list(code_norm), dtype=torch.float32).clamp(min=1e-3))
        else:
            self.code_norm = None
        self.key = key

    def _build_targets(self, gt_boxes_list, gt_labels_list, device):
        """클래스별 가우시안 heatmap + 셀별 회귀 타겟 + 마스크를 만든다."""
        B = len(gt_boxes_list)             # 배치 크기 실제: B=1 (train batch_size=1)
        H, W = self.bev_h, self.bev_w     # 실제: H=W=200, num_classes=10
        heatmap = torch.zeros((B, self.num_classes, H, W), dtype=torch.float32, device=device)  # 클래스별 heatmap 타겟 실제: [1, 10, 200, 200]
        # 회귀 타겟: 셀마다 10차원 벡터. mask는 GT 중심 셀만 1로 표시(거기서만 회귀 loss 계산).
        # 0:offset_x, 1:offset_y, 2:cz, 3:log_w, 4:log_l, 5:log_h, 6:sin, 7:cos, 8:vx, 9:vy
        reg_target = torch.zeros((B, 10, H, W), dtype=torch.float32, device=device)  # [B,10,H,W] 회귀 타겟
        reg_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)     # [B,1,H,W] GT 중심 마스크
        # [mAVE fix] 속도 채널 전용 moving 가중 맵. 중심 셀마다 w=1+min(|v_gt|,cap), 나머지 0.
        vel_wmap = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)      # [B,1,H,W] 속도 L1 가중
        # [B②] dir-bin 타겟: 중심 셀마다 bin(1 ⇔ cos(yaw)>0). 마스크는 reg_mask 재사용.
        dir_target = torch.zeros((B, H, W), dtype=torch.long, device=device)          # [B,H,W]
        # [A-yaw] MultiBin 타겟: bin id + 잔차 (sin,cos)
        yaw_bin_target = torch.zeros((B, H, W), dtype=torch.long, device=device)
        yaw_res_target = torch.zeros((B, 2, H, W), dtype=torch.float32, device=device)
        # [중심보정] 크기 적응형 offset 감독용 별도 타깃/가중/정규화 맵
        off_target = torch.zeros((B, 2, H, W), dtype=torch.float32, device=device)
        off_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
        off_norm = torch.ones((B, 1, H, W), dtype=torch.float32, device=device)

        # 실제: 샘플당 gb=gt_boxes [M, 10](M=객체수), gl=gt_labels [M]
        for b, (gb, gl) in enumerate(zip(gt_boxes_list, gt_labels_list)):  # 배치 샘플별로 (박스들, 라벨들)
            if gb is None or gb.numel() == 0:   # 객체가 없는 샘플은 건너뜀
                continue
            for i in range(gb.shape[0]):        # 박스 하나씩 처리
                cls_id = int(gl[i].item())      # 이 박스의 클래스 id
                if cls_id < 0 or cls_id >= self.num_classes:  # 범위 밖 클래스는 무시
                    continue
                cx, cy, cz = float(gb[i, 0]), float(gb[i, 1]), float(gb[i, 2])  # 박스 중심 좌표 (x전방, y좌우, z높이) [m]
                w_w, l_w, h_w = float(gb[i, 3]), float(gb[i, 4]), float(gb[i, 5])  # 박스 크기 (width, length, height) [m]
                if w_w <= 0 or l_w <= 0 or h_w <= 0:  # 비정상 크기 박스 제외
                    continue
                sin_y, cos_y = float(gb[i, 6]), float(gb[i, 7])  # yaw(방향)을 sin/cos로 인코딩한 값
                vx, vy = float(gb[i, 8]), float(gb[i, 9])        # 속도 (x, y) [m/s]
                # 실세계 좌표(cx,cy)를 BEV 셀 인덱스로 변환 (렌더러/`view` 규칙):
                # col = W/2 - y/res_col, row = H/2 - x/res_row.
                col_f = self.bev_w / 2.0 - cy / self.res_col   # 실수 col 좌표
                row_f = self.bev_h / 2.0 - cx / self.res_row   # 실수 row 좌표
                col_i, row_i = int(col_f), int(row_f)          # 정수 셀 인덱스
                if not (0 <= col_i < W and 0 <= row_i < H):    # BEV 범위 밖이면 무시
                    continue
                # heatmap 가우시안 splat. 객체 발자국: length는 x(→row), width는 y(→col) 방향.
                w_cell = w_w / self.res_col                    # width를 셀 단위로 환산
                l_cell = l_w / self.res_row                    # length를 셀 단위로 환산
                # [③] 클래스별 min_overlap: 값이 클수록 반경이 좁아진다. 12m급 객체가
                #   반경 ~11셀로 퍼져 peak가 모호해지는 문제를 클래스 단위로 제어한다.
                radius = gaussian_radius((l_cell, w_cell), min_overlap=self.gaussian_overlap[cls_id])
                radius = max(self.min_radius, int(radius))     # 최소 반경 보장
                if self.max_radius > 0:
                    radius = min(radius, self.max_radius)      # [③] 반경 상한
                draw_heatmap_gaussian(heatmap[b, cls_id], (col_i, row_i), radius)  # 해당 클래스 채널에 가우시안 봉우리 찍기
                # [A-yaw] MultiBin: yaw → bin id + 잔차각 (bin폭 90°, 잔차 ∈ ±45°)
                yaw = float(np.arctan2(sin_y, cos_y))
                bin_id = int(round(yaw / (np.pi / 2))) % 4
                res_ang = yaw - float(self.yaw_bin_centers[bin_id])
                res_ang = (res_ang + np.pi) % (2 * np.pi) - np.pi      # wrap to [-π, π]
                log_w = float(np.log(max(w_w, 1e-3)))
                log_l = float(np.log(max(l_w, 1e-3)))
                log_h = float(np.log(max(h_w, 1e-3)))
                vmag = (vx * vx + vy * vy) ** 0.5
                # [A-yaw] 3×3 이웃 감독: 중심 1.0, 이웃 neighborhood_weight(0이면 중심만).
                #   offset 타겟은 셀별 재계산 → 이웃 셀에서 decode해도 진짜 중심을 가리킴.
                #   겹침 규칙: 기존 가중치보다 높을 때만 덮어씀(중심이 이웃을 항상 이김).
                nbw = float(self.neighborhood_weight)
                offsets = [(0, 0, 1.0)]
                if nbw > 0.0:
                    offsets += [(di, dj, nbw) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                                if not (di == 0 and dj == 0)]

                # [중심보정 8/27] 크기 적응형 offset 감독. 실측: heatmap 피크 셀 오차 중앙값이
                #   car 0.7 / truck 1.3 / bus 1.4 / CV 2.2 / trailer 3.5 셀인데 위 3×3 감독은
                #   ±1셀뿐이라 car조차 37%, trailer는 87%가 감독 범위 밖 → decode가 읽는 셀의
                #   offset이 학습되지 않아 중심 오차가 구조적으로 복구 불가였다.
                #   → 피크가 흔들리는 범위(=heatmap 가우시안 반경) 전체에 offset을 감독한다.
                #   가중치는 가우시안 감쇠(피크가 있을 확률과 일치) → 먼 셀이 수적으로 압도하지 않음.
                #   offset 채널(0,1)에만 적용하고 나머지 채널은 기존 ±1셀 유지(과도한 blur 방지).
                if self.offset_radius_scale > 0.0:
                    r_off = int(min(self.offset_max_radius,
                                    max(1, round(radius * self.offset_radius_scale))))
                    sig2 = 2.0 * max(1.0, (r_off / 2.0) ** 2)
                    r0, r1 = max(0, row_i - r_off), min(H, row_i + r_off + 1)
                    c0, c1 = max(0, col_i - r_off), min(W, col_i + r_off + 1)
                    rr = torch.arange(r0, r1, device=device).view(-1, 1)
                    cc = torch.arange(c0, c1, device=device).view(1, -1)
                    d2 = (rr - row_i).float() ** 2 + (cc - col_i).float() ** 2
                    wmap_o = torch.exp(-d2 / sig2) * (d2 <= r_off * r_off)
                    # 겹침: 가중치가 더 큰 객체가 이김
                    prev = off_mask[b, 0, r0:r1, c0:c1]
                    win = wmap_o > prev
                    if bool(win.any()):
                        off_mask[b, 0, r0:r1, c0:c1] = torch.where(win, wmap_o, prev)
                        # 셀별 재계산된 offset 타깃(그 셀에서 본 진짜 중심까지의 변위)
                        tx = (col_f - cc.float() - 0.5).expand_as(d2)
                        ty = (row_f - rr.float() - 0.5).expand_as(d2)
                        off_target[b, 0, r0:r1, c0:c1] = torch.where(win, tx, off_target[b, 0, r0:r1, c0:c1])
                        off_target[b, 1, r0:r1, c0:c1] = torch.where(win, ty, off_target[b, 1, r0:r1, c0:c1])
                        # 스케일 불변 정규화: 잔차를 반경으로 나눠 클래스 간 gradient 균형 유지
                        off_norm[b, 0, r0:r1, c0:c1] = torch.where(
                            win, torch.full_like(d2, float(max(1.0, r_off))),
                            off_norm[b, 0, r0:r1, c0:c1])
                for di, dj, wgt in offsets:
                    r_i, c_i = row_i + di, col_i + dj
                    if not (0 <= r_i < H and 0 <= c_i < W):
                        continue
                    if float(reg_mask[b, 0, r_i, c_i]) >= wgt:
                        continue
                    reg_target[b, 0, r_i, c_i] = col_f - c_i - 0.5     # offset_x: 이 셀 기준 col 잔차
                    reg_target[b, 1, r_i, c_i] = row_f - r_i - 0.5     # offset_y: 이 셀 기준 row 잔차
                    reg_target[b, 2, r_i, c_i] = cz
                    # [①] 크기 사전이 있으면 잔차 회귀로 전환(디코드에서 동일 값을 더해 복원).
                    if self.size_prior is not None:
                        sp = self.size_prior[cls_id]
                        reg_target[b, 3, r_i, c_i] = log_w - float(sp[0])
                        reg_target[b, 4, r_i, c_i] = log_l - float(sp[1])
                        reg_target[b, 5, r_i, c_i] = log_h - float(sp[2])
                    else:
                        reg_target[b, 3, r_i, c_i] = log_w
                        reg_target[b, 4, r_i, c_i] = log_l
                        reg_target[b, 5, r_i, c_i] = log_h
                    reg_target[b, 6, r_i, c_i] = sin_y
                    reg_target[b, 7, r_i, c_i] = cos_y
                    reg_target[b, 8, r_i, c_i] = vx
                    reg_target[b, 9, r_i, c_i] = vy
                    # [②'] 클래스별 회귀 가중치를 마스크에 실어 box 손실에 그대로 반영한다.
                    cw_cls = 1.0 if self.class_reg_weights is None else self.class_reg_weights[cls_id]
                    reg_mask[b, 0, r_i, c_i] = wgt * cw_cls           # 가중 마스크(중심 1.0/이웃 nbw)×클래스가중
                    # [mAVE fix] moving 가중은 셀 가중과 곱해져 적용
                    vel_wmap[b, 0, r_i, c_i] = 1.0 + min(vmag, self.vel_weight_cap)
                    dir_target[b, r_i, c_i] = 1 if cos_y > 0 else 0    # [B②] dir-bin 타겟
                    yaw_bin_target[b, r_i, c_i] = bin_id
                    yaw_res_target[b, 0, r_i, c_i] = float(np.sin(res_ang))
                    yaw_res_target[b, 1, r_i, c_i] = float(np.cos(res_ang))
        return (heatmap, reg_target, reg_mask, vel_wmap, dir_target,
                yaw_bin_target, yaw_res_target,
                off_target, off_mask, off_norm)   # + [중심보정] 크기적응형 offset 타깃/가중/정규화

    def forward(self, prediction: dict, batch: dict) -> torch.Tensor:
        gt_boxes_list = batch.get(f"{self.key}_gt_boxes")    # GT 박스들 (배치별 텐서 또는 리스트)
        gt_labels_list = batch.get(f"{self.key}_gt_labels")  # GT 클래스 라벨들
        if gt_boxes_list is None or gt_labels_list is None:  # GT가 없으면 0 loss (gradient는 유지)
            return prediction[f"{self.key}_dense_heatmap"].new_zeros((), requires_grad=True)
        if isinstance(gt_boxes_list, torch.Tensor):          # 텐서 형태면 배치 차원으로 쪼개 리스트화
            gt_boxes_list = [gt_boxes_list[i] for i in range(gt_boxes_list.shape[0])]
        if isinstance(gt_labels_list, torch.Tensor):
            gt_labels_list = [gt_labels_list[i] for i in range(gt_labels_list.shape[0])]

        device = prediction[f"{self.key}_dense_heatmap"].device  # 연산 device

        with torch.no_grad():   # 타겟 생성은 미분 대상 아님(상수 타겟)
            (gt_hm, gt_reg, gt_mask, vel_wmap, gt_dir,
             gt_yaw_bin, gt_yaw_res,
             gt_off, off_mask, off_norm) = self._build_targets(gt_boxes_list, gt_labels_list, device)

        # --- (1) Heatmap focal loss (분류) ---
        hm_pred = prediction[f"{self.key}_dense_heatmap"].sigmoid().clamp(min=1e-4, max=1 - 1e-4)  # logit→확률, 수치 안정 클램프 실제: [B=1, 10, 200, 200]
        # [mAP fix] heatmap 분류 loss 선택: qfl(GFL 분류항) 또는 기존 gaussian_focal.
        if self.heatmap_loss_type == "qfl":
            cls_loss = _quality_focal_loss(hm_pred, gt_hm, beta=self.qfl_beta)      # 품질-인지 BCE(랭킹 개선)
        else:
            cls_loss = _gaussian_focal_loss(hm_pred, gt_hm, alpha=2.0, gamma=4.0)  # 예측 heatmap vs 가우시안 GT

        # --- (2) Regression L1 (GT 중심 셀에서만) ---
        # 흩어진 dense 회귀 헤드들을 [B, 10, H, W] 한 텐서로 쌓는다 (gt_reg와 채널 순서 일치)
        # 실제 B=1, H=W=200. 아래 분기 shape의 B/H/W에 [1,..,200,200] 대입.
        pred_off = prediction[f"{self.key}_dense_offset"]    # [B, 2, H, W] offset_x, offset_y 실제: [1, 2, 200, 200]
        pred_h   = prediction[f"{self.key}_dense_height"]    # [B, 1, H, W] cz(높이)           실제: [1, 1, 200, 200]
        pred_d   = prediction[f"{self.key}_dense_dim_log"]   # [B, 3, H, W] log_w, log_l, log_h 실제: [1, 3, 200, 200]
        pred_r   = prediction[f"{self.key}_dense_rot"]       # [B, 2 또는 12, H, W] yaw (multibin이면 12ch)
        pred_v   = prediction[f"{self.key}_dense_vel"]       # [B, 2, H, W] vx, vy (속도)       실제: [1, 2, 200, 200]
        # [A-yaw] multibin이면 sin/cos(6,7)는 L1 스택에서 제외하고 별도 bin CE+잔차 L1로 감독
        if self.multibin_yaw:
            ch_idx = [0, 1, 2, 3, 4, 5, 8, 9]
            pred_reg = torch.cat([pred_off, pred_h, pred_d, pred_v], dim=1)   # [B, 8, H, W]
            gt_reg_s = gt_reg[:, ch_idx]
            cw_full = self.code_weights[ch_idx]
        else:
            ch_idx = list(range(10))
            pred_reg = torch.cat([pred_off, pred_h, pred_d, pred_r, pred_v], dim=1)  # [B, 10, H, W]
            gt_reg_s = gt_reg
            cw_full = self.code_weights

        cw = cw_full.to(device=device, dtype=pred_reg.dtype).view(1, -1, 1, 1)
        diff = (pred_reg - gt_reg_s).abs()     # 채널별 L1 오차
        # [plateau fix] 전채널 정규화: residual을 채널 std로 나눠 조건화(각 채널 O(1)) → 채널간 gradient 균형.
        if self.code_norm is not None:
            diff = diff / self.code_norm[ch_idx].to(device=device, dtype=pred_reg.dtype).view(1, -1, 1, 1)
        diff = diff * cw                       # code_weights 적용
        # [mAVE fix] 속도 채널(마지막 2ch)에만 moving 가중(w=1+min(|v|,cap)) 곱.
        if self.vel_moving_weight:
            wmap = torch.ones_like(diff)
            wmap[:, -2:] = vel_wmap
            diff = diff * wmap
        mask = gt_mask.expand_as(diff)         # 가중 마스크(중심 1.0/이웃 nbw) 확장
        num_pos = mask.sum().clamp(min=1.0)    # 가중 합(0나눗셈 방지)
        box_loss = (diff * mask).sum() / num_pos / float(cw_full.sum().item())  # 마스크 영역 평균 L1을 weight 합으로 재정규화

        # --- [A-yaw] MultiBin yaw loss (bin CE + GT-bin 잔차 L1, 가중 마스크 적용) ---
        yaw_loss = pred_reg.new_zeros(())
        if self.multibin_yaw and pred_r.shape[1] == 12:
            wcell = gt_mask[:, 0]                                        # [B, H, W] 셀 가중
            wsum = wcell.sum().clamp(min=1.0)
            ce_map = F.cross_entropy(pred_r[:, :4], gt_yaw_bin, reduction="none")  # [B,H,W]
            bin_loss = (ce_map * wcell).sum() / wsum
            # GT bin의 잔차 채널만 gather: [B,4,2,H,W] → [B,2,H,W]
            B_, _, H_, W_ = pred_r.shape
            res_all = pred_r[:, 4:].view(B_, 4, 2, H_, W_)
            idx = gt_yaw_bin[:, None, None, :, :].expand(-1, 1, 2, -1, -1)
            res_pred = res_all.gather(1, idx).squeeze(1)                 # [B,2,H,W]
            res_l1 = ((res_pred - gt_yaw_res).abs().sum(dim=1) * wcell).sum() / wsum / 2.0
            yaw_loss = self.yaw_bin_weight * bin_loss + self.yaw_res_weight * res_l1

        # --- (2b) [중심보정] 크기 적응형 offset L1 (가우시안 감쇠 가중, 반경 정규화) ---
        off_loss = pred_reg.new_zeros(())
        if self.offset_radius_scale > 0.0 and float(off_mask.sum()) > 0:
            diff_o = (pred_off - gt_off).abs() / off_norm.clamp(min=1.0)
            denom_o = off_mask.sum().clamp(min=1.0) * 2.0    # 2채널
            off_loss = (diff_o * off_mask).sum() / denom_o

        # --- (3) [B②] dir-bin CE (GT 중심 셀에서만) ---
        dir_loss = pred_reg.new_zeros(())
        dir_key = f"{self.key}_dense_dir"
        if self.dir_weight > 0.0 and dir_key in prediction and prediction[dir_key] is not None:
            pred_dir = prediction[dir_key]                     # [B, 2, H, W] logits
            ce_map = F.cross_entropy(pred_dir, gt_dir, reduction="none")  # [B, H, W]
            m = gt_mask[:, 0]                                  # [B, H, W] GT 중심 마스크
            dir_loss = (ce_map * m).sum() / m.sum().clamp(min=1.0)

        # 총손실 = heatmap*가중치 + (cls는 0으로 비활성) + box*가중치 + [B②] dir + [A-yaw] multibin
        # cls_weight*0.0 → cls 항은 dead(항상 0 기여).
        return (self.heatmap_weight * cls_loss + self.cls_weight * 0.0
                + self.box_weight * box_loss + self.dir_weight * dir_loss + yaw_loss
                + self.offset_weight * off_loss)


# ========================================================================
# C1: Object-aware Gaussian supervision (auxiliary, training-only)
# ========================================================================
# ========================================================================
# GaussMOT ④: Doppler consistency loss
# (레이더 도플러 속도로 BEV 속도장을 직접 supervise — mAVE/NDS 향상 핵심)
# ========================================================================
class DopplerConsistencyLoss(nn.Module):
    """Augmented-Gaussian BEV 속도장에 대한 Doppler 일관성 loss (GaussMOT ④).

    Augmented-Gaussian primitive(①)은 각 Gaussian의 속도를 BEV 속도장
    ``{key}_velocity_bev`` [B, 2, H, W] (ego/BEV 좌표계, m/s)로 splat한다. 레이더는 각 반환점에서
    (ego 보정된) ego-frame 속도를 직접 측정한다. 이 loss는 레이더 반환점이 떨어지는 위치마다
    예측 BEV 속도를 레이더 측정값 쪽으로 끌어당겨, GT 박스 없이도 센서로부터 직접 속도 헤드를
    학습시킨다 — mAVE / NDS를 끌어올리는 핵심 지렛대.

    타겟은 점별 보정 속도 ``radar_points[:, vel_cols]``를 BEV 격자에 scatter(셀마다 반환점 평균)하여
    만든다. CenterPointLoss와 동일한 ``view`` 좌표 규칙을 쓴다. 점유 셀에 대해서만 L1.
    """

    def __init__(
        self,
        bev_h: int = 200,                  # BEV 세로 셀 수
        bev_w: int = 200,                  # BEV 가로 셀 수
        x_min: float = -50.0,              # x(전방) 범위 [m]
        x_max: float = 50.0,
        y_min: float = -50.0,              # y(좌우) 범위 [m]
        y_max: float = 50.0,
        vel_cols: tuple = (15, 16),        # radar_points에서 (vx, vy) 보정 속도가 들어있는 컬럼 인덱스
        xy_cols: tuple = (0, 1),           # radar_points에서 (x, y) 좌표가 들어있는 컬럼 인덱스
        filter_col: int = -1,              # 유효성 필터에 쓸 컬럼(예: 유효 플래그)
        use_filter: bool = True,           # 필터 컬럼으로 잡음 반환점을 걸러낼지 여부
        key: str = "vehicle",
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.res_row = (x_max - x_min) / bev_h   # row(=x) 해상도 [m/cell]
        self.res_col = (y_max - y_min) / bev_w   # col(=y) 해상도 [m/cell]
        self.vel_cols = vel_cols
        self.xy_cols = xy_cols
        self.filter_col = filter_col
        self.use_filter = use_filter
        self.key = key

    @torch.no_grad()   # 타겟은 레이더 측정 상수이므로 gradient 없음
    def _build_target(self, radar_points, device):
        H, W = self.bev_h, self.bev_w     # 실제: H=W=200
        # 실제: radar_points는 길이 B(=1) list, 각 원소 [N_pts, C≈50]
        B = len(radar_points)              # 배치 크기 실제: B=1
        tgt = torch.zeros(B, 2, H, W, device=device)   # 레이더 속도 타겟장 [B,2,H,W] (vx, vy) 실제: [1, 2, 200, 200]
        mask = torch.zeros(B, 1, H, W, device=device)  # 레이더 반환점이 있는 셀만 1로 표시 실제: [1, 1, 200, 200]
        for b, pts in enumerate(radar_points):  # 배치 샘플별 레이더 포인트들
            if pts is None or pts.numel() == 0:
                continue
            pts = pts.to(device).float()        # [N_points, C] 실제: [N_pts, C≈50] (vel_cols=15,16 / xy_cols=0,1 / filter_col=-1)
            x = pts[:, self.xy_cols[0]]         # 각 점의 x 좌표 [N]
            y = pts[:, self.xy_cols[1]]         # 각 점의 y 좌표 [N]
            vx = pts[:, self.vel_cols[0]]       # 각 점의 보정 속도 vx [N]
            vy = pts[:, self.vel_cols[1]]       # 각 점의 보정 속도 vy [N]
            keep = torch.ones_like(x, dtype=torch.bool)  # 사용할 점 마스크(초기 전부 True)
            if self.use_filter and abs(self.filter_col) <= pts.shape[1]:
                keep &= pts[:, self.filter_col] > 0.5    # 유효 플래그가 켜진 점만 사용
            col = (W / 2.0 - y / self.res_col).long()    # y → BEV col 인덱스 (CenterPoint와 동일 규칙)
            row = (H / 2.0 - x / self.res_row).long()    # x → BEV row 인덱스
            keep &= (col >= 0) & (col < W) & (row >= 0) & (row < H)  # 격자 범위 안의 점만 유지
            if keep.sum() == 0:
                continue
            flat = (row[keep] * W + col[keep])           # (row,col) → 1D 평탄 인덱스 (셀당 누적용)
            # 같은 셀에 여러 점이 떨어질 수 있으므로 셀별 개수/속도합을 index_add_로 누적
            cnt = torch.zeros(H * W, device=device).index_add_(
                0, flat, torch.ones_like(flat, dtype=torch.float32))  # 셀별 점 개수
            sx = torch.zeros(H * W, device=device).index_add_(0, flat, vx[keep])  # 셀별 vx 합
            sy = torch.zeros(H * W, device=device).index_add_(0, flat, vy[keep])  # 셀별 vy 합
            occ = cnt > 0                                # 점유된(반환점 있는) 셀
            mx = torch.zeros_like(sx); my = torch.zeros_like(sy)
            mx[occ] = sx[occ] / cnt[occ]                # 셀별 평균 vx (합/개수)
            my[occ] = sy[occ] / cnt[occ]                # 셀별 평균 vy
            tgt[b, 0] = mx.view(H, W)                   # 평탄 → [H,W]로 복원해 타겟 채널0(vx)에
            tgt[b, 1] = my.view(H, W)                   # 채널1(vy)에
            mask[b, 0] = occ.float().view(H, W)         # 점유 셀 마스크 [H,W]
        return tgt, mask                                # (레이더 속도 타겟장, 점유 마스크)

    def forward(self, prediction: dict, batch: dict) -> torch.Tensor:
        # [수정 ③] 검출 box 속도 head(dense_vel)를 직접 supervise (기존 velocity_bev는 박스 속도에
        #   안 쓰이는 보조장이라 mAVE에 무효였음). radar 반환점 셀마다 dense 속도 지도를 줘서
        #   sparse GT L1만으론 underfit하던 velocity head를 보강 → mAVE 개선. (전제: ①② radar 프레임 정합)
        vel_key = f"{self.key}_dense_vel"      # 검출 head의 dense 속도 출력 [B,2,H,W]
        if vel_key not in prediction or prediction[vel_key] is None:  # 없으면 0 loss(gradient 연결만 유지)
            ref = next(iter(prediction.values()))
            return ref.sum() * 0.0
        pred_vel = prediction[vel_key]  # [B, 2, H, W] 예측된 box 속도장 실제: [B=1, 2, 200, 200]
        radar_points = batch.get("radar_points")  # 레이더 포인트 (없으면 0 loss) 실제: 길이 B=1 list, 각 [N_pts, C≈50]
        if radar_points is None:
            return pred_vel.sum() * 0.0
        target, mask = self._build_target(radar_points, pred_vel.device)  # 레이더로부터 속도 타겟장 구성
        denom = mask.sum().clamp(min=1.0)   # 점유 셀 수로 정규화(0나눗셈 방지)
        # 점유 셀에서만 예측 속도 vs 레이더 속도 L1, /2.0은 (vx,vy) 두 채널 평균
        l1 = ((pred_vel - target).abs() * mask).sum() / denom / 2.0
        return l1
