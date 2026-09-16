"""모델 구성 컴포넌트들을 한곳에서 re-export하는 패키지 진입점.

여기서 모은 클래스들로 GaussMOT 파이프라인(센서 인코딩→GS 융합→BEV→검출/추적)을 조립한다.
"""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)  # 프로젝트 루트를 sys.path에 등록(절대 import 가능하게)

from gaussianmot.modeling.components.concat_fuser import ConcatFuser          # 멀티모달 feature를 채널 concat으로 융합
from gaussianmot.modeling.components.det_bev_encoder import DetBEVEncoder     # BEV feature map을 검출용으로 인코딩
from gaussianmot.modeling.components.image_encoder import AGPNeck, PixelsToGaussians  # 카메라 neck + 픽셀→Gaussian 변환
from gaussianmot.modeling.components.center_head import CenterHead           # CenterPoint식 dense 검출 head(heatmap+회귀)
from gaussianmot.modeling.components.cc3dt_tracker import CC3DTPPTracker     # CC-3DT++ 기반 3D 멀티오브젝트 tracker
from gaussianmot.modeling.components.radar_encoder import PointsToGaussians  # radar 포인트→Gaussian 변환
from gaussianmot.modeling.components.track_head import TrackHead            # per-pixel track embedding 예측 head

# Inference-time bbox decoder (used by tools/evaluate.py).
from .query_bbox_decoder import QueryBBoxDecoder   # 평가 시 dense 출력→박스[N,9] 디코더
