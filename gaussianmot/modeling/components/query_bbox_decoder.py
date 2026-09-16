"""Query-based BBox decoder (Phase 3c).

QueryDetectionHead 출력을 (boxes, scores, labels, embeds) per-batch list로 변환.
heatmap topk 대신 score threshold + 옵션 max_num만 적용.

Track embedding은 query 자체에는 없음 — 별도 TrackHead 출력
({key}_track_embed [B, D, H, W])에서 query의 predicted center 픽셀을 sample.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np                       # legacy 경로의 cx/cy를 numpy로 옮겨 픽셀 인덱싱할 때 사용
import torch
import torch.nn as nn


def world_to_bev_pixel(
    x: float, y: float,                  # LiDAR 실세계 좌표(미터): x=전방, y=좌측
    x_min: float, x_max: float,          # BEV가 커버하는 x 범위(예: -50~50m)
    y_min: float, y_max: float,          # BEV가 커버하는 y 범위(예: -50~50m)
    bev_w: int, bev_h: int,              # BEV 픽셀 해상도(가로 W, 세로 H)
) -> Tuple[int, int]:                    # 반환: (px=열, py=행) 정수 픽셀 인덱스
    """
    LiDAR(metric) coord → BEV pixel coord.

    nuScenes 컨벤션 (labels.py의 view 매트릭스와 일관):
      - lidar +X (forward) → BEV 이미지 위쪽 (py 작음)
      - lidar +Y (left)    → BEV 이미지 왼쪽 (px 작음)

    매핑:
      px = (y_max - y) / (y_max - y_min) * bev_w
      py = (x_max - x) / (x_max - x_min) * bev_h
    """
    px = int((y_max - y) / (y_max - y_min) * bev_w)   # y(좌측+) → 열: y 클수록 px 작아짐(이미지 왼쪽)
    py = int((x_max - x) / (x_max - x_min) * bev_h)   # x(전방+) → 행: x 클수록 py 작아짐(이미지 위쪽)
    # 경계 클램핑
    px = max(0, min(bev_w - 1, px))      # 열 인덱스를 [0, W-1]로 제한(범위 밖 박스 보호)
    py = max(0, min(bev_h - 1, py))      # 행 인덱스를 [0, H-1]로 제한
    return px, py                        # 샘플링에 쓸 (열, 행)


class QueryBBoxDecoder(nn.Module):
    # 모델의 query head 출력(dense logits/boxes)을 평가용 박스 리스트로 디코딩하는 후처리 모듈.
    def __init__(
        self,
        key: str = "vehicle",            # pred dict 키 prefix(예: "vehicle_pred_logits")
        x_min: float = -50.0, x_max: float = 50.0,   # BEV x(전방) 범위 — legacy 픽셀 샘플링용
        y_min: float = -50.0, y_max: float = 50.0,   # BEV y(좌측) 범위 — legacy 픽셀 샘플링용
        bev_h: int = 200, bev_w: int = 200,          # BEV 해상도(200×200)
        max_num: int = 300,              # 한 프레임당 남길 최대 박스 수(상위 score 우선)
        score_threshold: float = 0.1,    # 이 점수 미만 query는 버림
        # [NMS fix] decode 후 클래스별 circle-NMS (CenterPoint 표준 후처리).
        #   CenterHead의 3×3 max-pool NMS는 반경 0.75m뿐이라 대형 클래스(trailer 12m)의
        #   중복 피크가 살아남아 FP 23배 홍수 → nuScenes AP precision-10% 클리핑으로 AP 0.
        #   같은 클래스 예측끼리 중심거리 < radius면 저점수 쪽 제거. None이면 비활성(기존 동작).
        nms_circle_radii: Optional[List[float]] = None,   # 클래스 id 순 반경 [m]
        # [중심보정 8/26] heatmap centroid 기반 중심 재추정 (추론 전용, 재학습 불필요).
        #   근거: 모델은 GT 중심에 가우시안을 놓도록 학습되므로 blob 형태는 참 중심을 향한다.
        #   문제는 넓고 평탄한 blob 안에서 argmax가 흔들리는 것(추정량 분산). 실측 피크 셀 오차
        #   중앙값: car 0.7 / truck 1.3 / bus 1.4 / CV 2.2 / trailer 3.5 셀 (1셀=0.5m).
        #   → argmax(고분산) 대신 점수 가중 centroid(저분산)로 중심을 재추정한다.
        #   window 반경은 객체 크기에서 유도(학습 때 gaussian_radius와 동일 개념).
        centroid_refine: bool = False,
        centroid_min_ratio: float = 0.3,   # peak 대비 이 비율 미만 셀은 제외(이웃 객체 오염 방지)
        centroid_max_radius: int = 12,     # window 반경 상한 [셀]
    ):
        super().__init__()
        self.key = key
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.bev_h, self.bev_w = bev_h, bev_w
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.nms_circle_radii = list(nms_circle_radii) if nms_circle_radii is not None else None
        self.centroid_refine = bool(centroid_refine)
        self.centroid_min_ratio = float(centroid_min_ratio)
        self.centroid_max_radius = int(centroid_max_radius)

    @staticmethod
    def _circle_nms(cx, cy, scores, labels, radii):
        """클래스별 greedy circle-NMS. 반환: keep bool mask [K].

        점수 내림차순으로 순회하며, 이미 채택된 같은 클래스 박스와 중심거리 < radius[cls]면 제거.
        K ≤ max_num(300)이라 O(K²)로 충분히 빠름.
        """
        K = scores.shape[0]
        keep = torch.zeros(K, dtype=torch.bool, device=scores.device)
        order = scores.argsort(descending=True)
        for idx in order.tolist():
            c = int(labels[idx])
            r = radii[c] if c < len(radii) else 0.0
            if r <= 0:
                keep[idx] = True
                continue
            kept = keep & (labels == labels[idx])
            if kept.any():
                d2 = (cx[kept] - cx[idx]) ** 2 + (cy[kept] - cy[idx]) ** 2
                if bool((d2 < r * r).any()):
                    continue
            keep[idx] = True
        return keep

    def _refine_centers(self, bb, lb, heatmap, anchor):
        """[중심보정] heatmap 점수 가중 centroid로 박스 중심 재추정 (추론 전용).

        bb: [K,10] (cx,cy,cz,w,l,h,sin,cos,vx,vy) — cx,cy는 미터
        lb: [K] 클래스 id / heatmap: [C,H,W] (logit) / anchor: [K,2] = (col,row) 피크 셀
        각 검출마다 자기 클래스 heatmap에서 피크 주변 window의 확률 가중 무게중심을 구해
        (col,row)를 갱신하고, 학습된 sub-cell offset(원 중심 − 피크 셀)은 그대로 더한다.
        window 반경은 예측 크기에서 유도(학습 시 gaussian radius와 같은 개념).
        """
        C, H, W = heatmap.shape
        prob = heatmap.sigmoid()
        res_row = (self.x_max - self.x_min) / H
        res_col = (self.y_max - self.y_min) / W
        out = bb.clone()
        # 원 중심(미터) → 셀 좌표. 학습된 sub-cell offset = 원 셀좌표 − 피크 셀좌표.
        col0 = W / 2.0 - bb[:, 1] / res_col
        row0 = H / 2.0 - bb[:, 0] / res_row
        off_col = col0 - anchor[:, 0]
        off_row = row0 - anchor[:, 1]
        for i in range(bb.shape[0]):
            c = int(lb[i])
            if c >= C:
                continue
            # 반경: 객체 크기(셀)의 절반 정도를 window로. 최소 2, 상한 centroid_max_radius.
            w_cell = float(bb[i, 3]) / res_col
            l_cell = float(bb[i, 4]) / res_row
            r = int(max(2, min(self.centroid_max_radius, round(max(w_cell, l_cell) / 2.0))))
            pc, pr = int(anchor[i, 0]), int(anchor[i, 1])
            c0, c1 = max(0, pc - r), min(W, pc + r + 1)
            r0, r1 = max(0, pr - r), min(H, pr + r + 1)
            win = prob[c, r0:r1, c0:c1]
            peak = win.max()
            if float(peak) <= 0:
                continue
            # 이웃 객체 오염 방지: peak 대비 일정 비율 미만은 0으로
            wgt = torch.where(win >= peak * self.centroid_min_ratio, win, torch.zeros_like(win))
            s = wgt.sum()
            if float(s) <= 0:
                continue
            rows = torch.arange(r0, r1, device=win.device, dtype=win.dtype).view(-1, 1) + 0.5
            cols = torch.arange(c0, c1, device=win.device, dtype=win.dtype).view(1, -1) + 0.5
            cen_row = (wgt * rows).sum() / s
            cen_col = (wgt * cols).sum() / s
            # centroid(셀 중심 기준) + 학습된 sub-cell offset → 미터 변환
            out[i, 0] = (H / 2.0 - (cen_row + off_row[i])) * res_row
            out[i, 1] = (W / 2.0 - (cen_col + off_col[i])) * res_col
        return out

    @torch.no_grad()                     # 추론 전용 — gradient 불필요
    def forward(self, pred: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        logits = pred[f"{self.key}_pred_logits"]   # [L, B, N, C]  L=디코더 layer, N=query, C=클래스
        boxes = pred[f"{self.key}_pred_boxes"]     # [L, B, N, 10] 박스 회귀(10채널)
        # Use last decoder layer for inference.
        logits = logits[-1]      # [B, N, C]   마지막 디코더 layer만 사용(가장 정제된 출력)
        boxes = boxes[-1]        # [B, N, 10]

        scores_all, labels_all = logits.sigmoid().max(dim=-1)   # [B, N], [B, N]  query별 최고 클래스 점수와 그 라벨

        # Prefer per-query track embedding (detection→tracking native).
        query_embed_key = f"{self.key}_query_track_embed"       # query별 track embedding 키
        query_track_embed = pred.get(query_embed_key)  # [B, N, D] or None  (있으면 우선 사용)
        embed_key = f"{self.key}_track_embed"                   # per-pixel track embedding 키(legacy)
        track_embed = pred.get(embed_key)  # [B, D, H, W] (per-pixel, legacy fallback)

        B = boxes.shape[0]               # 배치 크기
        results = []                     # 배치별 결과 dict 모음
        for b in range(B):               # 프레임(샘플)별로 디코딩
            keep = scores_all[b] >= self.score_threshold   # [N] bool: 임계값 넘는 query만 유지
            if keep.sum() > self.max_num:                  # 살아남은 query가 max_num보다 많으면
                topv, topi = scores_all[b].topk(self.max_num)  # 점수 상위 max_num개의 인덱스
                keep = torch.zeros_like(keep)              # mask 초기화
                keep[topi] = True                          # 상위 max_num개만 True로(나머지 버림)

            sb = scores_all[b][keep]     # [K] 유지된 박스들의 점수
            lb = labels_all[b][keep]     # [K] 유지된 박스들의 클래스 라벨
            bb = boxes[b][keep]      # [K, 10]  유지된 박스 회귀값(K=유지 수)

            # [중심보정] heatmap centroid로 중심 재추정 (circle-NMS보다 먼저 → NMS가 보정된 중심 사용)
            if self.centroid_refine and len(bb) > 0:
                hm = pred.get(f"{self.key}_dense_heatmap")
                anchor = pred.get(f"{self.key}_anchor_cell")
                if hm is not None and anchor is not None:
                    bb = self._refine_centers(bb, lb, hm[b], anchor[b][keep])

            # [NMS fix] 클래스별 circle-NMS — 대형 클래스 중복 피크 제거(FP 홍수 → AP 0 방지)
            if self.nms_circle_radii is not None and len(bb) > 0:
                nms_keep = self._circle_nms(bb[:, 0], bb[:, 1], sb, lb, self.nms_circle_radii)
                sb, lb, bb = sb[nms_keep], lb[nms_keep], bb[nms_keep]
                keep_idx = keep.nonzero(as_tuple=True)[0][nms_keep]   # embedding 인덱싱용 원본 인덱스
            else:
                keep_idx = keep.nonzero(as_tuple=True)[0]

            # 10d → 9d (yaw = atan2(sin, cos))
            cx, cy, cz, w, l, h, s, c, vx, vy = bb.unbind(dim=-1)  # 10채널 분해: 중심·크기·yaw(sin,cos)·속도
            yaw = torch.atan2(s, c)      # sin/cos 두 채널 → 단일 yaw 각도(라디안)
            boxes_9d = torch.stack([cx, cy, cz, w, l, h, yaw, vx, vy], dim=-1)  # [K, 9] 최종 박스 (x,y,z,w,l,h,yaw,vx,vy)

            embeds = None                # track embedding(없을 수도)
            # Path 1 (preferred): per-query embedding from detection head.
            if query_track_embed is not None:
                qe_b = query_track_embed[b]  # [N, D]  이 프레임의 query별 embedding
                if len(bb) > 0:
                    embeds = qe_b[keep_idx]   # [K, D] NMS 이후 살아남은 query의 embedding만 추출
                else:
                    embeds = torch.zeros((0, qe_b.shape[-1]),   # 박스 0개면 [0, D] 빈 텐서
                                         dtype=qe_b.dtype, device=qe_b.device)
            # Path 2 (legacy): per-pixel embed, sample at predicted center.
            elif track_embed is not None and len(bb) > 0:
                D = track_embed.shape[1]                        # embedding 차원
                embeds = torch.zeros((len(bb), D), dtype=track_embed.dtype, device=track_embed.device)  # [K, D] 채울 버퍼
                te = track_embed[b]  # [D, H, W]  이 프레임의 per-pixel embedding map
                Ht, Wt = te.shape[-2:]                          # embedding map의 실제 H, W
                cx_np = cx.detach().cpu().numpy()               # 박스 중심 x를 CPU numpy로(픽셀 변환용)
                cy_np = cy.detach().cpu().numpy()               # 박스 중심 y를 CPU numpy로
                for i in range(len(bb)):                        # 박스마다 중심 픽셀에서 embedding 샘플
                    px, py = world_to_bev_pixel(                # 실세계 (cx,cy) → BEV 픽셀 (열,행)
                        float(cx_np[i]), float(cy_np[i]),
                        self.x_min, self.x_max, self.y_min, self.y_max,
                        Wt, Ht,
                    )
                    embeds[i] = te[:, py, px]                   # 해당 픽셀의 D차원 벡터를 박스 embedding으로
            elif track_embed is not None:
                embeds = torch.zeros((0, track_embed.shape[1]), # 박스 0개인데 map만 있으면 [0, D] 빈 텐서
                                     dtype=track_embed.dtype, device=track_embed.device)

            result = {"boxes_3d": boxes_9d, "scores": sb, "labels": lb}  # 프레임 결과: 박스[K,9]/점수[K]/라벨[K]
            if embeds is not None:
                result["embeds"] = embeds          # track embedding 있으면 추가(tracker가 association에 사용)
            results.append(result)                 # 프레임 결과 누적
        return results                             # 배치 길이 B의 결과 리스트 반환
