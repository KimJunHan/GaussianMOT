import copy
from typing import Any, Dict, Literal

import lightning as L      # PyTorch Lightning: 학습 루프/로깅/옵티마이저 보일러플레이트
import torch
import torch.nn as nn


class GaussianMOTModule(L.LightningModule):
    # GaussianMOT 모델을 감싸는 Lightning 학습 모듈.
    # 역할: forward 실행 → loss/metric 계산 → 로깅 → 옵티마이저/스케줄러 구성,
    #       그리고 temporal(prev frame) 처리와 EMA teacher 갱신을 담당.
    def __init__(
        self,
        cfg: Any = None,        # Hydra 설정(trainer/optimizer/scheduler/task 등)
        model: nn.Module = None,  # GaussianMOT 본체
        losses: Any = None,       # loss 묶음(호출 시 (loss, details, weights) 반환)
        metrics: Any = None,      # metric 모듈 dict
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.losses = losses
        self.metrics = torch.nn.ModuleDict(metrics)   # name→metric 등록(디바이스 자동 이동)

        # Phase 4 prep: EMA teacher (옵션). prev frame forward에 student 대신 사용.
        # config: task.tracking.ema_teacher=true 일 때만 활성화.
        # task.tracking 설정을 안전하게 꺼내 EMA teacher 사용 여부/모멘텀 결정.
        task_cfg = getattr(cfg, "task", None) if cfg is not None else None
        tracking_cfg = getattr(task_cfg, "tracking", None) if task_cfg is not None else None
        self._use_ema_teacher = bool(tracking_cfg.get("ema_teacher", False)) if tracking_cfg else False
        self._ema_momentum = float(tracking_cfg.get("ema_momentum", 0.999)) if tracking_cfg else 0.999
        if self._use_ema_teacher:
            self.teacher = copy.deepcopy(model)          # student를 복제해 teacher 생성
            for p in self.teacher.parameters():
                p.requires_grad_(False)                  # teacher는 학습 안 함(EMA로만 갱신)


    def forward(self, batch: Dict[str, Any]):
        # Lightning forward: 모델 본체로 위임.
        return self.model(batch)

    @torch.no_grad()
    def _update_ema_teacher(self):
        """student → teacher EMA (parameter copy)."""
        # teacher 미사용이면 아무것도 안 함.
        if not self._use_ema_teacher:
            return
        m = self._ema_momentum
        # 파라미터 EMA: teacher = m*teacher + (1-m)*student.
        for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)
        # BN running stats 등 buffer는 직접 동기화 (EMA 안 함)
        for b_t, b_s in zip(self.teacher.buffers(), self.model.buffers()):
            b_t.data.copy_(b_s.data)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 매 학습 step 종료 후 teacher EMA 갱신.
        self._update_ema_teacher()


    def common_step(
        self,
        batch: Dict[str, Any],
        stage: Literal["train", "val"] = "train",
    ) -> Dict[str, Any]:
        # train/val 공통 1 step: (옵션) 정밀도 변환 → (옵션) prev frame forward
        #   → 현재 forward → loss/metric 계산 → 로깅 → {"loss"} 반환.

        # Move batch to device.
        # bf16 설정이면 batch 텐서를 bfloat16으로 캐스팅(리스트 내부 텐서 포함).
        if self.cfg.trainer.precision == "bf16":
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(dtype=torch.bfloat16)
                elif isinstance(v, list):
                    batch[k] = [
                        item.to(dtype=torch.bfloat16)
                        if isinstance(item, torch.Tensor)
                        else item
                        for item in v
                    ]

        # Get batch size.
        B = batch["image"].shape[0]                # 로깅용 배치 크기

        # Phase 3b: temporal — prev frame forward FIRST (no_grad, 단방향).
        # batch에 prev_image가 있고 task.tracking.temporal=True이면 활성화.
        # 현재 frame forward보다 먼저 실행해서 prev BEV feature를 batch에 채워 넣어야
        # 기하 ego-warp(model.forward 내부)이 이를 소비할 수 있다.
        task_cfg = getattr(self.cfg, "task", None)
        tracking_cfg = getattr(task_cfg, "tracking", None) if task_cfg is not None else None
        # temporal 활성 조건: tracking.temporal=True 이고 batch에 prev_image가 있어야 함.
        temporal_enabled = (
            tracking_cfg is not None
            and bool(tracking_cfg.get("temporal", False))
            and "prev_image" in batch
        )
        if temporal_enabled:
            # batch에서 "prev_" 접두사를 떼어 이전 프레임용 sub-batch 구성.
            prev_batch = {
                k[len("prev_"):]: v for k, v in batch.items() if k.startswith("prev_")
            }
            # ema_teacher=true이면 EMA teacher가, 아니면 student가 prev를 forward.
            forward_module = self.teacher if self._use_ema_teacher else self.model
            # ④ symmetric cross-frame: cross_frame_grad=True이고 student(EMA 아님)면 prev 경로에
            #   grad를 허용해 양방향 contrastive 신호(2×, non-stale target)를 만든다.
            #   teacher 사용 시엔 teacher가 frozen이라 grad 무의미 → 기존 no_grad 경로 유지.
            cross_frame_grad = bool(tracking_cfg.get("cross_frame_grad", False)) \
                and not self._use_ema_teacher
            if cross_frame_grad:
                # 옵션 B: track_head_grad_only=True → prev trunk(encoder~decode)는 no_grad,
                #   track_head에만 grad. prev contrastive grad가 공유 trunk를 det에서 끌어당기는 것 방지
                #   (det loss 상승 완화) + prev trunk activation 미보존으로 메모리·backward↓.
                prev_track_out = forward_module.forward_track_only(prev_batch, track_head_grad_only=True)
                for k, v in prev_track_out.items():
                    # bev_feat(ego-warp ② 입력)은 단방향 유지 위해 detach, 나머지(track proj/embed)는
                    # contrastive loss로 grad가 흐르도록 그대로 둔다.
                    batch[f"prev_{k}"] = v.detach() if k == "bev_feat" else v
            else:
                with torch.no_grad():                                  # prev 경로 단방향(grad X)
                    prev_track_out = forward_module.forward_track_only(prev_batch)
                for k, v in prev_track_out.items():
                    # prev_bev_feat은 ego-warp 입력, 나머지(track embed 등)는 tracking loss용.
                    batch[f"prev_{k}"] = v.detach()                    # detach로 prev branch 역전파 차단

            # [multi-frame 8/21] t-2 … t-K 프레임도 no_grad forward해서 bev_feat만 채운다.
            #   dataset이 prev2_*/prev3_* 키를 넣어줄 때만 동작(K=1이면 루프 자체가 안 돎).
            #   tracking loss는 t-1만 쓰므로 여기선 bev_feat 외 출력은 버린다.
            k_extra = 2
            while f"prev{k_extra}_image" in batch:
                extra_batch = {
                    kk[len(f"prev{k_extra}_"):]: vv for kk, vv in batch.items()
                    if kk.startswith(f"prev{k_extra}_")
                }
                with torch.no_grad():
                    extra_out = forward_module.forward_track_only(extra_batch)
                if "bev_feat" in extra_out:
                    batch[f"prev{k_extra}_bev_feat"] = extra_out["bev_feat"].detach()
                k_extra += 1

        # Forward pass (temporal_warp=True면 batch["prev_bev_feat"]+["bev_warp"] 소비).
        # 현재 프레임 forward — 위에서 채운 prev_* 키를 모델 내부 ego-warp가 소비.
        outputs = self(batch)

        # Compute losses (query path만 — heatmap aux 제거됨).
        # outputs["output"](head 결과) + batch(GT)로 총 loss와 세부 항목 계산.
        loss, loss_details, weights = self.losses(outputs["output"], batch)

        # Update metrics.
        # 등록된 모든 metric을 현재 step 출력/GT로 누적 업데이트.
        for k in self.metrics.keys():
            self.metrics[k].update(outputs["output"], batch)

        # Log losses.
        # 총 loss를 epoch 단위로 로깅.
        self.log(
            f"{stage}/loss",
            loss.detach(), on_step=False, on_epoch=True,
            logger=True, batch_size=B,
        )
        # 항목별 세부 loss를 한 번에 로깅.
        self.log_dict(
            {f"{stage}/loss/{k}": v.detach() for k, v in loss_details.items()},
            on_step=False, on_epoch=True,
            logger=True, batch_size=B,
        )

        # Detection / Tracking 집계 — progress bar에 노출
        # 세부 항목에서 detection(set)/tracking(track) loss를 뽑아 prog bar에 표시(없으면 0).
        zero = torch.zeros((), device=loss.device)
        det_loss = loss_details.get("set", zero)
        track_loss = loss_details.get("track", zero)
        self.log(
            f"{stage}/det_loss", det_loss.detach(),
            on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=B,
        )
        self.log(
            f"{stage}/track_loss", track_loss.detach(),
            on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=B,
        )

        # 평균 Gaussian 수 로깅(camera/radar) — 렌더 밀도 모니터링.
        if "num_gaussians_cam" in outputs:
            self.log(
                f"{stage}/num_gaussians_cam",
                outputs["num_gaussians_cam"],
                on_step=False, on_epoch=True,
                logger=True, batch_size=B,
            )
        if "num_gaussians_radar" in outputs:
            self.log(
                f"{stage}/num_gaussians_radar",
                outputs["num_gaussians_radar"],
                on_step=False, on_epoch=True,
                logger=True, batch_size=B,
            )

        # Lightning이 역전파에 쓰는 loss 반환.
        return {
            "loss": loss,
        }


    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        # 학습 step → common_step(train).
        return self.common_step(batch, stage="train")


    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        # 검증 step → common_step(val).
        return self.common_step(batch, stage="val")


    def log_per_epoch_metrics(
        self,
        stage: Literal["train", "val"] = "train",
    ) -> None:
        # epoch 단위로 누적된 metric을 계산·로깅하고 평균(mIoU)까지 기록한 뒤 리셋.
        ious = list()
        for k in self.metrics.keys():
            res = self.metrics[k].compute()                  # 누적값 → 최종 metric
            self.log(f"{stage}/metrics/{k}", res[k], on_epoch=True, logger=True, )
            self.log(f"{stage}/metrics/{k}_max_threshold", res["max_threshold"], on_epoch=True, logger=True, )
            ious.append(res[k])
            self.metrics[k].reset()                          # 다음 epoch 위해 초기화
        if len(ious) > 0:
            self.log(f'{stage}/metrics/mIoU', torch.stack(ious).mean(), on_epoch=True, logger=True, )  # 평균 IoU


    def on_validation_start(self) -> None:
        # 검증 시작 시점에 train metric을 마저 집계/로깅.
        self.log_per_epoch_metrics(stage="train")


    def on_validation_epoch_end(self) -> None:
        # 검증 epoch 종료 시 val metric 집계/로깅.
        self.log_per_epoch_metrics(stage="val")


    def configure_optimizers(self) -> Dict[str, Any]:
        # 옵티마이저/스케줄러 구성. AdamW + (OneCycle/Constant/MultiStep) 만 지원.

        if self.cfg.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")
        
        if self.cfg.scheduler.name == "OneCycleLR":
            # OneCycle: warmup 후 max_lr 도달, 이후 annealing. total_steps는 전체 step 수.
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.scheduler.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.cfg.scheduler.pct_start,
                anneal_strategy=self.cfg.scheduler.anneal_strategy,
                div_factor=self.cfg.scheduler.div_factor,
                final_div_factor=self.cfg.scheduler.final_div_factor,
            )
        elif self.cfg.scheduler.name == "ConstantLR":
            # 상수 LR: 항상 배수 1.0 → lr 고정.
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        elif self.cfg.scheduler.name == "MultiStepLR":
            # CRN-style step decay (+ linear warmup), implemented at step granularity.
            # CRN 스타일 step decay: epoch milestone마다 gamma배 감쇠 + 초반 linear warmup.
            # epoch 단위 milestone을 step 단위로 환산해 LambdaLR로 구현.
            total = float(self.trainer.estimated_stepping_batches)
            spe = max(1.0, total / float(self.cfg.trainer.max_epochs))   # steps per epoch  # epoch당 step 수
            warmup_steps = int(self.cfg.scheduler.get("warmup_epochs", 1) * spe)            # warmup step 수
            milestone_steps = [int(m * spe) for m in self.cfg.scheduler.milestones]  # milestones in EPOCHS  # epoch→step
            gamma = float(self.cfg.scheduler.gamma)                                         # 감쇠 배수

            def _step_decay_lambda(step):
                # warmup 구간: 0→1로 선형 증가.
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                # 이후: 통과한 milestone마다 gamma를 곱해 LR 배수 계산.
                f = 1.0
                for ms in milestone_steps:
                    if step >= ms:
                        f *= gamma
                return f

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _step_decay_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler.name}")

        # step 단위로 스케줄러를 갱신하도록 Lightning에 반환.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
