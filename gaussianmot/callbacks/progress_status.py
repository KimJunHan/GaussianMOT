"""Live training status callback.

Writes one concise line per `log_every_n_steps` to a status file:
  [HH:MM:SS] e=2 step=350/4000 (8.7%) det=4.21 track=0.78 lr=1.0e-4 gpu=18.3GB eta=2h12m

Designed to be tailed with `tail -f` or Monitor — each line is one notification.
On epoch end, also writes an EPOCH summary line.
"""
from __future__ import annotations

import os                                # (현재 직접 사용은 없으나 경로/환경 관련 표준 import)
import time                              # wall-clock 시간 측정(ETA·sps 계산용)
from datetime import datetime           # 각 status 줄의 [HH:MM:SS] 타임스탬프 생성
from pathlib import Path                # status 파일 경로/부모 디렉터리 처리
from typing import Optional             # _t_run_start 등 None 허용 필드 타입 힌트

import lightning as L                   # PyTorch Lightning — Callback/Trainer/LightningModule
import torch                            # 텐서 metric 값(.item())과 GPU 메모리 조회용


class ProgressStatusCallback(L.Callback):
    # Lightning 훅에 끼어들어 학습 진행상황 한 줄을 status 파일에 append하는 콜백.
    def __init__(
        self,
        status_path: str = "/workspace/outputs/logs/train_status.txt",  # 모니터링이 tail하는 status 파일 경로
        log_every_n_steps: int = 10,    # 몇 opt-step마다 한 줄 기록할지(우리가 보는 step/10728 줄의 간격)
    ):
        super().__init__()
        self.status_path = Path(status_path)                  # 문자열 경로 → Path 객체
        self.log_every_n_steps = max(1, int(log_every_n_steps))  # 최소 1 보장(0이면 ZeroDiv/항상 기록 방지)
        self._step_in_run = 0                                 # 이번 run에서 실제로 센 opt-step 수(누적)
        self._t_run_start: Optional[float] = None             # run 시작 시각(sec). on_train_start에서 채움
        self._last_log_t: Optional[float] = None              # 마지막으로 "기록한" 시각(매 opt-step 아님) — sps/ETA 기준
        self._last_log_step: int = 0                          # 마지막 기록 시점의 global_step
        self._prev_global_step: int = -1                      # 직전 batch_end의 global_step — 새 opt-step 발생 판별용
        # ensure dir exists
        self.status_path.parent.mkdir(parents=True, exist_ok=True)  # 로그 디렉터리 없으면 생성

    # ---- helpers ----
    def _fmt_dur(self, seconds: float) -> str:
        # 초(float) → "1h02m"/"5m03s"/"42s" 형태의 사람이 읽는 ETA 문자열로 포맷.
        seconds = max(0.0, float(seconds))   # 음수 방지(이미 끝난 경우 0으로)
        h = int(seconds // 3600)             # 시 단위
        m = int((seconds % 3600) // 60)      # 분 단위(시 제외 나머지)
        s = int(seconds % 60)                # 초 단위(분 제외 나머지)
        if h > 0:
            return f"{h}h{m:02d}m"           # 1시간 이상이면 시·분만
        if m > 0:
            return f"{m}m{s:02d}s"           # 1분 이상이면 분·초
        return f"{s}s"                       # 1분 미만이면 초만

    def _gpu_mem_gb(self) -> float:
        # 현재 프로세스가 할당한 GPU 메모리를 GB로 반환(없으면 0.0).
        if not torch.cuda.is_available():
            return 0.0                                        # CPU 환경이면 0
        return torch.cuda.memory_allocated() / (1024**3)      # bytes → GB

    def _write(self, line: str):
        # status 파일에 한 줄 append. 파일시스템 오류로 학습이 죽지 않게 best-effort.
        try:
            with open(self.status_path, "a") as f:            # append 모드(헤더 이후 줄들을 누적)
                f.write(line.rstrip() + "\n")                 # 끝 공백 제거 후 개행 붙여 한 줄 기록
        except Exception:
            pass                                              # 디스크/권한 오류는 무시(학습 우선)

    # ---- callbacks ----
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # 학습 시작 시 1회: 타이머/기준 step 초기화 + status 파일을 헤더로 새로 쓴다(truncate).
        self._t_run_start = time.time()                       # run 시작 절대시각 기록(전체 경과·ETA 기준)
        self._last_log_t = self._t_run_start                  # 첫 sps 계산의 기준 시각
        self._last_log_step = int(trainer.global_step)        # 첫 sps 계산의 기준 step(resume 시 0 아님)
        self._prev_global_step = int(trainer.global_step)     # 새 opt-step 판별 기준값
        ts = datetime.now().strftime("%H:%M:%S")              # 헤더용 현재 시각 문자열
        cfg = pl_module.cfg if hasattr(pl_module, "cfg") else None  # 모듈이 들고 있는 hydra cfg(없을 수도)
        max_ep = trainer.max_epochs                           # 총 epoch 수
        bs = cfg.trainer.batch_size if cfg else "?"           # 배치 크기(cfg 없으면 "?")
        lr = cfg.optimizer.lr if cfg else "?"                 # 초기 learning rate
        sched = cfg.scheduler.name if cfg else "?"            # 스케줄러 이름
        # Use truncate for header so old runs don't mix.
        try:
            with open(self.status_path, "w") as f:            # "w" = 기존 내용 비우고 새 run 헤더로 시작
                f.write(f"# Run start {ts}  max_epochs={max_ep}  bs={bs}  lr={lr}  sched={sched}\n")
        except Exception:
            pass                                              # 헤더 기록 실패해도 학습은 진행

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # training_step(=한 batch) 끝마다 호출 — 우리가 모니터링하는 "e=.. step=../10728 det=.. track=.. lr=.. sps=.. eta=.." 줄을 여기서 만든다.
        gs = int(trainer.global_step)                         # 현재까지의 누적 optimizer step 수
        # accumulating — no new opt step yet
        if gs == self._prev_global_step:                      # grad accumulation 중이면 step이 안 늘어남
            return                                            # 같은 step이면 중복 기록 방지하고 종료
        self._prev_global_step = gs                           # 새 opt-step이므로 기준값 갱신
        self._step_in_run += 1                                # 이번 run의 opt-step 카운터 +1
        if self._step_in_run % self.log_every_n_steps != 0:   # N step마다 한 번만 기록
            return                                            # 기록 주기 아니면 종료

        # Pull recent logged metrics
        lm = trainer.callback_metrics                         # self.log()로 쌓인 metric dict
        # train/det_loss_step(=set loss)을 float로. 키 없으면 NaN.
        det = float(lm.get("train/det_loss_step", torch.tensor(float("nan"))).item()) if "train/det_loss_step" in lm else float("nan")
        # train/track_loss_step(=track loss)을 float로. 키 없으면 NaN.
        trk = float(lm.get("train/track_loss_step", torch.tensor(float("nan"))).item()) if "train/track_loss_step" in lm else float("nan")
        lr_v = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")  # 현재 lr(첫 param group)
        gpu = self._gpu_mem_gb()                              # 현재 GPU 메모리(GB)

        # ETA based on rate since last LOG (interval = log_every_n_steps opt steps)
        now = time.time()                                     # 현재 시각
        dt = now - (self._last_log_t or now)                  # 마지막 기록 이후 경과 초(첫 호출이면 0)
        d_steps = gs - self._last_log_step                    # 마지막 기록 이후 진행한 step 수
        sps = d_steps / dt if dt > 0 else 0.0  # opt steps / sec  ← status의 sps= 값(처리 속도)
        total_steps = int(trainer.estimated_stepping_batches) if trainer.estimated_stepping_batches else 0  # 전체 예상 step(=.../10728의 분모)
        remain = max(0, total_steps - gs)                     # 남은 step 수
        eta_sec = remain / sps if sps > 0 else 0.0            # 남은 step / 속도 = ETA(초)

        ts = datetime.now().strftime("%H:%M:%S")              # 줄 앞 [HH:MM:SS]
        pct = (100.0 * gs / total_steps) if total_steps else 0.0  # 진행률(%) — (x.x%) 부분
        line = (                                              # ↓ 모니터링용 한 줄 포맷 조립
            f"[{ts}] e={trainer.current_epoch} step={gs}/{total_steps} ({pct:.1f}%) "  # 시각·epoch·step/총step·진행률
            f"det={det:.3f} track={trk:.3f} lr={lr_v:.2e} gpu={gpu:.1f}GB "             # set loss·track loss·lr·GPU메모리
            f"sps={sps:.2f} eta={self._fmt_dur(eta_sec)}"                               # 초당 step·예상 남은시간
        )
        self._write(line)                                     # status 파일에 append

        self._last_log_t = now                                # 다음 sps 계산을 위해 기준 시각 갱신
        self._last_log_step = gs                              # 다음 sps 계산을 위해 기준 step 갱신

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # epoch 끝마다 1회: 그 epoch 평균 손실들을 EPOCH 요약 줄로 남긴다.
        ts = datetime.now().strftime("%H:%M:%S")              # 타임스탬프
        lm = trainer.callback_metrics                         # metric dict
        det = lm.get("train/det_loss_epoch")                  # epoch 평균 det(set) loss 텐서(없으면 None)
        trk = lm.get("train/track_loss_epoch")                # epoch 평균 track loss 텐서(없으면 None)
        loss = lm.get("train/loss")                           # epoch 평균 total loss 텐서(없으면 None)
        det_s = f"{det.item():.3f}" if isinstance(det, torch.Tensor) else "?"   # 텐서면 값, 아니면 "?"
        trk_s = f"{trk.item():.3f}" if isinstance(trk, torch.Tensor) else "?"
        loss_s = f"{loss.item():.3f}" if isinstance(loss, torch.Tensor) else "?"
        self._write(f"[{ts}] === EPOCH {trainer.current_epoch} END  train/loss={loss_s} det={det_s} track={trk_s} ===")  # 요약 줄 기록

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # validation epoch 끝마다 1회: val 손실(total/set/track)을 VAL 요약 줄로 남긴다.
        ts = datetime.now().strftime("%H:%M:%S")              # 타임스탬프
        lm = trainer.callback_metrics                         # metric dict
        v_loss = lm.get("val/loss")                           # val total loss 텐서
        v_det = lm.get("val/loss/set")                        # val set(det) loss 텐서
        v_trk = lm.get("val/loss/track")                      # val track loss 텐서
        v_loss_s = f"{v_loss.item():.3f}" if isinstance(v_loss, torch.Tensor) else "?"  # 텐서면 값, 아니면 "?"
        v_det_s = f"{v_det.item():.3f}" if isinstance(v_det, torch.Tensor) else "?"
        v_trk_s = f"{v_trk.item():.3f}" if isinstance(v_trk, torch.Tensor) else "?"
        self._write(f"[{ts}] === VAL e={trainer.current_epoch}  val/loss={v_loss_s} set={v_det_s} track={v_trk_s} ===")  # VAL 요약 줄 기록

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # 전체 학습 종료 시 1회: 총 소요시간과 마지막 step을 DONE 줄로 남긴다.
        ts = datetime.now().strftime("%H:%M:%S")              # 타임스탬프
        total = time.time() - (self._t_run_start or time.time())  # run 시작부터 지금까지 총 경과 초
        self._write(f"[{ts}] === TRAINING DONE  total={self._fmt_dur(total)}  final_step={trainer.global_step} ===")  # 종료 줄 기록
