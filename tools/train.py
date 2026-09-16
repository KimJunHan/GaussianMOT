# --- 표준/외부 라이브러리 import ---
import logging                 # 학습 단계 로그 출력
import random                  # 파이썬 기본 난수 시드 고정용

import hydra                   # config 합성 + _target_ 기반 객체 instantiate
import lightning as L          # PyTorch Lightning: Trainer/LightningModule/DataModule
import numpy as np             # numpy 난수 시드 고정용
import torch                   # 텐서/CUDA 난수 시드, matmul 정밀도 설정
import rootutils               # 프로젝트 루트(.project-root)를 sys.path에 등록(절대 import 보장)
from omegaconf import DictConfig, OmegaConf  # config 타입 + 커스텀 resolver 등록

# .project-root 마커 기준으로 루트를 잡아 pythonpath에 추가 → gaussianmot.* 절대 import 가능.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)  # 이 파일 전용 로거


# (inlined from gaussianmot/utils/config.py)
def register_new_resolvers():
    # config(yaml)의 ${...} 안에서 쓸 커스텀 OmegaConf resolver를 등록한다.
    CUSTOM_RESOLVERS = {
        # mult: 곱셈. 실제 사용처 — decoder out_channels = ${mult:2, ${embed_dims}} = 2×128 = 256.
        "mult": lambda x, y: x * y,
        # last_token: 점(.)으로 구분된 문자열의 마지막 토큰(예: 'a.b.c' → 'c').
        "last_token": lambda x: x.split(".")[-1],
    }

    for name, func in CUSTOM_RESOLVERS.items():
        if not OmegaConf.has_resolver(name):           # 중복 등록 방지(이미 있으면 건너뜀)
            OmegaConf.register_new_resolver(name, func)


def set_seed(seed: int):
    # 재현성을 위해 모든 난수원(Lightning/python/numpy/torch/CUDA)의 시드를 고정.
    L.seed_everything(seed)                # Lightning 통합 시드(dataloader worker 등 포함)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       # 현재 GPU
        torch.cuda.manual_seed_all(seed)   # 모든 GPU(DDP 멀티 GPU)

    # cudnn을 결정론적 모드로 → 같은 입력엔 같은 출력(재현성 우선, 약간의 속도 손해).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 입력 크기별 최적 알고리즘 자동탐색 끔(결정론성 보장)


def instantiate_trainer(cfg: DictConfig) -> L.Trainer:
    """Lightning Trainer를 config 값으로 구성한다."""

    # wandb 등 logger들을 config에서 instantiate (없으면 빈 리스트).
    loggers = []
    if cfg.get("loggers", None):
        for logger_cfg in cfg.loggers:
            loggers.append(hydra.utils.instantiate(logger_cfg))

    # 체크포인트/모니터 등 callback들을 config에서 instantiate (없으면 빈 리스트).
    callbacks = []
    if cfg.get("callbacks", None):
        for callback_cfg in cfg.callbacks:
            callbacks.append(hydra.utils.instantiate(callback_cfg))


    return L.Trainer(
        accelerator=cfg.trainer.accelerator,   # 실제: "cuda"
        devices=cfg.trainer.devices,           # 실제: 3 (GPU 0,1,2 → DDP)
        # 실제: "ddp_find_unused_parameters_true" — 일부 파라미터가 forward에서 안 쓰여도(예: aux head weight0,
        # zero-init aug_proj 초기, ④ cross-frame mixed forward) DDP가 죽지 않게 허용.
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,     # 실제: 24
        precision=cfg.trainer.precision,       # 실제: "32-true"(fp32). module.py의 bf16 분기는 비활성.
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,  # 실제: 0 (학습 시작 전 sanity val 생략)
        # check_val_every_n_epoch=max_epochs(24) → validation은 마지막 epoch에만 1회 실행
        # (매 epoch validation 제거로 학습 가속). ckpt는 train/loss 기준이라 영향 없음.
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        # 유효 배치 = bs × devices × accumulate. accumulate를 역산:
        #   accumulate = effective_batch_size // (batch_size × devices) = 63 // (1×3) = 21.
        #   → bs1 × GPU3 × accum21 = 유효 배치 63 (메모리 한계 bs1을 grad 누적으로 보완).
        accumulate_grad_batches=cfg.trainer.effective_batch_size
            // (cfg.trainer.batch_size * cfg.trainer.devices),
        gradient_clip_val=cfg.trainer.gradient_clip_val,  # 실제: 5.0 (grad-clip — loss 균형/폭주 방지의 그 'budget')
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),  # 1.0=전체 train 사용(디버그 시 축소)
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),      # 1.0=전체 val 사용
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),       # 실제: 10 (10 step마다 로깅)
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """학습 전체 흐름을 실행하는 엔트리포인트.

    흐름: 정밀도/시드/resolver 세팅 → datamodule 구성 → module(모델) 구성
          → (옵션) warm-start 가중치 로드 → Trainer 구성 → trainer.fit.
    """
    # TensorFloat-32 허용: fp32 matmul을 TF32로 가속(정확도 약간↓, 속도↑).
    torch.set_float32_matmul_precision('high')
    set_seed(cfg.seed)              # 실제: seed=42, 전 난수원 고정
    register_new_resolvers()        # ${mult:...} 등 config 접근(아래 instantiate) '전에' 등록되어야 함

    # 1) 데이터 파이프라인 구성 후 fit 스테이지 setup(train/val dataset 준비).
    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # 2) LightningModule(모델+옵티마이저+loss+metric) 구성. cfg를 통째로 넘겨 내부에서 참조.
    log.info(f"Loading module <{cfg.module._target_}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)

    # 3) Warm-start: '모델 가중치만' 로드(옵티마이저/스케줄러/epoch은 새로 시작).
    #    - protected(0.32) ckpt에서 가중치를 가져와 검출을 좋은 지점에서 출발시키는 용도.
    #    - strict=False: 구조가 바뀐 부분(새 GaussMOT/track_head 키=missing, 옛 aux=unexpected)을 허용.
    #    Full Lightning resume(epoch/optimizer/scheduler 포함)이 필요하면 아래 ckpt_path를 사용.
    warm_start_path = cfg.get("warm_start_path", None) or None
    if warm_start_path:
        log.info(f"Warm-starting model weights from: {warm_start_path}")
        ck = torch.load(warm_start_path, map_location="cpu", weights_only=False)  # ckpt 로드(CPU)
        sd = ck["state_dict"] if "state_dict" in ck else ck   # Lightning ckpt면 state_dict 추출
        # [B③] shape 불일치 키 필터: strict=False도 size mismatch에는 에러를 내므로,
        #   구조가 바뀐 head(예: radar 융합으로 vel_head 입력 128→131)는 제외하고 로드(새로 학습).
        model_sd = module.state_dict()
        shape_mismatch = [k for k, v in sd.items()
                          if k in model_sd and model_sd[k].shape != v.shape]
        for k in shape_mismatch:
            del sd[k]
        if shape_mismatch:
            log.warning(f"  shape-mismatch keys dropped: {len(shape_mismatch)} (e.g. {shape_mismatch[:3]})")
        missing, unexpected = module.load_state_dict(sd, strict=False)  # 일부 키 불일치 허용 로드
        if missing:                                            # 모델엔 있는데 ckpt엔 없는 키(새로 학습할 부분)
            log.warning(f"  missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:                                         # ckpt엔 있는데 모델엔 없는 키(폐기된 부분)
            log.warning(f"  unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    # 4) Trainer 구성(DDP/정밀도/accumulate/grad-clip 등).
    log.info("Instantiating the trainer...")
    trainer: L.Trainer = instantiate_trainer(cfg)

    # 5) 학습 시작. ckpt_path가 있으면 'full resume'(epoch/optimizer/scheduler까지 이어받음).
    #    warm_start_path(가중치만)와 ckpt_path(완전 재개)는 서로 다른 용도임에 주의.
    log.info("Starting training...")
    ckpt_path = cfg.get("ckpt_path", None) or None
    if ckpt_path:
        log.info(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)

    log.info("Training completed.")


if __name__ == "__main__":
    main()
