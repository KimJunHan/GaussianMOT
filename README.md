# GaussianMOT

Gaussian Splatting for joint 3D object detection and tracking with camera–radar fusion.

Camera images and radar returns are lifted into a common set of Gaussian
primitives, splatted onto a bird's-eye-view (BEV) grid, and decoded by a
center-based detection head together with a tracking head. Each primitive
carries a planar motion vector and an identity embedding in addition to
geometry and semantics, and the radar Doppler measurement is injected as a
velocity prior at two points of the pipeline.

This repository accompanies the paper submitted to IEEE T-ITS and contains the
code, configurations, and evaluation protocol needed to reproduce the reported
numbers.

---

## Results

nuScenes validation split, standard protocol (ten detection classes, seven
tracking classes), 448×800 input, two frames (the current frame and the
previous-frame BEV feature). These are the numbers reported in the paper
(Tables I and II), measured with inference seed 0.

| Detection | | Tracking | |
|---|---|---|---|
| mAP  | 35.0 | AMOTA  | 0.348 |
| NDS  | 42.5 | AMOTP  | 1.31 |
| mATE | 0.539 | Recall | 0.411 |
| mASE | 0.307 | ID-S   | 897 |
| mAOE | 0.681 | | |
| mAVE | 0.511 | | |
| mAAE | 0.465 | | |

Throughput on a single RTX 3090 (batch size 1) is 6.5 FPS (paper Table III).
See [Inference speed](#inference-speed) for the measurement.

### Component analysis

Paper Table IV removes one component at a time at inference of the final
model (no retraining). Each component is switched off with an environment
variable read by the model:

| Variable | Effect |
|---|---|
| `GC_NO_VEL=1` | velocity channels of the augmented Gaussians set to zero |
| `GC_NO_ID=1` | identity channels of the augmented Gaussians set to zero |
| `GC_NO_RADARVEL=1` | radar Doppler input map set to zero |
| `GC_DENSIFY=0` | cloning and splitting omitted |

```bash
GC_NO_RADARVEL=1 python tools/evaluate.py checkpoint_path=<ckpt> device=cuda
```

### Reproducibility note

The densification stage draws a random planar shift when cloning a Gaussian,
and it runs at inference as well as during training, so the forward pass is
stochastic. Repeated evaluations of one checkpoint differ by up to 0.004 AMOTA
and about 70 identity switches. `tools/evaluate.py` therefore fixes the
inference seed (`eval_seed`, default `0`), and all numbers above are measured
with that default. Changing the seed will move the last digit.

---

## Pretrained checkpoint

The reported checkpoint is distributed separately because of its size:

**Download:** [Google Drive](https://drive.google.com/drive/project/1qB2_98QshIYjgws2UYHPCt7_EJHBilC9?usp=sharing)

The file is `epoch_23_loss_3.587.ckpt`, the state at the end of the 24th
(last) training epoch. Place it anywhere and pass the path to
`checkpoint_path` below.

---

## Environment

Python 3.12, PyTorch 2.8 + CUDA 12.9. The differentiable rasterizer under
`gaussianmot/ops/diff-gaussian-rasterization` is an editable install and is
compiled on first use; if you change the CUDA or PyTorch version you must
rebuild it.

The provided `Dockerfile` and `entrypoint.sh` build this environment and
compile the rasterizer on first launch:

```bash
make build
make volume-create      # one-time: persistent conda env volume
make run                # mounts $PATH_TO_NUSCENES at /data/nuscenes
```

To set it up without Docker:

```bash
conda env create -f environment.yml
conda activate gaussianmot
pip install -e gaussianmot/ops/diff-gaussian-rasterization
```

`Makefile` hard-codes `PATH_TO_NUSCENES`; override it with an environment
variable or edit it before building.

Mixed precision is not supported for training: the Point Transformer v3 radar
encoder has no half-precision backward path, and the rasterizer requires
single-precision inputs. Keep `trainer.precision: "32-true"`.

---

## Data

nuScenes `v1.0-trainval` with the six surround-view cameras and five radars.
LiDAR is used only for visualization.

The dataloader reads a preprocessed label layout: one JSON per scene listing
the samples, with the boxes of each sample stored alongside as an `.npz`. Point
`data.data_config.dataset_dir` and `data.data_config.labels_dir` in
`configs/train.yaml` at that root. `configs/train.yaml` also carries the image
resolution (`img_params.final_dim`, 448×800 for the reported model) and the BEV
grid (200×200 over ±50 m).

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py \
    warm_start_path=<path-to-fusion-checkpoint>
```

`configs/train.yaml` holds the recipe of the reported model: 24 epochs, AdamW
at a constant learning rate of 2e-4 (one warm-up epoch), batch size 1 on each
of two GPUs with 31-step gradient accumulation (effective batch 62), FP32.
Training starts from the weights of a camera–radar Gaussian fusion checkpoint
trained without the tracking head (`warm_start_path`; weights only, the
optimizer starts fresh). `effective_batch_size` is divided by
`batch_size × devices` to derive `accumulate_grad_batches`; change the
effective size rather than the accumulation count.

Loss weights (paper Table V): detection 1.0, temporal offset 1.0, selective
Doppler consistency 0.2 (moving returns above 0.8 m/s). The contrastive
identity term is off (`track_weight: 0.0`); the identity channels are kept and
are trained through the detection and offset objectives.

Validation runs once after the final epoch (`check_val_every_n_epoch` is tied
to `max_epochs`).

---

## Evaluation

```bash
python tools/evaluate.py \
    checkpoint_path=<path-to-checkpoint> \
    device=cuda
```

This reports detection and tracking metrics and per-stage runtime. Useful
options:

| Option | Effect |
|---|---|
| `eval_seed=<int>` | inference seed; default `0` |
| `+fast_infer=true` | parallel GroupNorm(1,C), skip the unused dense track head, compile the camera neck (same outputs) |
| `+camera_amp=bf16` | autocast on the camera branch only (radar and rasterizer stay FP32) |
| `ablation.affinity.enabled=True` | sweep the association weights (`w_e`, `w_p`, `w_v`) |
| `ablation.distance_range.enabled=True` | per-range breakdown |
| `ablation.track_match.enabled=True` | sweep the matching threshold |

Association uses position and velocity only, `(w_p, w_v, w_e) = (0.5, 0.5, 0)`,
with matching threshold 0.5, track creation score 0.4, and maintenance score
0.1. These are fixed in `tools/evaluate.py` (`tracker_cfg`).

The distance-range and weather ablations partition the predictions but not the
annotations, so their precision values are not comparable across partitions.
Only true-positive errors such as mATE are meaningful there.

### Inference speed

```bash
python tools/benchmark_latency.py checkpoint_path=<ckpt> --fast --camera-amp bf16
```

This times one current-frame forward pass at batch size 1 (median of 100
runs after 10 warm-up runs). The previous-frame BEV feature is reused from the
preceding frame, as in streaming inference. `--fast` and `--camera-amp bf16`
apply the same changes as `+fast_infer=true` and `+camera_amp=bf16` in
`tools/evaluate.py`. On an RTX 3090 this gives about 150–155 ms (6.5 FPS).
In a matched evaluation of the reported checkpoint these options change
AMOTA by less than 0.001 (0.3413 to 0.3408) and mAP by 0.09.

---

## Layout

```
gaussianmot/
  data/                    dataset, collate, augmentation
  modeling/
    model.py               GaussianMOT: encoders → splat → fuse → decode
    components/
      image_encoder.py     Pixels-to-Gaussians (ResNet-50 + AGP neck)
      radar_encoder.py     Points-to-Gaussians (Point Transformer v3)
      center_head.py       center-based detection head, multi-bin orientation
      ...
  render.py                orthographic BEV rasterizer wrapper
  losses.py                detection, tracking, and Doppler losses
  ops/                     differentiable Gaussian rasterizer (CUDA)
configs/                   Hydra configs: train.yaml, evaluate.yaml
tools/                     train.py, evaluate.py, benchmark_latency.py
Dockerfile, Makefile, entrypoint.sh, environment.yml
```

---

## Acknowledgements and provenance

This work is a derivative of
[GaussianCaR](https://github.com/santimontiel/gaussiancar) by Santiago Montiel
et al., released under the Apache License 2.0, from which the Gaussian encoding
and BEV rendering path originates. The differentiable rasterizer under
`gaussianmot/ops/diff-gaussian-rasterization` derives from
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

This repository is published as a single source release rather than as a fork
of the upstream history, so the commit log does not carry the original
authorship. The upstream `LICENSE` is retained unchanged and the attribution
above stands in its place.

Changes made relative to GaussianCaR: the segmentation task is replaced by
joint 3D detection and tracking; the Gaussian primitive is extended with a
motion vector and an identity embedding; the CMX fuser and DPT segmentation
decoder are replaced by a detection-oriented fusion and decoding path; radar
Doppler is introduced as a velocity prior; and densification is made
ground-truth free so that it runs identically at training and inference.
