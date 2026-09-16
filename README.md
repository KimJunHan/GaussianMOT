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
tracking classes), single frame, 448×800 input.

| Detection | | Tracking | |
|---|---|---|---|
| mAP  | 31.6 | AMOTA  | 0.317 |
| NDS  | 39.8 | AMOTP  | 1.353 |
| mATE | 0.565 | Recall | 0.426 |
| mASE | 0.315 | ID-S   | 1559 |
| mAOE | 0.690 | | |
| mAVE | 0.589 | | |
| mAAE | 0.443 | | |

Throughput on a single RTX 3090 (FP32, batch size 1) is 3.2 FPS; restricting
mixed precision to the camera branch raises it to 4.2 FPS at a cost of 0.12 mAP.

### Reproducibility note

The densification stage draws an isotropic perturbation when cloning a Gaussian,
and it runs at inference as well as during training, so the forward pass is
stochastic. Repeated evaluations of one checkpoint differ by up to 0.004 AMOTA
and about 70 identity switches. `tools/evaluate.py` therefore fixes the
inference seed (`eval_seed`, default `0`), and the numbers above are measured
with that default. Changing the seed will move the last digit.

---

## Pretrained checkpoint

The final checkpoint (50 epochs) is distributed separately because of its size:

**Download:** [Google Drive](https://drive.google.com/drive/project/1qB2_98QshIYjgws2UYHPCt7_EJHBilC9?usp=sharing)

The file is `last.ckpt` (about 1.0 GB), the state at the end of epoch 50. Place
it anywhere and pass the path to `checkpoint_path` below.

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
    trainer.devices=2 \
    trainer.max_epochs=50 \
    trainer.effective_batch_size=62 \
    +warm_start_path=<path-to-fusion-checkpoint>
```

The reported model is trained for 50 epochs at a constant learning rate of
2e-4, batch size 1 per device with gradient accumulation to an effective batch
of 62. `effective_batch_size` is divided by `batch_size × devices` to derive
`accumulate_grad_batches`; change the effective size rather than the
accumulation count.

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
| `+camera_amp=bf16` | autocast on the camera branch only (radar and rasterizer stay FP32) |
| `eval_seed=<int>` | inference seed; default `0` |
| `ablation.affinity.enabled=True` | sweep the association weights (`w_e`, `w_p`, `w_v`) |
| `ablation.distance_range.enabled=True` | per-range breakdown |
| `ablation.track_match.enabled=True` | sweep the matching threshold |

The association weights reported in the paper are
`(w_p, w_v, w_e) = (0.5, 0.5, 0)`; the appearance term is inactive, and the
sweep behind that choice is in the paper's appendix.

The distance-range and weather ablations partition the predictions but not the
annotations, so their precision values are not comparable across partitions.
Only true-positive errors such as mATE are meaningful there.

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
tools/                     train.py, evaluate.py
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
