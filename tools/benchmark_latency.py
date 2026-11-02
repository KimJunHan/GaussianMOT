"""추론 속도 측정 (논문 Table III의 FPS).

측정 조건: RTX 3090, batch size 1, 현재 프레임 forward 1회.
이전 프레임 BEV feature는 직전 프레임 forward 결과를 재사용하므로(스트리밍 추론),
prev 프레임 인코딩은 측정 구간에서 제외한다. 카메라 분기만 bf16 autocast,
--fast는 evaluate.py의 +fast_infer=true와 같은 변경(수식 동일)을 적용한다.

사용 예:
    python tools/benchmark_latency.py checkpoint_path=<ckpt> --fast --camera-amp bf16
"""
import argparse
import statistics as st
import types

import hydra
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def register_new_resolvers():
    for name, func in {"mult": lambda x, y: x * y,
                       "last_token": lambda x: x.split(".")[-1]}.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, func)


def apply_fast_infer(core):
    # evaluate.py의 +fast_infer=true와 동일: GroupNorm(1,C) 병렬화, dense track_head 생략, neck compile.
    def _gn1_forward(self, x):
        xf = x.float()
        var, mean = torch.var_mean(xf, dim=(1, 2, 3), keepdim=True, correction=0)
        y = (xf - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y.to(x.dtype)
    for m in core.modules():
        if isinstance(m, torch.nn.GroupNorm) and m.num_groups == 1:
            m.forward = types.MethodType(_gn1_forward, m)
    core.track_head = None
    core.image_encoder.neck = torch.compile(core.image_encoder.neck, dynamic=False)


def apply_camera_amp(core, dtype):
    # 카메라 분기만 autocast, 출력은 fp32로 되돌림 (radar PTv3·rasterizer는 fp32 유지).
    orig = core.forward_features_camera

    def _cam_amp(batch):
        with torch.autocast("cuda", dtype=dtype):
            out = orig(batch)
        return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in out.items()}
    core.forward_features_camera = _cam_amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*", help="evaluate.yaml overrides (checkpoint_path=... 필수)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--camera-amp", choices=["bf16", "fp16"], default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    register_new_resolvers()
    with hydra.initialize_config_dir(config_dir=str(rootutils.find_root(__file__, ".project-root") / "configs"),
                                     version_base="1.3"):
        cfg = hydra.compose(config_name="evaluate", overrides=args.overrides)

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    module = hydra.utils.instantiate(cfg.module)
    sd = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(sd.get("state_dict", sd), strict=False)
    module = module.cuda().eval()
    core = module.model
    torch.manual_seed(int(cfg.get("eval_seed", 0)))

    if args.fast:
        apply_fast_infer(core)
    if args.camera_amp:
        apply_camera_amp(core, {"bf16": torch.bfloat16, "fp16": torch.float16}[args.camera_amp])

    # val 첫 batch의 첫 샘플을 batch size 1로 사용.
    def first(v):
        if torch.is_tensor(v):
            return v[:1].cuda()
        if isinstance(v, list):
            return [v[0].cuda() if torch.is_tensor(v[0]) else v[0]]
        return v
    batch = {k: first(v) for k, v in next(iter(dm.val_dataloader())).items()}
    prev_batch = {k[len("prev_"):]: v for k, v in batch.items() if k.startswith("prev_")}

    times = []
    with torch.no_grad():
        # 직전 프레임 forward 결과(prev BEV)는 스트리밍 추론에서 이미 계산되어 있으므로 1회만 만든다.
        prev_out = core.forward_track_only(prev_batch)
        cur = dict(batch)
        for k, v in prev_out.items():
            if torch.is_tensor(v):
                cur[f"prev_{k}"] = v
        for i in range(args.warmup + args.runs):
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            module(dict(cur))
            e.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(s.elapsed_time(e))

    ms = st.median(times)
    print(f"current-frame forward: median {ms:.1f} ms over {args.runs} runs ({1000.0 / ms:.2f} FPS)")


if __name__ == "__main__":
    main()
