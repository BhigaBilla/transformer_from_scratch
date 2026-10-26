"""Prepare data, and run the baseline and ablations.

  python run.py prepare --data DATA_DIR
  python run.py train --data DATA_DIR --out RUNS_DIR --group baseline pos --seeds 0 1
  python run.py list

Every ablation changes one thing relative to the baseline (6 layers, d=256, 8 heads,
pre-LN, RoPE, tied embeddings, warmup 200 steps, ~41M training tokens).
"""
import argparse
from dataclasses import replace

from src.config import ModelConfig, TrainConfig

# name -> (model overrides, train overrides)
GROUPS = {
    "baseline": {"baseline": ({}, {})},
    "pos": {
        "pos_none": ({"pos": "none"}, {}),
        "pos_learned": ({"pos": "learned"}, {}),
        "pos_sinusoidal": ({"pos": "sinusoidal"}, {}),
    },
    "norm": {
        "pre_ln_nowarmup": ({}, {"warmup_steps": 0}),
        "post_ln": ({"norm": "post"}, {}),
        "post_ln_nowarmup": ({"norm": "post"}, {"warmup_steps": 0}),
    },
    "heads": {
        "heads_1": ({"n_head": 1}, {}),
        "heads_4": ({"n_head": 4}, {}),
        "heads_16": ({"n_head": 16}, {}),
    },
    # roughly equal parameter count: 12 * n_layer * d_model^2 is ~4.5-4.8M in all three
    "shape": {
        "shape_2x448": ({"n_layer": 2, "d_model": 448, "d_ff": 1792}, {}),
        "shape_12x176": ({"n_layer": 12, "d_model": 176, "d_ff": 704}, {}),
    },
    "extra": {
        "no_tie": ({"tie_weights": False}, {}),
        "no_ffn": ({"use_ffn": False}, {}),
    },
}


def configs(groups, seeds, steps=None):
    for g in groups:
        for name, (m, t) in GROUPS[g].items():
            for seed in seeds:
                mc = replace(ModelConfig(), **m)
                tc = replace(TrainConfig(), **t, seed=seed)
                if steps:
                    tc = replace(tc, steps=steps, eval_every=max(1, steps // 4),
                                 warmup_steps=min(tc.warmup_steps, steps // 10))
                yield f"{name}_s{seed}", mc, tc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "train", "list"])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--group", nargs="+", default=["baseline"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--steps", type=int, default=None, help="override for quick smoke tests")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    if a.cmd == "prepare":
        from src.data import prepare
        prepare(a.data)
    elif a.cmd == "list":
        from src.model import GPT
        for g in GROUPS:
            for name, mc, tc in configs([g], [0]):
                print(f"{g:9s} {name:22s} {GPT(mc).num_params():>10,} params")
    else:
        from src.train import train
        for name, mc, tc in configs(a.group, a.seeds, a.steps):
            train(name, mc, tc, a.data, a.out, a.device)


if __name__ == "__main__":
    main()
