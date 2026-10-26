"""Collect finished runs into a results table and loss-curve plots.

  python analyze.py --out runs
Writes results.md and plots/<group>.png next to this file.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from run import GROUPS


def load_runs(out_dir):
    runs = defaultdict(list)    # config name -> list of metrics, one per seed
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        if m.get("final") is None:
            continue
        runs[re.sub(r"_s\d+$", "", m["run"])].append(m)
    return runs


def fmt(values):
    if any(v is None for v in values):
        return "diverged"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {(max(values) - min(values)) / 2:.4f}"


def table(runs):
    lines = ["| group | config | params | seeds | val loss (mean ± half-range) | Δ vs baseline | min/run |",
             "|---|---|---|---|---|---|---|"]
    base = [m["final"]["val_loss"] for m in runs.get("baseline", [])]
    base_mean = sum(base) / len(base) if base else None
    for g, members in GROUPS.items():
        for name in members:
            if name not in runs:
                continue
            ms = runs[name]
            vals = [m["final"]["val_loss"] for m in ms]
            ok = [v for v in vals if v is not None]
            delta = f"{sum(ok) / len(ok) - base_mean:+.4f}" if ok and base_mean and len(ok) == len(vals) else ""
            mins = [m["final"].get("train_time_s", 0) / 60 for m in ms]
            lines.append(f"| {g} | {name} | {ms[0]['params']:,} | {len(ms)} | {fmt(vals)} | {delta} | "
                         f"{sum(mins) / len(mins):.1f} |")
    return "\n".join(lines)


def plot(runs, plot_dir):
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    for g, members in GROUPS.items():
        if g == "baseline" or not any(n in runs for n in members):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(["baseline", *members]):
            for j, m in enumerate(runs.get(name, [])):
                h = m["history"]
                ax.plot([r["step"] for r in h], [r["val_loss"] for r in h], color=f"C{i}",
                        alpha=0.9 if j == 0 else 0.5, label=name if j == 0 else None)
        ax.set_xlabel("step")
        ax.set_ylabel("validation loss (nats/token)")
        ax.set_title(g)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{g}.png"), dpi=120)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    runs = load_runs(a.out)
    t = table(runs)
    print(t)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.md"), "w") as f:
        f.write(t + "\n")
    plot(runs, os.path.join(here, "plots"))


if __name__ == "__main__":
    main()
