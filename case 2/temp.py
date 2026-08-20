"""Evaluate the RL reward across a grid of [vel, acc], before training."""
import argparse
import numpy as np
import pandas as pd
from analysis import Recording
from train_rla import GapEnv
from train_distillation_model import DistillModel
from metrics import CurrentGapMetric, PositionErrorMetric
from preprocess import default_preprocess


def main(csv, model_path, metric_name, move_index):
    model = DistillModel.load(model_path)
    metric = PositionErrorMetric() if metric_name == "position" else CurrentGapMetric()
    pre = default_preprocess()
    rec = Recording(csv)
    env = GapEnv(model, metric, rec, pre=pre)
    move = env.targets[move_index]

    vels = [20, 40, 60, 80, 100, 140, 180]
    accs = [40, 100, 200, 300, 400, 500, 600]

    print(f"metric: {metric_name}   move {move_index}   "
          f"joint {move.joint}  dist {move.dist:.3f} rad\n")

    rows = []
    for v in vels:
        for a in accs:
            score, cycle = env.score(move, v, a)
            rows.append(dict(vel=v, acc=a, score=score, cycle=cycle,
                             obj=score + cycle))
    d = pd.DataFrame(rows)

    for name in ("score", "cycle", "obj"):
        print(f"--- {name} ---")
        print(d.pivot(index="vel", columns="acc", values=name).round(4))
        print()

    s, c = d.score, d.cycle
    print(f"score  range: {s.min():.4f} to {s.max():.4f}   spread {s.max()-s.min():.4f}")
    print(f"cycle  range: {c.min():.4f} to {c.max():.4f}   spread {c.max()-c.min():.4f}")
    print(f"cycle spread / score spread = {(c.max()-c.min())/(s.max()-s.min()):.1f}x")
    best = d.loc[d.obj.idxmin()]
    print(f"\noptimum: vel={best.vel:.0f} acc={best.acc:.0f}  "
          f"score={best.score:.4f} cycle={best.cycle:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sim_to_real.csv")
    ap.add_argument("--model", default="models/distill.pkl")
    ap.add_argument("--metric", choices=("current", "position"), default="current")
    ap.add_argument("--move", type=int, default=0)
    a = ap.parse_args()
    main(a.csv, a.model, a.metric, a.move)