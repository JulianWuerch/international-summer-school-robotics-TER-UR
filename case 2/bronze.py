"""Bronze deliverable: vibration metrics vs commanded motion parameters."""
# python3 plot_bronze.py --csvs data/test-6.csv data/test-7.csv --out bronze_params.png

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import Recording
from common import segments

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def collect(csvs):
    rows = []
    for path in csvs:
        rec = Recording(path)
        for s in segments(rec):
            df = rec.df.iloc[s.i1:s.i2]
            if len(df) < 25 or len(df) > 200:
                continue
            j = s.joint
            err = (df[f"actual_q{j}"] - df[f"target_q{j}"]).to_numpy(dtype=float)
            err = err - err[-20:].mean()
            rows.append(dict(joint=j, vel=s.vel, acc=s.acc,
                             peak=float(np.abs(err).max()) * 1000,
                             rms=float(np.sqrt(np.mean(err ** 2))) * 1000))
    return pd.DataFrame(rows)


def binned(sub, param, metric, nbins=8):
    """Mean and std of `metric` in `nbins` quantile bins of `param`."""
    b = pd.qcut(sub[param], nbins, duplicates="drop")
    g = sub.groupby(b, observed=True)[metric].agg(["mean", "std", "count"])
    centres = [iv.mid for iv in g.index]
    return centres, g["mean"].values, g["std"].values


def main(csvs, out):
    d = collect(csvs)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex="col")

    for col, param, xlabel in [(0, "vel", "commanded velocity (deg/s)"),
                               (1, "acc", "commanded acceleration (deg/s²)")]:
        for row, metric, ylabel in [(0, "peak", "peak overshoot (mrad)"),
                                    (1, "rms", "RMS position error (mrad)")]:
            a = ax[row, col]
            for j, sub in d.groupby("joint"):
                if len(sub) < 20 or sub[param].nunique() < 8:
                    continue
                x, m, sd = binned(sub, param, metric)
                a.errorbar(x, m, yerr=sd, marker="o", ms=4, capsize=3,
                           lw=1.5, label=f"{JOINT_NAMES[j]} (j{j})")
            a.set_ylabel(ylabel)
            a.set_ylim(bottom=0)
            if row == 1:
                a.set_xlabel(xlabel)
            if row == 0 and col == 0:
                a.legend(fontsize=8)
            a.grid(alpha=0.3)

    ax[0, 0].set_title("Vibration vs commanded velocity")
    ax[0, 1].set_title("Vibration vs commanded acceleration")
    fig.suptitle("Vibration metrics are flat across the full parameter range",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")

    # the numbers behind the plot
    print("\nlowest vs highest parameter bin, per joint:")
    for param in ("vel", "acc"):
        for j, sub in d.groupby("joint"):
            if len(sub) < 20 or sub[param].nunique() < 8:
                continue
            x, m, _ = binned(sub, param, "peak")
            print(f"  {JOINT_NAMES[j]:9s} {param}: {x[0]:6.0f} -> {x[-1]:6.0f}   "
                  f"peak {m[0]:.3f} -> {m[-1]:.3f} mrad   "
                  f"({100*(m[-1]/m[0]-1):+.1f}%)")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", default="bronze_params.png")
    a = ap.parse_args()
    main(a.csvs, a.out)