"""Visualise what actually drives the reality gap: configuration, not speed."""
# python3 plot_findings.py --csvs data/test-6.csv data/test-7.csv --out findings.png

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
            rows.append(dict(
                file=path.split("/")[-1], joint=j, dist=s.dist,
                vel=s.vel, acc=s.acc,
                q1_end=float(rec.target_q[s.i1, 1]),
                peak=float(np.abs(err).max()) * 1000,      # mrad
                rms=float(np.sqrt(np.mean(err ** 2))) * 1000))
    return pd.DataFrame(rows)


def main(csvs, out):
    d = collect(csvs)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (a) the headline: peak error vs shoulder angle
    for j, sub in d.groupby("joint"):
                ax[0, 0].scatter(np.cos(sub.q1_end), sub.peak, s=12, alpha=0.6, label=...)
    ax[0, 0].set_xlabel("cos(shoulder angle) ∝ gravity moment arm")
    ax[0, 0].set_ylabel("peak position error (mrad)")
    ax[0, 0].set_title("(a) Gap is driven by arm configuration")
    ax[0, 0].legend(fontsize=8)

    # (b) the contrast: same error vs commanded speed
    for j, sub in d.groupby("joint"):
        ax[0, 1].scatter(sub.vel, sub.peak, s=12, alpha=0.6,
                         label=f"{JOINT_NAMES[j]} (j{j})")
    ax[0, 1].set_xlabel("commanded velocity (deg/s)")
    ax[0, 1].set_ylabel("peak position error (mrad)")
    ax[0, 1].set_title("(b) ...not by commanded speed")
    ax[0, 1].legend(fontsize=8)

    # (c) per-joint magnitude
    order = sorted(d.joint.unique())
    ax[1, 0].bar([JOINT_NAMES[j] for j in order],
                 [d[d.joint == j].peak.mean() for j in order],
                 yerr=[d[d.joint == j].peak.std() for j in order],
                 capsize=4, color="steelblue")
    ax[1, 0].set_ylabel("mean peak error (mrad)")
    ax[1, 0].set_title("(c) Gravity-loaded joints track worst")

    # (d) correlation summary
    vars_ = ["q1_end", "dist", "vel", "acc"]
    cors = {v: [] for v in vars_}
    labels = []
    for j, sub in d.groupby("joint"):
        if len(sub) < 20:
            continue
        labels.append(JOINT_NAMES[j])
        for v in vars_:
            c = sub[v].corr(sub.peak)
            cors[v].append(0.0 if np.isnan(c) else abs(c))
    x = np.arange(len(labels)); w = 0.2
    for k, v in enumerate(vars_):
        ax[1, 1].bar(x + (k - 1.5) * w, cors[v], w, label=v)
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(labels)
    ax[1, 1].set_ylabel("|correlation| with peak error")
    ax[1, 1].set_title("(d) What predicts the gap")
    ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", default="findings.png")
    a = ap.parse_args()
    main(a.csvs, a.out)