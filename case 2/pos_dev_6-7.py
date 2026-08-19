"""Peak/RMS position error for multi-joint recordings (test-6,7).

Direction alone is insufficient here: the same joint moves from many different
configurations, and gravity torque depends on pose, not just travel sign. So we
bin by start angle and report per joint.
"""
# python3 pos_dev_multi.py --csvs data/test-6.csv data/test-7.csv


import argparse
import numpy as np
import pandas as pd
from analysis import Recording
from common import segments


def collect(csvs, window="settle"):
    rows = []
    for path in csvs:
        rec = Recording(path)
        for s in segments(rec):
            lo, hi = (s.i1, s.i2) if window == "settle" else (s.i0, s.i1)
            df = rec.df.iloc[lo:hi]
            if len(df) < 25 or len(df) > 200:
                continue
            j = s.joint
            err = (df[f"actual_q{j}"] - df[f"target_q{j}"]).to_numpy(dtype=float)
            err = err - err[-20:].mean()
            # full configuration at the moment motion ends
            cfg = {f"q{k}_end": float(rec.target_q[s.i1, k]) for k in range(6)}
            rows.append(dict(file=path, joint=j, start=s.start, dest=s.dest,
                             dir=np.sign(s.dest - s.start), dist=s.dist,
                             vel=s.vel, acc=s.acc,
                             peak=float(np.abs(err).max()),
                             rms=float(np.sqrt(np.mean(err ** 2))), **cfg))
    return pd.DataFrame(rows)


def report(csvs, window):
    d = collect(csvs, window)
    print(f"\n{len(d)} segments  ({window} window)")

    print("\nper joint summary:")
    print(d.groupby("joint")[["peak", "rms", "dist"]].agg(["mean", "std", "count"]))

    print("\nper joint, binned by start angle (5 bins):")
    for j, sub in d.groupby("joint"):
        if len(sub) < 20:
            continue
        sub = sub.copy()
        sub["start_bin"] = pd.qcut(sub.start, 5, duplicates="drop")
        print(f"\n  joint {j}:")
        print(sub.groupby("start_bin", observed=True)[["peak", "rms"]].agg(["mean", "count"]))

    print("\nper joint correlations (peak vs each variable):")
    for j, sub in d.groupby("joint"):
        if len(sub) < 20:
            continue
        print(f"  joint {j} n={len(sub):3d}  "
              f"start={sub.start.corr(sub.peak):+.3f}  "
              f"dist={sub.dist.corr(sub.peak):+.3f}  "
              f"vel={sub.vel.corr(sub.peak):+.3f}  "
              f"acc={sub.acc.corr(sub.peak):+.3f}  "
              f"q1_end={sub.q1_end.corr(sub.peak):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--window", choices=("settle", "motion"), default="settle")
    a = ap.parse_args()
    report(a.csvs, a.window)