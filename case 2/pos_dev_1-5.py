"""Peak/RMS position error for single-joint sweep recordings (test-1..5)."""
#python3 pos_dev_single.py --csvs data/test-4.csv data/test-5.csv

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
            rows.append(dict(file=path, joint=j, start=s.start, dest=s.dest,
                             dir=np.sign(s.dest - s.start), dist=s.dist,
                             vel=s.vel, acc=s.acc,
                             peak=float(np.abs(err).max()),
                             rms=float(np.sqrt(np.mean(err ** 2)))))
    return pd.DataFrame(rows)


def report(csvs, window):
    d = collect(csvs, window)
    print(f"\n{len(d)} segments  ({window} window)")
    print("\nby (file, joint, dir):")
    g = d.groupby(["file", "joint", "dir"])[["peak", "rms"]].agg(["mean", "std", "count"])
    print(g)

    print("\ncorrelations within each (file, joint, dir):")
    for (f, j, dr), sub in d.groupby(["file", "joint", "dir"]):
        if len(sub) < 20:
            continue
        cv = sub.vel.corr(sub.peak)
        ca = sub.acc.corr(sub.peak)
        print(f"  {f.split('/')[-1]:15s} j{j} dir{dr:+.0f} n={len(sub):3d}  "
              f"corr(vel,peak)={cv:+.3f}  corr(acc,peak)={ca:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--window", choices=("settle", "motion"), default="settle")
    report(ap.parse_args().csvs, ap.parse_args().window)