"""Report peak overshoot and RMS position error per move, across recordings."""
# python3 vibration_report.py --csvs data/test-4.csv
# python3 pos_dev.py --csvs data/test-*.csv 2>&1 | head -100

import argparse
import numpy as np
import pandas as pd
from analysis import Recording
from common import segments


def report(csvs):
    rows = []
    for path in csvs:
        rec = Recording(path)
        for s in segments(rec):
            #df = rec.df.iloc[s.i1:s.i2]                 # settle window
            df = rec.df.iloc[s.i0:s.i1]                  # motion window
            if len(df) < 25:
                continue
            j = s.joint                                 # the moving joint
            err = (df[f"actual_q{j}"] - df[f"target_q{j}"]).to_numpy()
            #err = err - err[-20:].mean()                # remove steady-state offset
            rows.append((path, s.joint, s.vel, s.acc, s.dist,
                         float(np.abs(err).max()),
                         float(np.sqrt(np.mean(err ** 2)))))
    d = pd.DataFrame(rows, columns=["file", "joint", "vel", "acc", "dist", "peak", "rms"])
    for (f, j), sub in d.groupby(["file", "joint"]):
        if len(sub) < 20:
            continue
        print(f"{f} joint {j}  n={len(sub):4d}  "
              f"corr(vel,peak)={sub.vel.corr(sub.peak):+.3f}  "
              f"corr(vel,rms)={sub.vel.corr(sub.rms):+.3f}")
    print(f"\n{len(d)} segments scored")
    print("unique vel values:", sorted(d.vel.dropna().unique()))
    print("unique acc values:", sorted(d.acc.dropna().unique()))

    print("\nmean peak/rms grouped by (file, vel):")
    print(d.groupby(["file", "vel"])[["peak", "rms"]].agg(["mean", "count"]))

    print("\nmean peak/rms grouped by (file, acc):")
    print(d.groupby(["file", "acc"])[["peak", "rms"]].agg(["mean", "count"]))
    for f in sorted(d.file.unique()):
        sub = d[d.file == f]
        print(f"\n{f}  n={len(sub)}")
        print("  corr(vel, peak) =", round(sub.vel.corr(sub.peak), 3),
              " corr(vel, rms) =", round(sub.vel.corr(sub.rms), 3))
        print("  corr(acc, peak) =", round(sub.acc.corr(sub.peak), 3),
              " corr(acc, rms) =", round(sub.acc.corr(sub.rms), 3))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    report(ap.parse_args().csvs)