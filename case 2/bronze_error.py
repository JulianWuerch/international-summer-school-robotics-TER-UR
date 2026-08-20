"""Bronze: peak and RMS position error vs commanded vel/acc, from one sweep run.

The recording (data/test100.csv) sweeps one parameter at a time on the shoulder:

    phase 1   acc fixed 1.4,  vel 1.2 -> 0.12
    phase 2   vel fixed 1.2,  acc 1.4 -> 0.13

so each panel varies exactly one thing. Every move is split by common.segments
into a motion window [i0, i1] (the arm travelling) and a settle window [i1, i2]
(the arm stopped, ringing down), and the position error

    err = actual_q - target_q

is reduced to a peak and an RMS over each window. The settle error has its
steady-state offset removed first, so it measures the ring and not a constant
bias.

The ring is measured over the first ``RING_FRAC`` of the settle window, not all
of it. The oscillation decays within roughly the first quarter (RMS drops 2-10x
from the first quarter to the last, which sits at the noise floor), so averaging
over the whole window mostly averages silence and buries the signal: measured
over the full window the ring-vs-acceleration correlation reads +0.15, over the
first quarter +0.33, over the first eighth +0.55. The motion window is used
whole, since the arm is moving throughout it.

    python bronze_error.py --csv test100.csv --out bronze_error.png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import Recording
from common import segments
from utils import get_block

MRAD = 1e3          # rad -> mrad, so the numbers read in whole units
TAIL = 20           # rows at the end of a settle window taken as steady state
MIN_ROWS = 20       # skip a window too short to summarise
RING_FRAC = 0.25    # fraction of the settle window the ring is measured over
MIN_RING = 8        # never shrink the ring window below this many rows


def collect(csv: str) -> pd.DataFrame:
    """One row per (move, window): its commanded vel/acc and error summary."""
    rec = Recording(csv)
    err = (get_block(rec.df, "actual_q") - get_block(rec.df, "target_q")) * MRAD
    rows = []
    for s in segments(rec):
        if s.vel is None:
            continue                                  # no registers: nothing to plot against
        for win, lo, hi in (("motion", s.i0, s.i1), ("settle", s.i1, s.i2)):
            e = err[lo:hi, s.joint]
            if len(e) < MIN_ROWS:
                continue
            if win == "settle":
                e = e - e[-TAIL:].mean()              # ring only, not a constant offset
                e = e[:max(int(len(e) * RING_FRAC), MIN_RING)]   # where the ring still is
            rows.append(dict(win=win, vel=s.vel, acc=s.acc, joint=s.joint,
                             peak=float(np.abs(e).max()),
                             rms=float(np.sqrt((e ** 2).mean()))))
    return pd.DataFrame(rows)


def _panel(ax, d, x, win, title, xlabel):
    """Scatter peak and RMS against ``x``, with a mean-per-setting line."""
    sub = d[d.win == win]
    for col, colour, label in (("peak", "#C0392B", "peak"), ("rms", "#2471A3", "RMS")):
        ax.scatter(sub[x], sub[col], s=14, alpha=0.35, color=colour)
        m = sub.groupby(x)[col].mean().sort_index()
        r = sub[x].corr(sub[col])
        ax.plot(m.index, m.values, color=colour, lw=2,
                label=f"{label}   r = {r:+.2f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("position error (mrad)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")


def main(csv: str, out: str):
    d = collect(csv)
    acc_hi = d.acc.max()
    p1 = d[np.isclose(d.acc, acc_hi)]            # acc fixed, vel swept
    p2 = d[~np.isclose(d.acc, acc_hi)]           # vel fixed, acc swept

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    _panel(ax[0, 0], p1, "vel", "motion",
           "(a) While moving", "commanded vel (rad/s)")
    _panel(ax[0, 1], p2, "acc", "motion",
           "(b) While moving", "commanded acc (rad/s²)")
    _panel(ax[1, 0], p1, "vel", "settle",
           "(c) After stopping", "commanded vel (rad/s)")
    _panel(ax[1, 1], p2, "acc", "settle",
           "(d) After stopping", "commanded acc (rad/s²)")

    # Same y-range on each row, so the motion/settle contrast is visible at a glance.
    for row in (0, 1):
        hi = max(ax[row, c].get_ylim()[1] for c in (0, 1))
        for c in (0, 1):
            ax[row, c].set_ylim(0, hi)

    fig.suptitle("Position error vs commanded motion parameters", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    print(f"\n{'window':8s} {'n':>4s} {'peak (mrad)':>16s} {'rms (mrad)':>16s}")
    for win in ("motion", "settle"):
        s = d[d.win == win]
        print(f"{win:8s} {len(s):4d} {s.peak.mean():9.4f} +-{s.peak.std():5.4f} "
              f"{s.rms.mean():9.4f} +-{s.rms.std():5.4f}")
    print("\ncorrelation of error with the swept parameter:")
    for lbl, sub, x in (("vel (acc fixed)", p1, "vel"), ("acc (vel fixed)", p2, "acc")):
        for win in ("motion", "settle"):
            w = sub[sub.win == win]
            print(f"  {lbl:16s} {win:7s}  peak r={w[x].corr(w.peak):+.3f}   "
                  f"rms r={w[x].corr(w.rms):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="test100.csv", help="sweep recording")
    ap.add_argument("--out", default="bronze_error.png", help="output image")
    a = ap.parse_args()
    main(a.csv, a.out)
