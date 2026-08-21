"""Compare two recorded runs of the same motion: baseline vs optimized.

Both runs trace the same path, so any difference in position error comes from the
commanded vel/acc. Each move is split into a motion window [i0, i1] (arm
travelling) and a settle window [i1, i2] (arm stopped, ringing down), and the
error ``actual_q - target_q`` is summarised over each.

Only segments on the joint that actually traces the motion are compared, and only
those whose travel distance appears in both runs, so an approach move recorded in
one run and not the other cannot skew the totals.

    python plot_compare.py --baseline baseline_real.csv --optimized optimized_real.csv --joint 0
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

MRAD = 1e3
TAIL = 20           # rows at the end of a settle window taken as steady state
RING_FRAC = 0.25    # the ring decays early; measure it there, not over the whole window
MIN_RING = 8

BASE_C = "#5D6D7E"
OPT_C = "#1E8449"
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def collect(csv: str, joint: int, measure: int = None) -> pd.DataFrame:
    """Per-segment error summary.

    ``joint`` selects which segments to use (those whose widest-travel joint it
    is, which is how ``common.segments`` labels them). ``measure`` is the joint
    the error is read from, defaulting to ``joint``. They differ when several
    joints move together: a movej drives joints 0, 1 and 2 along one profile, so
    every segment is labelled by joint 0 while joints 1 and 2 move on all of
    them and can be measured on the same segments.
    """
    if measure is None:
        measure = joint
    rec = Recording(csv)
    err = (get_block(rec.df, "actual_q") - get_block(rec.df, "target_q")) * MRAD
    rows = []
    for s in segments(rec):
        if s.joint != joint:
            continue                                   # e.g. the approach move
        m = err[s.i0:s.i1, measure]
        e = err[s.i1:s.i2, measure]
        if len(m) < 10 or len(e) < TAIL:
            continue
        e = e - e[-TAIL:].mean()
        e = e[:max(int(len(e) * RING_FRAC), MIN_RING)]
        rows.append(dict(dist=round(s.dist, 2),
                         motion_peak=np.abs(m).max(),
                         motion_rms=np.sqrt((m ** 2).mean()),
                         ring_peak=np.abs(e).max(),
                         ring_rms=np.sqrt((e ** 2).mean()),
                         cycle=(s.i1 - s.i0) * rec.dt))
    return pd.DataFrame(rows)


def moving_joints(csv: str, min_travel: float = 0.5) -> list[int]:
    """Joints that actually travel over the run, so static ones are not plotted."""
    rec = Recording(csv)
    travel = np.abs(np.diff(rec.target_q, axis=0)).sum(axis=0)
    return [j for j in range(travel.shape[0]) if travel[j] >= min_travel]


def main(base_csv, opt_csv, joint, out):
    B, O = collect(base_csv, joint), collect(opt_csv, joint)
    shared = sorted(set(B.dist) & set(O.dist))
    B, O = B[B.dist.isin(shared)], O[O.dist.isin(shared)]
    if B.empty or O.empty:
        raise SystemExit(f"no comparable joint-{joint} segments in both runs")

    metrics = [("motion_peak", "motion\npeak"), ("motion_rms", "motion\nRMS"),
               ("ring_peak", "ring\npeak"), ("ring_rms", "ring\nRMS")]

    fig, ax = plt.subplots(1, 3, figsize=(14, 5),
                           gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # (a) grouped bars: every error measure, baseline vs optimized
    x = np.arange(len(metrics))
    bw = 0.36
    bv = [B[k].mean() for k, _ in metrics]
    ov = [O[k].mean() for k, _ in metrics]
    ax[0].bar(x - bw / 2, bv, bw, color=BASE_C, label="baseline")
    ax[0].bar(x + bw / 2, ov, bw, color=OPT_C, label="optimized")
    # Label the change the way a reader expects: a drop in error shows as -N%.
    for i, (b, o) in enumerate(zip(bv, ov)):
        ax[0].text(i, max(b, o) * 1.03, f"{100*(o-b)/b:+.1f}%",
                   ha="center", fontsize=9,
                   color=OPT_C if o < b else "#C0392B")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax[0].set_ylabel("position error (mrad)")
    ax[0].set_title("(a) Error, all measures")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25, axis="y")

    # (b) ring RMS split by move size: does the gain hold for every move?
    w = 0.36
    xi = np.arange(len(shared))
    ax[1].bar(xi - w / 2, [B[B.dist == d].ring_rms.mean() for d in shared], w,
              color=BASE_C, label="baseline")
    ax[1].bar(xi + w / 2, [O[O.dist == d].ring_rms.mean() for d in shared], w,
              color=OPT_C, label="optimized")
    ax[1].set_xticks(xi)
    ax[1].set_xticklabels([f"{d} rad" for d in shared])
    ax[1].set_xlabel("move distance")
    ax[1].set_ylabel("ring RMS (mrad)")
    ax[1].set_title("(b) Ring, by move size")
    ax[1].grid(alpha=0.25, axis="y")

    # (c) the thing the error was traded against
    ax[2].bar([0], [B.cycle.mean()], 0.5, color=BASE_C)
    ax[2].bar([1], [O.cycle.mean()], 0.5, color=OPT_C)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["baseline", "optimized"])
    ax[2].set_ylabel("move time (s)")
    ax[2].set_title("(c) Cycle time")
    ax[2].text(0.5, max(B.cycle.mean(), O.cycle.mean()) * 1.02,
               f"{100*(O.cycle.mean()-B.cycle.mean())/B.cycle.mean():+.1f}%",
               ha="center", fontsize=10)
    ax[2].grid(alpha=0.25, axis="y")

    fig.suptitle(f"Real UR5: baseline vs optimized  "
                 f"({len(B)} vs {len(O)} moves, joint {joint})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}\n")

    print(f"{'':20s}{'baseline':>12s}{'optimized':>12s}{'change':>11s}")
    for k, lbl in metrics + [("cycle", "move time (s)")]:
        b, o = B[k].mean(), O[k].mean()
        print(f"{lbl.replace(chr(10),' '):20s}{b:12.4f}{o:12.4f}{100*(o-b)/b:+10.1f}%")
    print("\n(negative = optimized is lower = better)")


def table(base_csv, opt_csv, seg_joint):
    """analysis.py's per-joint table, restricted to the moves both runs share.

    A run recorded from a different starting pose carries an extra approach move
    into the script's first waypoint. That move is not part of the motion being
    compared, and it swings joints the script itself never drives, so leaving it
    in makes the two runs look different in the ``moved`` column and the error of
    a joint that only moved in one of them meaningless. Keeping only segments on
    ``seg_joint`` drops it, and the ``moved`` column then matches between runs,
    which is what makes the comparison like-for-like.
    """
    def summarise(csv):
        rec = Recording(csv)
        segs = [s for s in segments(rec) if s.joint == seg_joint]
        err = get_block(rec.df, "actual_q") - get_block(rec.df, "target_q")
        blocks, travel, dur = [], np.zeros(len(JOINT_NAMES)), 0.0
        for s in segs:
            blocks.append(np.abs(err[s.i0:s.i2]) * MRAD)
            travel += np.abs(np.diff(rec.target_q[s.i0:s.i1], axis=0)).sum(axis=0)
            dur += (s.i2 - s.i0) * rec.dt
        E = np.concatenate(blocks, axis=0)
        return E, travel, dur, len(segs)

    (Eb, tb, db, nb), (Eo, to, do, no) = summarise(base_csv), summarise(opt_csv)
    print(f"matched moves: {nb} baseline, {no} optimized "
          f"(approach move excluded)\n")
    print(f"{'joint':10s}{'moved (base)':>14s}{'moved (opt)':>13s}"
          f"{'baseline':>11s}{'optimized':>11s}{'change':>10s}")
    for j, name in enumerate(JOINT_NAMES):
        b = np.sqrt((Eb[:, j] ** 2).mean())
        o = np.sqrt((Eo[:, j] ** 2).mean())
        ch = f"{100*(o-b)/b:+9.1f}%" if b > 1e-9 else f"{'-':>10s}"
        print(f"{name:10s}{tb[j]:13.3f}r{to[j]:12.3f}r"
              f"{b:9.2f}mr{o:9.2f}mr{ch}")
    print(f"\n{'run time':10s}{'':27s}{db:9.1f}s {do:9.1f}s"
          f"{100*(do-db)/db:+9.1f}%")
    print("\n(negative = optimized is lower = better)")


def main_joints(base_csv, opt_csv, seg_joint, out):
    """Per-joint position error, baseline vs optimized, over the matched moves.

    One bar pair per joint, RMS over each move's full window. Restricted to
    segments on ``seg_joint`` so the approach move is excluded and both runs
    cover the same motion.
    """
    def summarise(csv):
        rec = Recording(csv)
        segs = [s for s in segments(rec) if s.joint == seg_joint]
        err = get_block(rec.df, "actual_q") - get_block(rec.df, "target_q")
        E = np.concatenate([np.abs(err[s.i0:s.i2]) * MRAD for s in segs], axis=0)
        dur = sum((s.i2 - s.i0) * rec.dt for s in segs)
        return np.sqrt((E ** 2).mean(axis=0)), dur, len(segs)

    b, db, nb = summarise(base_csv)
    o, do, no = summarise(opt_csv)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(JOINT_NAMES))
    w = 0.36
    ax.bar(x - w / 2, b, w, color=BASE_C, label="baseline (acc 300)")
    ax.bar(x + w / 2, o, w, color=OPT_C, label="optimized (acc 200)")
    for i, (bi, oi) in enumerate(zip(b, o)):
        if bi > 1e-9:
            ax.text(i, max(bi, oi) * 1.04, f"{100*(oi-bi)/bi:+.1f}%",
                    ha="center", fontsize=9,
                    color=OPT_C if oi <= bi else "#C0392B")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES)
    ax.set_ylabel("position error, RMS (mrad)")
    ax.set_title(f"Real UR5: position error per joint  "
                 f"({nb} moves, cycle time unchanged: {db:.1f}s vs {do:.1f}s)")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}\n")
    print(f"{'joint':10s}{'baseline':>11s}{'optimized':>11s}{'change':>10s}")
    for j, name in enumerate(JOINT_NAMES):
        ch = f"{100*(o[j]-b[j])/b[j]:+9.1f}%" if b[j] > 1e-9 else f"{'-':>10s}"
        print(f"{name:10s}{b[j]:9.2f}mr{o[j]:9.2f}mr{ch}")
    print(f"\ncycle time: {db:.1f}s -> {do:.1f}s ({100*(do-db)/db:+.1f}%)")
    print("(negative = optimized is lower = better)")


def main_all(base_csv, opt_csv, seg_joint, out):
    """One panel per moving joint, measured over the same segments."""
    joints = sorted(set(moving_joints(base_csv)) & set(moving_joints(opt_csv)))
    if not joints:
        raise SystemExit("no joint travels far enough in both runs")

    metrics = [("motion_peak", "motion\npeak"), ("motion_rms", "motion\nRMS"),
               ("ring_peak", "ring\npeak"), ("ring_rms", "ring\nRMS")]
    fig, ax = plt.subplots(1, len(joints), figsize=(4.6 * len(joints), 5),
                           squeeze=False)
    ax = ax[0]
    summary = []
    for k, j in enumerate(joints):
        B = collect(base_csv, seg_joint, measure=j)
        O = collect(opt_csv, seg_joint, measure=j)
        shared = sorted(set(B.dist) & set(O.dist))
        B, O = B[B.dist.isin(shared)], O[O.dist.isin(shared)]
        x = np.arange(len(metrics))
        bw = 0.36
        bv = [B[c].mean() for c, _ in metrics]
        ov = [O[c].mean() for c, _ in metrics]
        ax[k].bar(x - bw / 2, bv, bw, color=BASE_C, label="baseline")
        ax[k].bar(x + bw / 2, ov, bw, color=OPT_C, label="optimized")
        for i, (b, o) in enumerate(zip(bv, ov)):
            ax[k].text(i, max(b, o) * 1.03, f"{100*(o-b)/b:+.1f}%", ha="center",
                       fontsize=8, color=OPT_C if o < b else "#C0392B")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([l for _, l in metrics], fontsize=8)
        ax[k].set_title(f"{JOINT_NAMES[j]} (j{j})")
        ax[k].grid(alpha=0.25, axis="y")
        if k == 0:
            ax[k].set_ylabel("position error (mrad)")
            ax[k].legend(fontsize=8)
        summary.append((j, bv, ov))

    fig.suptitle("Real UR5: baseline vs optimized, per joint", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}\n")
    print(f"{'joint':10s}{'measure':14s}{'baseline':>11s}{'optimized':>11s}{'change':>10s}")
    for j, bv, ov in summary:
        for (c, lbl), b, o in zip(metrics, bv, ov):
            print(f"{JOINT_NAMES[j]:10s}{lbl.replace(chr(10),' '):14s}"
                  f"{b:11.4f}{o:11.4f}{100*(o-b)/b:+9.1f}%")
    print("\n(negative = optimized is lower = better)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", default="baseline_real.csv")
    ap.add_argument("--optimized", default="optimized_real.csv")
    ap.add_argument("--joint", type=int, default=0,
                    help="joint whose segments define the moves being compared")
    ap.add_argument("--all-joints", action="store_true",
                    help="one panel per moving joint instead of a single joint")
    ap.add_argument("--table", action="store_true",
                    help="print the per-joint table over matched moves, no plot")
    ap.add_argument("--joints", action="store_true",
                    help="one bar pair per joint: position error, matched moves")
    ap.add_argument("--out", default="compare.png")
    a = ap.parse_args()
    if a.table:
        table(a.baseline, a.optimized, a.joint)
    elif a.joints:
        main_joints(a.baseline, a.optimized, a.joint, a.out)
    elif a.all_joints:
        main_all(a.baseline, a.optimized, a.joint, a.out)
    else:
        main(a.baseline, a.optimized, a.joint, a.out)
