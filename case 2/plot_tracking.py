"""Commanded vs measured joint angle, and their residual, for a few cycles.

The plainest view of the reality gap: what the controller asked the joint to do
(``target_q``), what the joint actually did (``actual_q``), and the difference.
On URSim the two lines are identical and the residual is exactly zero; on a real
arm the measured angle lags while moving and rings briefly after each stop.

One cycle is two movejs (out and back), so ``--cycles 5`` shows ten moves. The
settle window of each move -- where the arm has stopped and is ringing down -- is
shaded, so the ring is easy to separate from the tracking lag during motion.

    python plot_tracking.py --csv test100.csv --joint 1 --cycles 5
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import Recording
from common import segments

MRAD = 1e3          # rad -> mrad for the residual, which is tiny in rad
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def main(csv: str, joint: int, cycles: int, out: str):
    rec = Recording(csv)
    segs = [s for s in segments(rec) if s.joint == joint]
    if not segs:
        raise SystemExit(f"no segments move joint {joint}")

    moves = segs[:cycles * 2]                     # a cycle is out and back
    i0, i1 = moves[0].i0, moves[-1].i2
    t = np.arange(i1 - i0) * rec.dt
    tgt = rec.target_q[i0:i1, joint]
    act = rec.actual_q[i0:i1, joint]
    res = (act - tgt) * MRAD

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax0.plot(t, tgt, color="#2471A3", lw=2.4, label="target (commanded)")
    ax0.plot(t, act, color="#C0392B", lw=1.2, label="actual (measured)")
    ax0.set_ylabel("joint angle (rad)")
    ax0.set_title(f"{JOINT_NAMES[joint]} joint (j{joint}): commanded vs measured, "
                  f"{cycles} cycles")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(alpha=0.25)

    ax1.axhline(0, color="#555", lw=0.8)
    ax1.plot(t, res, color="#8E44AD", lw=1.0)
    ax1.set_ylabel("residual (mrad)")
    ax1.set_xlabel("time (s)")
    ax1.set_title("Residual: actual - target")
    ax1.grid(alpha=0.25)

    # Shade each move's settle window on both panels.
    for k, s in enumerate(moves):
        a, b = (s.i1 - i0) * rec.dt, (s.i2 - i0) * rec.dt
        for ax in (ax0, ax1):
            ax.axvspan(a, b, color="#F39C12", alpha=0.14,
                       label="settle window" if k == 0 else None)
    ax1.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    print(f"\n{len(moves)} moves, {t[-1]:.1f}s, vel {moves[0].vel:.3f} "
          f"acc {moves[0].acc:.3f} -> vel {moves[-1].vel:.3f} acc {moves[-1].acc:.3f}")
    print(f"residual: max |{np.abs(res).max():.4f}| mrad   rms {np.sqrt((res**2).mean()):.4f} mrad")
    mo = np.concatenate([res[s.i0 - i0:s.i1 - i0] for s in moves])
    se = np.concatenate([res[s.i1 - i0:s.i2 - i0] for s in moves])
    print(f"  while moving : peak {np.abs(mo).max():.4f}  rms {np.sqrt((mo**2).mean()):.4f} mrad")
    print(f"  after stopping: peak {np.abs(se).max():.4f}  rms {np.sqrt((se**2).mean()):.4f} mrad")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="test100.csv")
    ap.add_argument("--joint", type=int, default=1, help="joint index (1 = shoulder)")
    ap.add_argument("--cycles", type=int, default=5, help="cycles to show (2 moves each)")
    ap.add_argument("--out", default="tracking.png")
    a = ap.parse_args()
    main(a.csv, a.joint, a.cycles, a.out)
