"""Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion, current
gap, position lag), and plot one joint's target vs actual current.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from train_distillation_model_own import DistillModel

from utils import (
    ACC_COL,
    JOINT_NAMES,
    N_JOINTS,
    SCL_COL,
    SCRIPT_COL,
    TIME_COL,
    VEL_COL,
)


class Recording:
    """One recorded run loaded from a ``record.py`` CSV, as numpy arrays."""

    def __init__(self, path: str, df=None):
        if df is None:
            df = pd.read_csv(path)

        self.path = path
        self.df = df

        self.t = df[TIME_COL].to_numpy(dtype=float)

        self.target_q = np.column_stack(
            [df[f"target_q{j}"] for j in range(6)]
        )

        self.actual_q = np.column_stack(
            [df[f"actual_q{j}"] for j in range(6)]
        )

        self.target_qd = np.column_stack(
            [df[f"target_qd{j}"] for j in range(6)]
        )

        self.target_current = np.column_stack(
            [df[f"target_current{j}"] for j in range(6)]
        )

        self.actual_current = np.column_stack(
            [df[f"actual_current{j}"] for j in range(6)]
        )

        self.vel_cmd = (
            df[VEL_COL].to_numpy(dtype=float)
            if VEL_COL in df
            else None
        )

        self.acc_cmd = (
            df[ACC_COL].to_numpy(dtype=float)
            if ACC_COL in df
            else None
        )

        self.scl = (
            df[SCL_COL].to_numpy()
            if SCL_COL in df
            else np.zeros(len(self.t), dtype=int)
        )

        self.script = (
            df[SCRIPT_COL].to_numpy()
            if SCRIPT_COL in df
            else np.zeros(len(self.t), dtype=int)
        )

    @property
    def dt(self) -> float:
        """Median sample period (s)."""
        return float(np.median(np.diff(self.t)))

    def current_gap(self, joint: int) -> np.ndarray:
        """Actual minus target current for one joint (A), per row."""
        return (
            self.actual_current[:, joint]
            - self.target_current[:, joint]
        )


    def plot(self, joint: int = 1, predicted_current=None):
        """Plot target, actual, predicted current and their differences."""
        import matplotlib.pyplot as plt

        name = JOINT_NAMES[joint]

        target = self.target_current[:, joint]
        actual = self.actual_current[:, joint]

        fig, (ax_c, ax_g) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True
        )

        # ---------------------------------------------------------
        # Upper graph: currents
        # ---------------------------------------------------------

        ax_c.plot(
            self.t[:2388],
            target[:2388],
            label="target current",
            lw=2,
        )

        ax_c.plot(
            self.t[:2388],
            actual[:2388],
            label="actual current",
            lw=1,
        )

        if predicted_current is not None:
            ax_c.plot(
                self.t[:2388],
                predicted_current[:2388, joint],
                label="predicted current",
                lw=2,
            )

        ax_c.set_ylabel("current (A)")
        ax_c.set_title(f"{name} joint")
        ax_c.legend(loc="best")

        # ---------------------------------------------------------
        # Lower graph: differences
        # ---------------------------------------------------------

        # Actual - original target
        actual_error = actual - target

        ax_g.plot(
            self.t[:2388],
            actual_error[:2388],
            label="actual - target",
            lw=1,
        )

        # Predicted - original target
        if predicted_current is not None:
            predicted_error = (
                actual - predicted_current[:, joint]
            )

            ax_g.plot(
                self.t[:2388],
                predicted_error[:2388],
                label="actual - predicted",
                lw=1,
            )

        ax_g.axhline(
            0,
            color="grey",
            lw=0.8,
        )

        ax_g.set_ylabel("difference (A)")
        ax_g.set_xlabel("time (s)")
        ax_g.legend(loc="best")

        fig.tight_layout()

        return fig

def predict_target_current(recording, model):
    """
    Calculate predicted target current without modifying the recording.

    Returns:
        np.ndarray with shape (N, 6)
    """
    prediction = model.predict(recording.df)
    predicted_current = prediction["actual_current"]

    if predicted_current.shape != recording.target_current.shape:
        raise ValueError(
            f"Prediction shape {predicted_current.shape} does not match "
            f"target_current shape {recording.target_current.shape}"
        )

    return predicted_current


def main():
    ap = argparse.ArgumentParser(
        description="Per-joint stats and a plot for a recorded run."
    )

    ap.add_argument(
        "--csv",
        default="data/test-4.csv",
        help="recorded run CSV",
    )

    ap.add_argument(
        "--joint",
        type=int,
        default=1,
        help="joint index 0..5 to plot (default 1 = shoulder)",
    )

    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="print stats only",
    )

    ap.add_argument(
        "--predict",
        action="store_true",
        help="Calculate and show the model prediction",
    )

    ap.add_argument(
        "--model",
        default="models.distill.pkl",
        help="Path to the trained model",
    )

    args = ap.parse_args()

    if not 0 <= args.joint < N_JOINTS:
        ap.error(f"--joint must be between 0 and {N_JOINTS - 1}")

    rec = Recording(args.csv)

    # Keep the original target_current untouched.
    predicted_current = None

    if args.predict:
        model = DistillModel.load(args.model)
        predicted_current = predict_target_current(
            rec,
            model,
        )

    print(
        f"{args.csv}  "
        f"({len(rec.t)} rows, dt {rec.dt * 1e3:.1f} ms)"
    )

    # Per-joint summary
    print(
        f"{'joint':10s} "
        f"{'moved':>9s} "
        f"{'gap RMS':>9s} "
        f"{'gap max':>9s} "
        f"{'pos err':>10s}"
    )

    for j in range(N_JOINTS):
        moved = (
            rec.target_q[:, j].max()
            - rec.target_q[:, j].min()
        )

        gap = rec.current_gap(j)

        gap_rms = float(
            np.sqrt(np.mean(gap ** 2))
        )

        gap_max = float(
            np.max(np.abs(gap))
        )

        pos_err = float(
            np.sqrt(
                np.mean(
                    (
                        rec.actual_q[:, j]
                        - rec.target_q[:, j]
                    ) ** 2
                )
            )
        )

        print(
            f"{JOINT_NAMES[j]:10s} "
            f"{moved:8.3f}r "
            f"{gap_rms:8.3f}A "
            f"{gap_max:8.3f}A "
            f"{pos_err * 1e3:7.2f}mrad"
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        rec.plot(
            args.joint,
            predicted_current,
        )

        plt.show()


if __name__ == "__main__":
    main()