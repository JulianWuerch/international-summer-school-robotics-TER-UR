"""Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion,
current gap, position lag), and plot one joint's target vs actual position.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).

With ``--predict``, the trained position model is used to predict
actual_q0 ... actual_q5 from:

    target_q
    target_qd
    target_qdd
    target_current
    vel
    acc

Example:

    python analysis.py --csv data/test-4.csv --joint 1 --predict

"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import train_distillation_model_position
from train_distillation_model_position import (
    DistillModel,
    OwnModel,
    SimpleNN,
)

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

        # ---------------------------------------------------------
        # Target joint positions
        # ---------------------------------------------------------

        self.target_q = np.column_stack(
            [df[f"target_q{j}"] for j in range(N_JOINTS)]
        )

        # ---------------------------------------------------------
        # Actual joint positions
        # ---------------------------------------------------------

        self.actual_q = np.column_stack(
            [df[f"actual_q{j}"] for j in range(N_JOINTS)]
        )

        # ---------------------------------------------------------
        # Target joint velocities
        # ---------------------------------------------------------

        self.target_qd = np.column_stack(
            [df[f"target_qd{j}"] for j in range(N_JOINTS)]
        )

        # ---------------------------------------------------------
        # Target joint accelerations
        #
        # These are already present in the CSV and are also used
        # directly by the position model.
        # ---------------------------------------------------------

        self.target_qdd = np.column_stack(
            [df[f"target_qdd{j}"] for j in range(N_JOINTS)]
        )

        # ---------------------------------------------------------
        # Target current
        # ---------------------------------------------------------

        self.target_current = np.column_stack(
            [df[f"target_current{j}"] for j in range(N_JOINTS)]
        )

        # ---------------------------------------------------------
        # Actual current
        #
        # Kept because the existing statistics still report
        # current gap.
        # ---------------------------------------------------------

        self.actual_current = np.column_stack(
            [df[f"actual_current{j}"] for j in range(N_JOINTS)]
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
            else np.zeros(
                len(self.t),
                dtype=int,
            )
        )

        self.script = (
            df[SCRIPT_COL].to_numpy()
            if SCRIPT_COL in df
            else np.zeros(
                len(self.t),
                dtype=int,
            )
        )

    @property
    def dt(self) -> float:
        """Median sample period (s)."""
        return float(
            np.median(
                np.diff(self.t)
            )
        )

    def current_gap(
        self,
        joint: int,
    ) -> np.ndarray:
        """Actual minus target current for one joint (A), per row."""
        return (
            self.actual_current[:, joint]
            - self.target_current[:, joint]
        )

    def position_error(
        self,
        joint: int,
    ) -> np.ndarray:
        """Actual minus target position for one joint (rad), per row."""
        return (
            self.actual_q[:, joint]
            - self.target_q[:, joint]
        )

    def predicted_position_error(
        self,
        predicted_q: np.ndarray,
        joint: int,
    ) -> np.ndarray:
        """Actual minus predicted position for one joint (rad), per row."""
        return (
            self.actual_q[:, joint]
            - predicted_q[:, joint]
        )

    def plot(
        self,
        joint: int = 1,
        predicted_q=None,
    ):
        """
        Plot target, actual, predicted position and their differences.
        """

        import matplotlib.pyplot as plt

        name = JOINT_NAMES[joint]

        target = self.target_q[:, joint]
        actual = self.actual_q[:, joint]

        # ---------------------------------------------------------
        # Limit plot to the same first 2388 samples as the
        # original analysis.
        # ---------------------------------------------------------
        start = 2388
        end = 2388 * 2
        n_plot = min(
            end - start,
            len(self.t),
        )

        t = self.t[start:end]

        fig, (ax_q, ax_g) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            sharex=True,
        )

        # ---------------------------------------------------------
        # Upper graph: positions
        # ---------------------------------------------------------

        ax_q.plot(
            t,
            target[start:end],
            label="target position",
            lw=2,
        )

        ax_q.plot(
            t,
            actual[start:end],
            label="actual position",
            lw=1,
        )

        if predicted_q is not None:
            ax_q.plot(
                t,
                predicted_q[start:end, joint],
                label="predicted position",
                lw=2,
            )

        ax_q.set_ylabel("position (rad)")
        ax_q.set_title(f"{name} joint")
        ax_q.legend(loc="best")

        # ---------------------------------------------------------
        # Lower graph: differences
        # ---------------------------------------------------------

        # Actual - target
        actual_error = (
            actual
            - target
        )

        ax_g.plot(
            t,
            actual_error[start:end],
            label="actual - target",
            lw=1,
        )

        # Actual - predicted
        if predicted_q is not None:
            predicted_error = (
                actual
                - predicted_q[:, joint]
            )

            ax_g.plot(
                t,
                predicted_error[start:end],
                label="actual - predicted",
                lw=1,
            )

        ax_g.axhline(
            0,
            color="grey",
            lw=0.8,
        )

        ax_g.set_ylabel("difference (rad)")
        ax_g.set_xlabel("time (s)")
        ax_g.legend(loc="best")

        fig.tight_layout()

        return fig


def predict_actual_q(
    recording,
    model,
):
    """
    Calculate predicted actual joint positions without modifying
    the recording.

    Returns:
        np.ndarray with shape (N, 6)
    """

    prediction = model.predict(
        recording.df
    )

    predicted_q = prediction[
        "actual_q"
    ]

    if predicted_q.shape != recording.actual_q.shape:
        raise ValueError(
            f"Prediction shape {predicted_q.shape} does not match "
            f"actual_q shape {recording.actual_q.shape}"
        )

    return predicted_q


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Per-joint stats and a plot "
            "for a recorded run."
        )
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
        help=(
            "joint index 0..5 to plot "
            "(default 1 = shoulder)"
        ),
    )

    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="print stats only",
    )

    ap.add_argument(
        "--predict",
        action="store_true",
        help=(
            "Calculate and show the "
            "position-model prediction"
        ),
    )

    ap.add_argument(
        "--model",
        default="models/distill_position.pkl",
        help=(
            "Path to the trained position model"
        ),
    )

    args = ap.parse_args()

    if not 0 <= args.joint < N_JOINTS:
        ap.error(
            f"--joint must be between "
            f"0 and {N_JOINTS - 1}"
        )

    rec = Recording(
        args.csv
    )

    # ---------------------------------------------------------
    # Keep the original actual_q untouched.
    # ---------------------------------------------------------

    predicted_q = None

    if args.predict:
        import sys
        print(args.model)
        sys.modules["__main__"].OwnModel = OwnModel
        model = DistillModel.load(
            args.model
        )

        predicted_q = predict_actual_q(
            rec,
            model,
        )

    print(
        f"{args.csv}  "
        f"({len(rec.t)} rows, "
        f"dt {rec.dt * 1e3:.1f} ms)"
    )

    # ---------------------------------------------------------
    # Per-joint summary
    # ---------------------------------------------------------

    print(
        f"{'joint':10s} "
        f"{'moved':>9s} "
        f"{'gap RMS':>9s} "
        f"{'gap max':>9s} "
        f"{'pos err':>10s}"
    )

    for j in range(
        N_JOINTS
    ):

        # -----------------------------------------------------
        # Target range of motion
        # -----------------------------------------------------

        moved = (
            rec.target_q[:, j].max()
            - rec.target_q[:, j].min()
        )

        # -----------------------------------------------------
        # Current gap
        # -----------------------------------------------------

        gap = rec.current_gap(
            j
        )

        gap_rms = float(
            np.sqrt(
                np.mean(
                    gap ** 2
                )
            )
        )

        gap_max = float(
            np.max(
                np.abs(gap)
            )
        )

        # -----------------------------------------------------
        # Actual position error
        #
        # actual - target
        # -----------------------------------------------------

        position_error = (
            rec.actual_q[:, j]
            - rec.target_q[:, j]
        )

        pos_err = float(
            np.sqrt(
                np.mean(
                    position_error ** 2
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

    # ---------------------------------------------------------
    # Optional model prediction statistics
    # ---------------------------------------------------------

    if predicted_q is not None:

        print()
        print(
            "Position model prediction error:"
        )

        print(
            f"{'joint':10s} "
            f"{'RMSE':>12s} "
            f"{'max abs':>12s}"
        )

        for j in range(
            N_JOINTS
        ):

            prediction_error = (
                predicted_q[:, j]
                - rec.actual_q[:, j]
            )

            prediction_rmse = float(
                np.sqrt(
                    np.mean(
                        prediction_error ** 2
                    )
                )
            )

            prediction_max = float(
                np.max(
                    np.abs(
                        prediction_error
                    )
                )
            )

            print(
                f"{JOINT_NAMES[j]:10s} "
                f"{prediction_rmse * 1e3:10.3f}mrad "
                f"{prediction_max * 1e3:10.3f}mrad"
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    if not args.no_plot:

        import matplotlib.pyplot as plt

        rec.plot(
            args.joint,
            predicted_q,
        )

        plt.show()


if __name__ == "__main__":
    main()