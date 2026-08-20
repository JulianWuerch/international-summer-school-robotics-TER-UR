"""Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion,
current gap, position lag), and plot one joint's target vs actual position.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).

With ``--predict``, the trained position-error model is used to predict:

    error_q = target_q - actual_q

using ONLY:

    target_q
    target_qd

Example:

    python analysis.py --csv data/test-4.csv --joint 1 --predict
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import train_distillation_model_error
from train_distillation_model_error import (
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

        self.t = df[
            TIME_COL
        ].to_numpy(
            dtype=float
        )

        # ---------------------------------------------------------
        # Target joint positions
        # ---------------------------------------------------------

        self.target_q = np.column_stack(
            [
                df[f"target_q{j}"]
                for j in range(N_JOINTS)
            ]
        )

        # ---------------------------------------------------------
        # Actual joint positions
        # ---------------------------------------------------------

        self.actual_q = np.column_stack(
            [
                df[f"actual_q{j}"]
                for j in range(N_JOINTS)
            ]
        )

        # ---------------------------------------------------------
        # Target joint velocities
        #
        # These are used by the prediction model.
        # ---------------------------------------------------------

        self.target_qd = np.column_stack(
            [
                df[f"target_qd{j}"]
                for j in range(N_JOINTS)
            ]
        )

        # ---------------------------------------------------------
        # Target joint accelerations
        #
        # Kept here in case other analysis code uses them.
        # They are NOT used by the prediction model.
        # ---------------------------------------------------------

        self.target_qdd = np.column_stack(
            [
                df[f"target_qdd{j}"]
                for j in range(N_JOINTS)
            ]
        )

        # ---------------------------------------------------------
        # Target current
        #
        # Kept for current-gap statistics.
        # It is NOT used by the prediction model.
        # ---------------------------------------------------------

        self.target_current = np.column_stack(
            [
                df[f"target_current{j}"]
                for j in range(N_JOINTS)
            ]
        )

        # ---------------------------------------------------------
        # Actual current
        #
        # Kept because the existing statistics report current gap.
        # ---------------------------------------------------------

        self.actual_current = np.column_stack(
            [
                df[f"actual_current{j}"]
                for j in range(N_JOINTS)
            ]
        )

        self.vel_cmd = (
            df[VEL_COL].to_numpy(
                dtype=float
            )
            if VEL_COL in df
            else None
        )

        self.acc_cmd = (
            df[ACC_COL].to_numpy(
                dtype=float
            )
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
        """
        Target minus actual position for one joint (rad), per row.

        This uses the same sign convention as the neural network:

            error = target_q - actual_q
        """

        return wrapped_angle_error(
            self.target_q[:, joint],
            self.actual_q[:, joint],
        )

    def plot(
        self,
        joint: int = 1,
        predicted_error=None,
    ):
        """
        Plot target/actual position and compare true position error
        against predicted position error.

        The model predicts:

            target_q - actual_q
        """

        import matplotlib.pyplot as plt

        name = JOINT_NAMES[joint]

        target = self.target_q[:, joint]
        actual = self.actual_q[:, joint]

        # ---------------------------------------------------------
        # Limit plot to the same 2388-sample window.
        # ---------------------------------------------------------

        start = 0

        end = min(
            2388,
            len(self.t),
        )

        t = self.t[
            start:end
        ]

        fig, (
            ax_q,
            ax_g,
        ) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            sharex=True,
        )

        # ---------------------------------------------------------
        # Upper graph: target vs actual position
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


        # ---------------------------------------------------------
        # Reconstruct predicted actual position:
        #
        # predicted_error = target - actual
        #
        # therefore:
        #
        # predicted_actual = target - predicted_error
        # ---------------------------------------------------------

        if predicted_error is not None:

            predicted_position = (
                target
                - predicted_error[:, joint]
            )

            ax_q.plot(
                t,
                predicted_position[start:end],
                label="predicted position",
                lw=2,
            )

        ax_q.set_ylabel(
            "position (rad)"
        )

        ax_q.set_title(
            f"{name} joint"
        )

        ax_q.legend(
            loc="best"
        )

        # ---------------------------------------------------------
        # Lower graph: TRUE error vs PREDICTED error
        #
        # Both use:
        #
        #     target - actual
        # ---------------------------------------------------------

        true_error = wrapped_angle_error(
            target,
            actual,
        )

        ax_g.plot(
            t,
            true_error[start:end],
            label="actual error",
            lw=1,
        )

        if predicted_error is not None:

            ax_g.plot(
                t,
                predicted_error[
                    start:end,
                    joint
                ],
                label="predicted error",
                lw=2,
            )

            residual = (
                true_error
                - predicted_error[:, joint]
            )

            ax_g.plot(
                t,
                residual[start:end],
                label="prediction residual",
                lw=1,
            )

            ax_g.plot(
                t,
                self.target_current[start:end, joint] / 10000.0,
                label="target current",
                lw=1,
            )

        ax_g.axhline(
            0,
            color="grey",
            lw=0.8,
        )

        ax_g.set_ylabel(
            "error (rad)"
        )

        ax_g.set_xlabel(
            "time (s)"
        )

        ax_g.legend(
            loc="best"
        )

        fig.tight_layout()

        return fig


# ============================================================================
# Prediction
# ============================================================================

def predict_error(
    recording,
    model,
):
    """
    Predict position error.

    The model uses ONLY:

        target_q
        target_qd

    and predicts:

        target_q - actual_q
    """

    prediction = model.predict(
        recording.df
    )

    # ---------------------------------------------------------
    # The updated OwnModel returns "error_q".
    # ---------------------------------------------------------

    predicted_error = prediction[
        "error_q"
    ]

    # ---------------------------------------------------------
    # Verify output shape.
    # ---------------------------------------------------------

    if predicted_error.shape != recording.actual_q.shape:

        raise ValueError(
            f"Prediction shape "
            f"{predicted_error.shape} "
            f"does not match "
            f"actual_q shape "
            f"{recording.actual_q.shape}"
        )

    return predicted_error


# ============================================================================
# Angle error
# ============================================================================

def wrapped_angle_error(
    target,
    actual,
):
    """
    Calculate wrapped position error:

        target - actual

    Result is in [-pi, pi).
    """

    error = (
        target
        - actual
    )

    return (
        (error + np.pi)
        % (2.0 * np.pi)
    ) - np.pi


# ============================================================================
# Main
# ============================================================================

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
            "position-error model prediction"
        ),
    )

    ap.add_argument(
        "--model",
        default="models/distill_position.pkl",
        help=(
            "Path to the trained position-error model"
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

    predicted_error = None

    if args.predict:

        import sys

        # Required when loading the pickle depending on
        # how the model was originally serialized.
        sys.modules[
            "__main__"
        ].OwnModel = OwnModel

        sys.modules[
            "__main__"
        ].SimpleNN = SimpleNN

        model = DistillModel.load(
            args.model
        )

        predicted_error = predict_error(
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
        # target - actual
        # -----------------------------------------------------

        position_error = wrapped_angle_error(
            rec.target_q[:, j],
            rec.actual_q[:, j],
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
    # Model prediction evaluation
    # ---------------------------------------------------------

    if predicted_error is not None:

        print()
        print(
            "Position error prediction:"
        )

        print(
            f"{'joint':10s} "
            f"{'true RMS':>12s} "
            f"{'pred RMS':>12s} "
            f"{'res RMS':>12s} "
            f"{'res max':>12s}"
        )

        for j in range(
            N_JOINTS
        ):

            # -------------------------------------------------
            # True error:
            #
            # target - actual
            # -------------------------------------------------

            true_error = wrapped_angle_error(
                rec.target_q[:, j],
                rec.actual_q[:, j],
            )

            # -------------------------------------------------
            # Model prediction
            # -------------------------------------------------

            pred_error = predicted_error[:, j]

            # -------------------------------------------------
            # Prediction residual
            # -------------------------------------------------

            residual = (
                true_error
                - pred_error
            )

            # -------------------------------------------------
            # RMS of actual error
            # -------------------------------------------------

            true_rms = float(
                np.sqrt(
                    np.mean(
                        true_error ** 2
                    )
                )
            )

            # -------------------------------------------------
            # RMS of predicted error
            # -------------------------------------------------

            pred_rms = float(
                np.sqrt(
                    np.mean(
                        pred_error ** 2
                    )
                )
            )

            # -------------------------------------------------
            # RMS prediction residual
            # -------------------------------------------------

            residual_rms = float(
                np.sqrt(
                    np.mean(
                        residual ** 2
                    )
                )
            )

            # -------------------------------------------------
            # Maximum absolute prediction residual
            # -------------------------------------------------

            residual_max = float(
                np.max(
                    np.abs(
                        residual
                    )
                )
            )

            print(
                f"{JOINT_NAMES[j]:10s} "
                f"{true_rms * 1e3:10.3f}mrad "
                f"{pred_rms * 1e3:10.3f}mrad "
                f"{residual_rms * 1e3:10.3f}mrad "
                f"{residual_max * 1e3:10.3f}mrad"
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    if not args.no_plot:

        import matplotlib.pyplot as plt

        rec.plot(
            args.joint,
            predicted_error,
        )

        plt.show()


if __name__ == "__main__":
    main()