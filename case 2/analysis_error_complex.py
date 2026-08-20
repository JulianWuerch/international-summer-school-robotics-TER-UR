"""Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion,
current gap, position lag), and plot one joint's target vs actual position.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).

With ``--predict``, the trained position-error model is used to predict:

    error_q = target_q - actual_q

using all target-side information:

    target_q
    target_qd
    target_qdd
    target_current
    target_moment
    vel
    acc

Example:

    python analysis.py --csv data/test-4.csv --joint 1 --predict
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import train_distillation_model_error_complex
from train_distillation_model_error_complex import (
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


# ============================================================================
# Recording
# ============================================================================

class Recording:
    """One recorded run loaded from a ``record.py`` CSV."""

    def __init__(
        self,
        path: str,
        df=None,
    ):

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
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Actual joint positions
        # ---------------------------------------------------------

        self.actual_q = np.column_stack(
            [
                df[f"actual_q{j}"]
                for j in range(N_JOINTS)
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Target joint velocities
        # ---------------------------------------------------------

        self.target_qd = np.column_stack(
            [
                df[f"target_qd{j}"]
                for j in range(N_JOINTS)
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Target joint accelerations
        # ---------------------------------------------------------

        self.target_qdd = np.column_stack(
            [
                df[f"target_qdd{j}"]
                for j in range(N_JOINTS)
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Target joint currents
        # ---------------------------------------------------------

        self.target_current = np.column_stack(
            [
                df[f"target_current{j}"]
                for j in range(N_JOINTS)
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Target joint moments
        # ---------------------------------------------------------

        self.target_moment = np.column_stack(
            [
                df[f"target_moment{j}"]
                for j in range(N_JOINTS)
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Actual current
        #
        # Kept because the statistics report current gap.
        # ---------------------------------------------------------

        self.actual_current = np.column_stack(
            [
                df[f"actual_current{j}"]
                for j in range(N_JOINTS)
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Global velocity command
        # ---------------------------------------------------------

        self.vel_cmd = (
            df[VEL_COL].to_numpy(
                dtype=np.float32
            )
            if VEL_COL in df
            else None
        )

        # ---------------------------------------------------------
        # Global acceleration command
        # ---------------------------------------------------------

        self.acc_cmd = (
            df[ACC_COL].to_numpy(
                dtype=np.float32
            )
            if ACC_COL in df
            else None
        )

        # ---------------------------------------------------------
        # Script/control information
        # ---------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Current gap
    # ------------------------------------------------------------------

    def current_gap(
        self,
        joint: int,
    ) -> np.ndarray:
        """Actual minus target current for one joint."""

        return (
            self.actual_current[:, joint]
            - self.target_current[:, joint]
        )

    # ------------------------------------------------------------------
    # Position error
    # ------------------------------------------------------------------

    def position_error(
        self,
        joint: int,
    ) -> np.ndarray:
        """
        Target minus actual position for one joint.

        Same sign convention as the neural network:

            error = target_q - actual_q
        """

        return wrapped_angle_error(
            self.target_q[:, joint],
            self.actual_q[:, joint],
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

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
        # Limit plot window
        # ---------------------------------------------------------

        start = 2388

        end = min(
            2388*4,
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
        # Upper graph:
        #
        # target vs actual position
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
        # Lower graph:
        #
        # true error vs predicted error
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

            # -----------------------------------------------------
            # Optional target current overlay.
            # -----------------------------------------------------

            ax_g.plot(
                t,
                self.target_current[
                    start:end,
                    joint
                ] / 10000.0,
                label="target current / 10000",
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

    The model uses all target-side inputs:

        target_q
        target_qd
        target_qdd
        target_current
        target_moment
        vel
        acc

    and predicts:

        target_q - actual_q
    """

    prediction = model.predict(
        recording.df
    )

    predicted_error = prediction[
        "error_q"
    ]

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

    predicted_error = None

    if args.predict:

        import sys

        # Required for pickle loading depending on
        # how the model was serialized.

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

        moved = (
            rec.target_q[:, j].max()
            - rec.target_q[:, j].min()
        )

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

            true_error = wrapped_angle_error(
                rec.target_q[:, j],
                rec.actual_q[:, j],
            )

            pred_error = predicted_error[
                :,
                j,
            ]

            residual = (
                true_error
                - pred_error
            )

            true_rms = float(
                np.sqrt(
                    np.mean(
                        true_error ** 2
                    )
                )
            )

            pred_rms = float(
                np.sqrt(
                    np.mean(
                        pred_error ** 2
                    )
                )
            )

            residual_rms = float(
                np.sqrt(
                    np.mean(
                        residual ** 2
                    )
                )
            )

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