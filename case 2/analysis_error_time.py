"""
Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion,
current gap, position lag), and plot one joint's target vs actual position.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).

With ``--predict``, the trained position-error model is used to predict:

    error_q = target_q - actual_q

The temporal model uses:

    t-20
    t-10
    t-5
    t-2
    t-1
    t

For every prediction at time t, the model receives the following features
from all six timeframes:

    target_q
    target_qd
    target_qdd
    target_current
    target_moment
    vel
    acc

There are:

    44 features per timeframe
    6 timeframes
    264 total input features

Example:

    python analysis.py \
        --csv data/test-4.csv \
        --joint 1 \
        --predict
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import train_distillation_model_time

from train_distillation_model_time import (
    DistillModel,
    OwnModel,
    SimpleNN,
    TIME_OFFSETS,
    N_TIMEFRAMES,
    MAX_HISTORY,
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

    def __init__(
        self,
        path: str,
        df=None,
    ):

        if df is None:

            df = pd.read_csv(
                path
            )

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
        # Used by the temporal prediction model.
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
        # Used by the temporal prediction model.
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
        # Used by the temporal prediction model and current-gap
        # statistics.
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
        # Used by the current-gap statistics.
        # ---------------------------------------------------------

        self.actual_current = np.column_stack(
            [
                df[f"actual_current{j}"]
                for j in range(N_JOINTS)
            ]
        )

        # ---------------------------------------------------------
        # Scalar velocity command
        #
        # Used by the temporal prediction model.
        # ---------------------------------------------------------

        self.vel_cmd = (
            df[
                VEL_COL
            ].to_numpy(
                dtype=float
            )
            if VEL_COL in df
            else None
        )

        # ---------------------------------------------------------
        # Scalar acceleration command
        #
        # Used by the temporal prediction model.
        # ---------------------------------------------------------

        self.acc_cmd = (
            df[
                ACC_COL
            ].to_numpy(
                dtype=float
            )
            if ACC_COL in df
            else None
        )

        # ---------------------------------------------------------
        # Other recording information.
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

        if len(self.t) < 2:

            return float("nan")

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

        using the temporal frames:

            t-20
            t-10
            t-5
            t-2
            t-1
            t
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
                label=(
                    "predicted error "
                    "(t-20,t-10,t-5,t-2,t-1,t)"
                ),
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
                self.target_current[
                    start:end,
                    joint
                ] / 10000.0,
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
    Predict position error using the temporal model.

    The model uses:

        t-20
        t-10
        t-5
        t-2
        t-1
        t

For each frame it uses:

        target_q
        target_qd
        target_qdd
        target_current
        target_moment
        vel
        acc

Therefore:

        44 features/frame
        6 frames
        264 input features

The model predicts:

        target_q(t) - actual_q(t)

for the current frame t.

The temporal window is constructed inside OwnModel.predict().
The first MAX_HISTORY rows are returned as NaN.
"""

    # ---------------------------------------------------------
    # Verify temporal configuration.
    # ---------------------------------------------------------

    expected_input_size = (
        OwnModel.FRAME_INPUT_SIZE
        * N_TIMEFRAMES
    )

    if OwnModel.INPUT_SIZE != expected_input_size:

        raise ValueError(
            "Temporal model configuration "
            "is inconsistent: "
            f"FRAME_INPUT_SIZE="
            f"{OwnModel.FRAME_INPUT_SIZE}, "
            f"N_TIMEFRAMES="
            f"{N_TIMEFRAMES}, "
            f"INPUT_SIZE="
            f"{OwnModel.INPUT_SIZE}"
        )

    # ---------------------------------------------------------
    # Verify that the CSV contains the required model inputs.
    # ---------------------------------------------------------

    required_columns = []

    for j in range(
        N_JOINTS
    ):

        required_columns.extend(
            [
                f"target_q{j}",
                f"target_qd{j}",
                f"target_qdd{j}",
                f"target_current{j}",
                f"target_moment{j}",
            ]
        )

    required_columns.extend(
        [
            VEL_COL,
            ACC_COL,
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in recording.df.columns
    ]

    if missing:

        raise ValueError(
            "CSV is missing input columns "
            "required by the temporal model:\n"
            + "\n".join(
                f"  {column}"
                for column in missing
            )
        )

    # ---------------------------------------------------------
    # Pass the COMPLETE DataFrame.
    #
    # OwnModel.predict() constructs:
    #
    #     t-20
    #     t-10
    #     t-5
    #     t-2
    #     t-1
    #     t
    #
    # internally.
    # ---------------------------------------------------------

    prediction = model.predict(
        recording.df
    )

    # ---------------------------------------------------------
    # The model returns:
    #
    #     "error_q"
    #
    # with shape:
    #
    #     (number_of_rows, 6)
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
            "temporal position-error model prediction"
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

        # -----------------------------------------------------
        # Required when loading the pickle depending on how
        # the model was originally serialized.
        # -----------------------------------------------------

        sys.modules[
            "__main__"
        ].OwnModel = OwnModel

        sys.modules[
            "__main__"
        ].SimpleNN = SimpleNN

        # -----------------------------------------------------
        # Load trained temporal model.
        # -----------------------------------------------------

        model = DistillModel.load(
            args.model
        )

        # -----------------------------------------------------
        # Check model configuration.
        # -----------------------------------------------------

        if not hasattr(
            model,
            "INPUT_SIZE",
        ):

            raise ValueError(
                "Loaded model does not contain "
                "INPUT_SIZE. It may have been "
                "trained with an older version "
                "of the model."
            )

        expected_input_size = (
            OwnModel.FRAME_INPUT_SIZE
            * N_TIMEFRAMES
        )

        if model.INPUT_SIZE != expected_input_size:

            raise ValueError(
                "Loaded model is not compatible "
                "with the current temporal model.\n"
                f"Expected input size: "
                f"{expected_input_size}\n"
                f"Loaded model input size: "
                f"{model.INPUT_SIZE}\n\n"
                "Make sure you retrain the model "
                "with the new temporal version."
            )

        # -----------------------------------------------------
        # Predict.
        # -----------------------------------------------------

        predicted_error = predict_error(
            rec,
            model,
        )

    print(
        f"{args.csv}  "
        f"({len(rec.t)} rows, "
        f"dt {rec.dt * 1e3:.1f} ms)"
    )

    if args.predict:

        print(
            f"Temporal model: "
            f"{N_TIMEFRAMES} frames"
        )

        print(
            "Temporal offsets: "
            + ", ".join(
                (
                    f"t-{offset}"
                    if offset != 0
                    else "t"
                )
                for offset in TIME_OFFSETS
            )
        )

        print(
            f"Model inputs: "
            f"{OwnModel.INPUT_SIZE}"
        )

        print(
            f"First valid prediction: "
            f"row {MAX_HISTORY}"
        )

        print()

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
            "Temporal offsets: "
            + ", ".join(
                (
                    f"t-{offset}"
                    if offset != 0
                    else "t"
                )
                for offset in TIME_OFFSETS
            )
        )

        print()

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
            # Only evaluate rows where the temporal model has
            # a complete history.
            #
            # First MAX_HISTORY rows contain NaN.
            # -------------------------------------------------

            valid = np.isfinite(
                pred_error
            )

            true_error_valid = (
                true_error[valid]
            )

            pred_error_valid = (
                pred_error[valid]
            )

            if len(
                pred_error_valid
            ) == 0:

                print(
                    f"{JOINT_NAMES[j]:10s} "
                    f"no valid predictions"
                )

                continue

            # -------------------------------------------------
            # Prediction residual
            # -------------------------------------------------

            residual = (
                true_error_valid
                - pred_error_valid
            )

            # -------------------------------------------------
            # RMS of actual error
            # -------------------------------------------------

            true_rms = float(
                np.sqrt(
                    np.mean(
                        true_error_valid ** 2
                    )
                )
            )

            # -------------------------------------------------
            # RMS of predicted error
            # -------------------------------------------------

            pred_rms = float(
                np.sqrt(
                    np.mean(
                        pred_error_valid ** 2
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