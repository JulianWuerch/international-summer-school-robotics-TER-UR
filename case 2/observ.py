"""
3D visualization of the learned position-error model.

The model predicts:

    error_q = target_q - actual_q

using:

    target_q0 ... target_q5
    target_qd0 ... target_qd5

For a selected joint, this script plots:

    X = target_q[joint]
    Z = target_qd[joint]
    Y = target_q[joint] - actual_q[joint]

The recorded samples are shown as a 3D scatter plot.

A grid is then evaluated through the neural network. For the selected
joint, target_q and target_qd are varied across the X/Z plane.

The other 10 model inputs are held constant at their mean values.

Example:

    python visualize_error_model.py \
        --csv data/test-4.csv \
        --model models/distill_position.pkl \
        --joint 1

Joint indices:

    0 = base
    1 = shoulder
    2 = elbow
    3 = wrist1
    4 = wrist2
    5 = wrist3
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train_distillation_model_error_complex import (
    DistillModel,
    OwnModel,
    SimpleNN,
)

from utils import (
    JOINT_NAMES,
    N_JOINTS,
)


# ============================================================================
# Data loading
# ============================================================================

def load_data(csv_path: str):
    """
    Load the CSV and extract the 12 model inputs and true errors.
    """

    df = pd.read_csv(
        csv_path
    )

    target_q = np.column_stack(
        [
            df[f"target_q{j}"].to_numpy(
                dtype=np.float32
            )
            for j in range(N_JOINTS)
        ]
    )

    target_qd = np.column_stack(
        [
            df[f"target_qd{j}"].to_numpy(
                dtype=np.float32
            )
            for j in range(N_JOINTS)
        ]
    )

    actual_q = np.column_stack(
        [
            df[f"actual_q{j}"].to_numpy(
                dtype=np.float32
            )
            for j in range(N_JOINTS)
        ]
    )

    # --------------------------------------------------------------
    # Same error definition used during NN training:
    #
    #     target_q - actual_q
    #
    # Wrap angular error to [-pi, pi).
    # --------------------------------------------------------------

    error = (
        target_q
        - actual_q
    )

    error = (
        (error + np.pi)
        % (2.0 * np.pi)
    ) - np.pi

    # --------------------------------------------------------------
    # Construct the exact 12-input representation used by the NN.
    # --------------------------------------------------------------

    X = np.column_stack(
        [
            target_q,
            target_qd,
        ]
    ).astype(
        np.float32
    )

    return (
        df,
        target_q,
        target_qd,
        actual_q,
        error,
        X,
    )


# ============================================================================
# Model prediction
# ============================================================================

def predict_grid(
    model,
    base_input,
    joint,
    q_values,
    qd_values,
):
    """
    Evaluate the NN over a 2D q/qd grid.

    The selected joint's q and qd are varied.

    All other q/qd inputs remain fixed at their mean values.

    Returns:

        Q
        QD
        prediction
    """

    Q, QD = np.meshgrid(
        q_values,
        qd_values,
    )

    # --------------------------------------------------------------
    # Create one NN input for every point on the grid.
    #
    # Shape:
    #
    #     (grid_points, 12)
    # --------------------------------------------------------------

    grid_input = np.tile(
        base_input,
        (
            Q.size,
            1,
        ),
    )

    # --------------------------------------------------------------
    # Input layout:
    #
    #     [q0 q1 q2 q3 q4 q5 qd0 qd1 qd2 qd3 qd4 qd5]
    #
    # Therefore:
    #
    #     q  index = joint
    #     qd index = N_JOINTS + joint
    # --------------------------------------------------------------

    grid_input[
        :,
        joint,
    ] = Q.ravel()

    grid_input[
        :,
        N_JOINTS + joint,
    ] = QD.ravel()

    # --------------------------------------------------------------
    # Run network.
    # --------------------------------------------------------------

    import torch

    tensor = torch.from_numpy(
        grid_input
    ).float()

    model.modell.eval()

    with torch.no_grad():

        prediction = model.modell(
            tensor
        )

    prediction = (
        prediction
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------------
    # Only plot the selected joint's predicted error.
    # --------------------------------------------------------------

    Y = prediction[
        :,
        joint,
    ].reshape(
        Q.shape
    )

    return (
        Q,
        QD,
        Y,
    )


# ============================================================================
# Plot
# ============================================================================

def plot_model(
    target_q,
    target_qd,
    true_error,
    model,
    joint,
    grid_size=60,
):
    """
    Create a 3D scatter + NN prediction surface.
    """

    name = JOINT_NAMES[joint]

    # --------------------------------------------------------------
    # Recorded data.
    # --------------------------------------------------------------

    x = target_q[
        :,
        joint,
    ]

    z = target_qd[
        :,
        joint,
    ]

    y = true_error[
        :,
        joint,
    ]

    # --------------------------------------------------------------
    # Base input for the NN.
    #
    # All other joints are held at their mean values.
    # --------------------------------------------------------------

    base_input = np.concatenate(
        [
            np.mean(
                target_q,
                axis=0,
            ),
            np.mean(
                target_qd,
                axis=0,
            ),
        ]
    ).astype(
        np.float32
    )

    # --------------------------------------------------------------
    # Grid limits.
    #
    # Use the observed range of the selected joint.
    # --------------------------------------------------------------

    q_min = float(
        np.min(x)
    )

    q_max = float(
        np.max(x)
    )

    qd_min = float(
        np.min(z)
    )

    qd_max = float(
        np.max(z)
    )

    # Add a small margin.
    q_margin = (
        q_max - q_min
    ) * 0.03

    qd_margin = (
        qd_max - qd_min
    ) * 0.03

    if q_margin == 0:
        q_margin = 0.01

    if qd_margin == 0:
        qd_margin = 0.01

    q_values = np.linspace(
        q_min - q_margin,
        q_max + q_margin,
        grid_size,
    )

    qd_values = np.linspace(
        qd_min - qd_margin,
        qd_max + qd_margin,
        grid_size,
    )

    # --------------------------------------------------------------
    # Evaluate neural network over the X/Z plane.
    # --------------------------------------------------------------

    Q, QD, prediction = predict_grid(
        model,
        base_input,
        joint,
        q_values,
        qd_values,
    )

    # --------------------------------------------------------------
    # Plot.
    # --------------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # --------------------------------------------------------------
    # Recorded samples.
    # --------------------------------------------------------------

    scatter = ax.scatter(
        x,
        y,
        z,
        s=8,
        alpha=0.35,
        label="recorded",
    )

    # --------------------------------------------------------------
    # NN prediction surface.
    #
    # Note that matplotlib's surface arguments are:
    #
    #     X
    #     Y
    #     Z
    #
    # Our physical axes are:
    #
    #     X = target_q
    #     Y = error
    #     Z = target_qd
    #
    # Therefore the NN surface is:
    #
    #     Q
    #     prediction
    #     QD
    # --------------------------------------------------------------

    surface = ax.plot_surface(
        Q,
        prediction,
        QD,
        alpha=0.55,
        linewidth=0,
        antialiased=True,
    )

    # --------------------------------------------------------------
    # Labels.
    # --------------------------------------------------------------

    ax.set_xlabel(
        f"target_q{joint} (rad)"
    )

    ax.set_ylabel(
        "position error (rad)"
    )

    ax.set_zlabel(
        f"target_qd{joint} (rad/s)"
    )

    ax.set_title(
        f"{name} — measured error vs neural-network prediction"
    )

    ax.legend()

    # --------------------------------------------------------------
    # Improve viewing angle.
    # --------------------------------------------------------------

    ax.view_init(
        elev=25,
        azim=-125,
    )

    plt.tight_layout()

    return fig


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "3D visualization of the "
            "position-error neural network."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Recorded CSV file.",
    )

    parser.add_argument(
        "--model",
        default="models/distill_position.pkl",
        help="Trained model pickle.",
    )

    parser.add_argument(
        "--joint",
        type=int,
        default=1,
        help=(
            "Joint to visualize, "
            "0..5."
        ),
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=60,
        help=(
            "Number of grid points in "
            "each dimension."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Validate joint.
    # --------------------------------------------------------------

    if not 0 <= args.joint < N_JOINTS:

        parser.error(
            f"--joint must be between "
            f"0 and {N_JOINTS - 1}"
        )

    # --------------------------------------------------------------
    # Load data.
    # --------------------------------------------------------------

    (
        df,
        target_q,
        target_qd,
        actual_q,
        true_error,
        X,
    ) = load_data(
        args.csv
    )

    print(
        f"Loaded {args.csv}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Joint: {args.joint} "
        f"({JOINT_NAMES[args.joint]})"
    )

    print(
        f"Input shape: {X.shape}"
    )

    # --------------------------------------------------------------
    # Load model.
    #
    # This handles pickles that refer to OwnModel/SimpleNN
    # through __main__.
    # --------------------------------------------------------------

    sys.modules[
        "__main__"
    ].OwnModel = OwnModel

    sys.modules[
        "__main__"
    ].SimpleNN = SimpleNN

    model = DistillModel.load(
        args.model
    )

    # --------------------------------------------------------------
    # Check that this is actually the expected 12-input model.
    # --------------------------------------------------------------

    first_layer = model.modell.modell[0]

    if first_layer.in_features != N_JOINTS * 2:

        raise ValueError(
            "Loaded model does not appear to be the "
            "12-input target_q + target_qd model. "
            f"Expected {N_JOINTS * 2} inputs, "
            f"got {first_layer.in_features}."
        )

    print(
        f"Model input size: "
        f"{first_layer.in_features}"
    )

    # --------------------------------------------------------------
    # Plot.
    # --------------------------------------------------------------

    plot_model(
        target_q,
        target_qd,
        true_error,
        model,
        args.joint,
        grid_size=args.grid,
    )

    plt.show()


if __name__ == "__main__":
    main()