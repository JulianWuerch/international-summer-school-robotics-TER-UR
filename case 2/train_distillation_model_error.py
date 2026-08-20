"""
Neural-network distillation model.

One CSV row = one training sample.

INPUTS for 6 joints:

    target_q0 ... target_q5
    target_qd0 ... target_qd5
    target_qdd0 ... target_qdd5
    target_current0 ... target_current5
    vel
    acc

Total input features:

    6 + 6 + 6 + 6 + 1 + 1 = 26

OUTPUTS:

    actual_q0 ... actual_q5

Total outputs:

    6

Network:

    26 -> 64 -> 64 -> 64 -> 6
"""

from __future__ import annotations

import argparse
import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from preprocess import Identity, Preprocess, default_preprocess
from utils import (
    JOINT_NAMES,
    N_JOINTS,
    VEL_COL,
    ACC_COL,
    frame_dt,
    get_block,
    set_block,
)


# ============================================================================
# Base model
# ============================================================================

class DistillModel:
    """
    Base interface for distilled models.
    """

    def fit(self, recordings):
        raise NotImplementedError

    def predicts(self) -> list[str]:
        raise NotImplementedError

    def predict(self, df) -> dict:
        raise NotImplementedError

    def bounds(self):
        return None

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

# ============================================================================
# Neural network
# ============================================================================

class SimpleNN(nn.Module):

    def __init__(
        self,
        n_joints: int,
    ):
        super().__init__()

        # --------------------------------------------------------------
        # For 6 joints:
        #
        # target_q  = 6
        # target_qd = 6
        #
        # Total = 12
        # --------------------------------------------------------------

        input_size = n_joints * 2

        # Predict position error for every joint.
        output_size = n_joints

        self.modell = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.Tanh(),

            nn.Linear(64, 64),

            nn.Tanh(),

            nn.Linear(64, 32),

            nn.Tanh(),

            nn.Linear(32, 32),

            nn.Tanh(),

            nn.Linear(32, 32),

            nn.Tanh(),

            nn.Linear(32, 32),

            nn.Tanh(),

            nn.Linear(32, output_size),
        )

    def forward(self, x):

        return self.modell(x)


# ============================================================================
# Neural-network model
# ============================================================================

class OwnModel(DistillModel):
    """
    Neural network that predicts joint position error.

    One row corresponds to one timestep.

    Input:

        target_q[0:6]
        target_qd[0:6]

    Output:

        error_q[0:6]

    where:

        error_q = target_q - actual_q
    """

    FEATURE_NAMES = (
        [
            f"target_q{i}"
            for i in range(N_JOINTS)
        ]
        + [
            f"target_qd{i}"
            for i in range(N_JOINTS)
        ]
    )

    def __init__(self):

        self.modell = SimpleNN(
            N_JOINTS
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def predicts(self) -> list[str]:

        return ["error_q"]

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _row_features(
        self,
        target_q,
        target_qd,
    ) -> np.ndarray:
        """
        Build one feature row for every timestep.

        Input shapes:

            target_q  : (n, 6)
            target_qd : (n, 6)

        Output:

            (n, 12)
        """

        target_q = np.asarray(
            target_q,
            dtype=np.float32,
        )

        target_qd = np.asarray(
            target_qd,
            dtype=np.float32,
        )

        X = np.column_stack(
            [
                target_q,
                target_qd,
            ]
        )

        return X.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _design(
        self,
        recordings,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert recordings into one large dataset.

        One CSV row = one training sample.

        X shape:

            (number_of_rows, 12)

        y shape:

            (number_of_rows, 6)
        """

        X = []
        y = []

        for rec in recordings:

            # ----------------------------------------------------------
            # Only use target position and target velocity as inputs.
            # ----------------------------------------------------------

            features = self._row_features(
                rec.target_q,
                rec.target_qd,
            )

            X.append(
                features
            )

            # ----------------------------------------------------------
            # Target is ACTUAL POSITION ERROR.
            #
            # error = target_q - actual_q
            #
            # Shape:
            #
            #     (number_of_rows, 6)
            # ----------------------------------------------------------

            actuals = np.asarray(
                rec.actual_q,
                dtype=np.float32,
            )

            targets = np.asarray(
                rec.target_q,
                dtype=np.float32,
            )

            error = targets - actuals

            # Wrap angular error to [-pi, pi].
            error = (
                (error + np.pi) % (2.0 * np.pi)
            ) - np.pi

            y.append(
                error
            )

        # --------------------------------------------------------------
        # Combine recordings.
        # --------------------------------------------------------------

        X = np.vstack(X)

        y = np.vstack(y)

        print()
        print(
            "Designed dataset:"
        )
        print(
            f"  X: {X.shape}"
        )
        print(
            f"  y: {y.shape}"
        )
        print()

        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        recordings,
    ) -> "OwnModel":
        """
        Train the neural network.

        Inputs:

            target_q
            target_qd

        Target:

            target_q - actual_q
        """

        # --------------------------------------------------------------
        # Training settings
        # --------------------------------------------------------------

        epochs = 200
        batch_size = 2048

        # --------------------------------------------------------------
        # Build dataset ONCE.
        # --------------------------------------------------------------

        X, y = self._design(
            recordings
        )

        # --------------------------------------------------------------
        # Verify dimensions.
        # --------------------------------------------------------------

        expected_input_size = (
            N_JOINTS * 2
        )

        assert X.shape[1] == expected_input_size, (
            f"Expected {expected_input_size} input features, "
            f"got {X.shape[1]}"
        )

        # --------------------------------------------------------------
        # Convert numpy -> torch.
        # --------------------------------------------------------------

        X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        y = torch.tensor(
            y,
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Create network.
        # --------------------------------------------------------------

        self.modell = SimpleNN(
            N_JOINTS
        )

        # --------------------------------------------------------------
        # Loss.
        # --------------------------------------------------------------

        criterion = nn.MSELoss()

        # --------------------------------------------------------------
        # Optimizer.
        # --------------------------------------------------------------

        optimizer = torch.optim.Adam(
            self.modell.parameters(),
            lr=0.0015,
        )

        # --------------------------------------------------------------
        # Automatically reduce learning rate when the loss stagnates.
        # --------------------------------------------------------------

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-8,
            )
        )

        # --------------------------------------------------------------
        # Training loop.
        # --------------------------------------------------------------

        for epoch in range(
            epochs
        ):

            self.modell.train()

            # ----------------------------------------------------------
            # Shuffle rows.
            # ----------------------------------------------------------

            permutation = torch.randperm(
                len(X)
            )

            X_shuffled = X[
                permutation
            ]

            y_shuffled = y[
                permutation
            ]

            total_loss = 0.0

            number_of_batches = 0

            # ----------------------------------------------------------
            # Create batches.
            # ----------------------------------------------------------

            for batch_start in range(
                0,
                len(X_shuffled),
                batch_size,
            ):

                X_batch = X_shuffled[
                    batch_start:
                    batch_start + batch_size
                ]

                y_batch = y_shuffled[
                    batch_start:
                    batch_start + batch_size
                ]

                # ------------------------------------------------------
                # Reset gradients.
                # ------------------------------------------------------

                optimizer.zero_grad()

                # ------------------------------------------------------
                # Forward pass.
                # ------------------------------------------------------

                prediction = self.modell(
                    X_batch
                )

                # ------------------------------------------------------
                # Both are:
                #
                #     (batch_size, 6)
                # ------------------------------------------------------

                loss = criterion(
                    prediction,
                    y_batch,
                )

                # ------------------------------------------------------
                # Backpropagation.
                # ------------------------------------------------------

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                )

                number_of_batches += 1

            # ----------------------------------------------------------
            # Average loss.
            # ----------------------------------------------------------

            average_loss = (
                total_loss /
                max(
                    number_of_batches,
                    1,
                )
            )

            # ----------------------------------------------------------
            # Adjust learning rate if the loss stagnates.
            # ----------------------------------------------------------

            scheduler.step(
                average_loss
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {average_loss:.9f} | "
                f"LR: {current_lr:.8f}"
            )

        print()
        print("DONE")
        print()

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df,
    ) -> dict:
        """
        Predict joint position errors.

        Input:

            target_q
            target_qd

        Output:

            n rows x 6 predicted errors
        """

        # --------------------------------------------------------------
        # Get target position and velocity.
        # --------------------------------------------------------------

        target_q = get_block(
            df,
            "target_q",
        )

        target_qd = get_block(
            df,
            "target_qd",
        )

        # --------------------------------------------------------------
        # Construct exactly the same features used during training.
        # --------------------------------------------------------------

        X = self._row_features(
            target_q,
            target_qd,
        )

        # --------------------------------------------------------------
        # Convert to tensor.
        # --------------------------------------------------------------

        X_tensor = torch.from_numpy(
            X
        ).float()

        # --------------------------------------------------------------
        # Evaluation mode.
        # --------------------------------------------------------------

        self.modell.eval()

        with torch.no_grad():

            prediction = self.modell(
                X_tensor
            )

        # --------------------------------------------------------------
        # Torch -> numpy.
        # --------------------------------------------------------------

        prediction = (
            prediction
            .cpu()
            .numpy()
        )

        return {
            "error_q": prediction
        }

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def bounds(self):

        return None


# ============================================================================
# Augmentation
# ============================================================================

def augment(
    model: DistillModel,
    csv: str,
    pre: Preprocess = None,
):
    """
    Replace actual_* columns with model predictions.

    For OwnModel this means:

        actual_q0 ... actual_q5

    are replaced by the predicted values.
    """

    if pre is None:
        pre = Identity()

    # --------------------------------------------------------------
    # Read CSV.
    # --------------------------------------------------------------

    df = pd.read_csv(
        csv
    )

    # --------------------------------------------------------------
    # Apply preprocessing.
    # --------------------------------------------------------------

    df = pre.transform_distill(
        df
    )

    # --------------------------------------------------------------
    # Predict.
    # --------------------------------------------------------------

    preds = model.predict(
        df
    )

    # --------------------------------------------------------------
    # Write predictions.
    # --------------------------------------------------------------

    for base in model.predicts():

        set_block(
            df,
            base,
            preds[base],
        )

    # --------------------------------------------------------------
    # Revert preprocessing.
    # --------------------------------------------------------------

    df = pre.revert_distill(
        df
    )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    df.to_csv(
        csv,
        index=False,
    )

    print(
        f"overwrote {model.predicts()} "
        f"with predictions -> {csv}"
    )

    return df


# ============================================================================
# Holdout evaluation
# ============================================================================

# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train neural-network "
            "distillation model "
            "for actual joint position."
        )
    )

    parser.add_argument(
        "--csvs",
        nargs="+",
        default=sorted(
            glob.glob(
                "data/test-*.csv"
            )
        ),
        help=(
            "Recorded runs used "
            "for training."
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "models/"
            "distill_position.pkl"
        ),
        help=(
            "Output pickle file."
        ),
    )


    args = parser.parse_args()

    # --------------------------------------------------------------
    # Import Recording here to avoid circular imports.
    # --------------------------------------------------------------

    from analysis import Recording

    # --------------------------------------------------------------
    # Preprocessing.
    # --------------------------------------------------------------

    pre = default_preprocess()

    # --------------------------------------------------------------
    # Load recordings.
    # --------------------------------------------------------------

    recordings = []

    for path in args.csvs:

        print(
            f"Loading {path}"
        )

        rec = Recording(
            path
        )

        rec.df = pre.transform_distill(
            rec.df
        )

        recordings.append(
            rec
        )

    print()
    print(
        f"Loaded "
        f"{len(recordings)} recordings"
    )
    print()

    # --------------------------------------------------------------
    # Create model.
    # --------------------------------------------------------------

    model = OwnModel()

    # --------------------------------------------------------------
    # Train.
    # --------------------------------------------------------------

    model.fit(
        recordings
    )


    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    model.save(
        args.out
    )

    print()
    print(
        f"Saved model to:"
    )
    print(
        args.out
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()