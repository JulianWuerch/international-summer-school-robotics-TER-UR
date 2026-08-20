"""
Neural-network distillation model.

One CSV row = one training sample.

INPUTS:

    target_q0 ... target_q5
    target_qd0 ... target_qd5
    target_qdd0 ... target_qdd5
    target_current0 ... target_current5
    target_moment0 ... target_moment5

    vel
    acc

For 6 joints:

    target_q       = 6
    target_qd      = 6
    target_qdd     = 6
    target_current = 6
    target_moment  = 6
    vel            = 1
    acc            = 1

Total:

    44 input features

OUTPUTS:

    error_q0 ... error_q5

where:

    error_q = target_q - actual_q

Total outputs:

    6

Network:

    44 -> 64 -> 64 -> 64 -> 32 -> 32 -> 32 -> 32 -> 6
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
        input_size: int,
    ):
        super().__init__()

        output_size = n_joints

        self.modell = nn.Sequential(

            nn.Linear(
                input_size,
                80,
            ),

            nn.Tanh(),

            nn.Linear(
                80,
                80,
            ),

            nn.Tanh(),

            nn.Linear(
                80,
                80,
            ),

            nn.Tanh(),

            nn.Linear(
                80,
                64,
            ),

            nn.Tanh(),

            nn.Linear(
                64,
                32,
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                32,
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                32,
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                32,
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                output_size,
            ),
        )

    def forward(self, x):

        return self.modell(x)


# ============================================================================
# Neural-network model
# ============================================================================

class OwnModel(DistillModel):
    """
    Neural network that predicts joint position error.

    Input:

        target_q
        target_qd
        target_qdd
        target_current
        target_moment
        vel
        acc

    Output:

        error_q

    where:

        error_q = target_q - actual_q
    """

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    FEATURE_NAMES = (
        [
            f"target_q{i}"
            for i in range(N_JOINTS)
        ]

        + [
            f"target_qd{i}"
            for i in range(N_JOINTS)
        ]

        + [
            f"target_qdd{i}"
            for i in range(N_JOINTS)
        ]

        + [
            f"target_current{i}"
            for i in range(N_JOINTS)
        ]

        + [
            f"target_moment{i}"
            for i in range(N_JOINTS)
        ]

        + [
            "vel",
            "acc",
        ]
    )

    INPUT_SIZE = len(
        FEATURE_NAMES
    )

    def __init__(self):

        self.modell = SimpleNN(
            N_JOINTS,
            self.INPUT_SIZE,
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def predicts(self) -> list[str]:

        return [
            "error_q"
        ]

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _row_features(
        self,
        target_q,
        target_qd,
        target_qdd,
        target_current,
        target_moment,
        vel,
        acc,
    ) -> np.ndarray:
        """
        Build one feature row for every timestep.

        Output:

            (n, 44)
        """

        target_q = np.asarray(
            target_q,
            dtype=np.float32,
        )

        target_qd = np.asarray(
            target_qd,
            dtype=np.float32,
        )

        target_qdd = np.asarray(
            target_qdd,
            dtype=np.float32,
        )

        target_current = np.asarray(
            target_current,
            dtype=np.float32,
        )

        target_moment = np.asarray(
            target_moment,
            dtype=np.float32,
        )

        vel = np.asarray(
            vel,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
        )

        acc = np.asarray(
            acc,
            dtype=np.float32,
        ).reshape(
            -1,
            1,
        )

        X = np.column_stack(
            [
                target_q,
                target_qd,
                target_qdd,
                target_current,
                target_moment,
                vel,
                acc,
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

        X:

            44 target-side features

        y:

            6 position errors
        """

        X = []
        y = []

        for rec in recordings:

            # ----------------------------------------------------------
            # Get all target-side data.
            # ----------------------------------------------------------

            target_qdd = get_block(
                rec.df,
                "target_qdd",
            )

            target_moment = get_block(
                rec.df,
                "target_moment",
            )

            # ----------------------------------------------------------
            # Construct input.
            # ----------------------------------------------------------

            features = self._row_features(
                rec.target_q,
                rec.target_qd,
                target_qdd,
                rec.target_current,
                target_moment,
                rec.vel_cmd,
                rec.acc_cmd,
            )

            X.append(
                features
            )

            # ----------------------------------------------------------
            # Target:
            #
            #     target_q - actual_q
            # ----------------------------------------------------------

            actuals = np.asarray(
                rec.actual_q,
                dtype=np.float32,
            )

            targets = np.asarray(
                rec.target_q,
                dtype=np.float32,
            )

            error = (
                targets
                - actuals
            )

            # ----------------------------------------------------------
            # Wrap angular error to [-pi, pi).
            # ----------------------------------------------------------

            error = (
                (error + np.pi)
                % (2.0 * np.pi)
            ) - np.pi

            y.append(
                error
            )

        # --------------------------------------------------------------
        # Combine recordings.
        # --------------------------------------------------------------

        X = np.vstack(
            X
        )

        y = np.vstack(
            y
        )

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

        print(
            f"  Input features: {X.shape[1]}"
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

        # --------------------------------------------------------------
        # Training settings
        # --------------------------------------------------------------

        epochs = 200

        batch_size = 2048

        # --------------------------------------------------------------
        # Build dataset.
        # --------------------------------------------------------------

        X, y = self._design(
            recordings
        )

        # --------------------------------------------------------------
        # Verify dimensions.
        # --------------------------------------------------------------

        expected_input_size = (
            self.INPUT_SIZE
        )

        assert X.shape[1] == expected_input_size, (
            f"Expected "
            f"{expected_input_size} "
            f"input features, "
            f"got "
            f"{X.shape[1]}"
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
            N_JOINTS,
            self.INPUT_SIZE,
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
        # Learning-rate scheduler.
        # --------------------------------------------------------------

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
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
            # Batches.
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
                # Loss.
                # ------------------------------------------------------

                loss = criterion(
                    prediction * 100,
                    y_batch * 100,
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
                total_loss
                / max(
                    number_of_batches,
                    1,
                )
            )

            # ----------------------------------------------------------
            # Learning-rate scheduling.
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
        print(
            "DONE"
        )
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

        The model uses all target-side data:

            target_q
            target_qd
            target_qdd
            target_current
            target_moment
            vel
            acc

        Output:

            n rows x 6 predicted errors
        """

        # --------------------------------------------------------------
        # Get target blocks.
        # --------------------------------------------------------------

        target_q = get_block(
            df,
            "target_q",
        )

        target_qd = get_block(
            df,
            "target_qd",
        )

        target_qdd = get_block(
            df,
            "target_qdd",
        )

        target_current = get_block(
            df,
            "target_current",
        )

        target_moment = get_block(
            df,
            "target_moment",
        )

        # --------------------------------------------------------------
        # Get scalar target commands.
        # --------------------------------------------------------------

        vel = df[
            VEL_COL
        ].to_numpy(
            dtype=np.float32
        )

        acc = df[
            ACC_COL
        ].to_numpy(
            dtype=np.float32
        )

        # --------------------------------------------------------------
        # Construct exactly the same features as training.
        # --------------------------------------------------------------

        X = self._row_features(
            target_q,
            target_qd,
            target_qdd,
            target_current,
            target_moment,
            vel,
            acc,
        )

        # --------------------------------------------------------------
        # Verify dimensions.
        # --------------------------------------------------------------

        if X.shape[1] != self.INPUT_SIZE:

            raise ValueError(
                f"Expected "
                f"{self.INPUT_SIZE} "
                f"features, "
                f"got "
                f"{X.shape[1]}"
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
    Run the model on a CSV.

    The model predicts:

        error_q = target_q - actual_q

    The prediction is stored in:

        error_q0 ... error_q5
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
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train neural-network "
            "distillation model "
            "for joint position error."
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
    # Print model input information.
    # --------------------------------------------------------------

    print(
        f"Model inputs: "
        f"{model.INPUT_SIZE}"
    )

    for i, name in enumerate(
        model.FEATURE_NAMES
    ):

        print(
            f"  {i:2d}: {name}"
        )

    print()

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
        "Saved model to:"
    )

    print(
        args.out
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    main()