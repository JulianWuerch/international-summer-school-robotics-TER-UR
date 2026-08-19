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
# Original linear model
# ============================================================================

class LinearModel(DistillModel):
    """
    Original per-joint linear model.

    Kept here for compatibility with the rest of the project.
    """

    FEATURE_NAMES = (
        [
            "target_current",
            "qd",
            "qdd",
            "pos",
            "vel",
            "acc",
        ]
        + [
            f"is_{n}"
            for n in JOINT_NAMES
        ]
    )

    def __init__(self):
        self.coef = None
        self.vel_range = None
        self.acc_range = None

    def predicts(self) -> list[str]:
        return [
            "actual_current"
        ]

    def _row_features(
        self,
        joint,
        tgt_i,
        pos,
        qd,
        qdd,
        vel,
        acc,
    ):
        tgt_i = np.asarray(tgt_i)
        pos = np.asarray(pos)
        qd = np.asarray(qd)
        qdd = np.asarray(qdd)
        vel = np.asarray(vel)
        acc = np.asarray(acc)

        onehot = np.zeros(
            (
                len(pos),
                N_JOINTS,
            )
        )

        onehot[:, joint] = 1.0

        return np.column_stack(
            [
                tgt_i,
                qd,
                qdd,
                pos,
                vel,
                acc,
                onehot,
            ]
        )

    def _design(self, recordings):

        X = []
        y = []

        for rec in recordings:

            if (
                rec.vel_cmd is None
                or rec.acc_cmd is None
            ):
                raise ValueError(
                    f"{rec.path} has no vel/acc registers"
                )

            qdd = get_block(
                rec.df,
                "target_qdd",
            )

            for j in range(N_JOINTS):

                X.append(
                    self._row_features(
                        j,
                        rec.target_current[:, j],
                        rec.target_q[:, j],
                        rec.target_qd[:, j],
                        qdd[:, j],
                        rec.vel_cmd,
                        rec.acc_cmd,
                    )
                )

                y.append(
                    rec.actual_current[:, j]
                )

        return (
            np.vstack(X),
            np.concatenate(y),
        )

    def fit(self, recordings):

        X, y = self._design(
            recordings
        )

        self.coef, *_ = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )

        self.vel_range = (
            float(X[:, 4].min()),
            float(X[:, 4].max()),
        )

        self.acc_range = (
            float(X[:, 5].min()),
            float(X[:, 5].max()),
        )

        return self

    def predict(self, df):

        target_current = get_block(
            df,
            "target_current",
        )

        q = get_block(
            df,
            "target_q",
        )

        qd = get_block(
            df,
            "target_qd",
        )

        qdd = get_block(
            df,
            "target_qdd",
        )

        vel = df[
            VEL_COL
        ].to_numpy(
            dtype=float
        )

        acc = df[
            ACC_COL
        ].to_numpy(
            dtype=float
        )

        out = np.zeros_like(q)

        for j in range(N_JOINTS):

            X = self._row_features(
                j,
                target_current[:, j],
                q[:, j],
                qd[:, j],
                qdd[:, j],
                vel,
                acc,
            )

            out[:, j] = (
                X @ self.coef
            )

        return {
            "actual_current": out
        }

    def bounds(self):

        if self.coef is None:
            return None

        return (
            self.vel_range,
            self.acc_range,
        )


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
        # target_q       = 6
        # target_qd      = 6
        # target_qdd     = 6
        # target_current = 6
        # vel            = 1
        # acc            = 1
        #
        # Total = 26
        # --------------------------------------------------------------

        input_size = (
            n_joints * 4 + 2
        )

        # Predict actual_q for every joint.

        output_size = n_joints

        self.modell = nn.Sequential(

            nn.Linear(
                input_size,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
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
    Neural network that predicts actual joint positions.

    One row corresponds to one timestep.

    Input:

        target_q[0:6]
        target_qd[0:6]
        target_qdd[0:6]
        target_current[0:6]
        vel
        acc

    Output:

        actual_q[0:6]
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
        + [
            f"target_qdd{i}"
            for i in range(N_JOINTS)
        ]
        + [
            f"target_current{i}"
            for i in range(N_JOINTS)
        ]
        + [
            "vel",
            "acc",
        ]
    )

    def __init__(self):

        self.vel_range = None
        self.acc_range = None

        self.modell = SimpleNN(
            N_JOINTS
        )

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def predicts(self) -> list[str]:

        return [
            "actual_q"
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
        vel,
        acc,
    ) -> np.ndarray:
        """
        Build one feature row for every timestep.

        Input shapes:

            target_q       : (n, 6)
            target_qd      : (n, 6)
            target_qdd     : (n, 6)
            target_current : (n, 6)
            vel            : (n,)
            acc            : (n,)

        Output:

            (n, 26)
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

        One CSV row = one training sample.

        X shape:

            (number_of_rows, 26)

        y shape:

            (number_of_rows, 6)
        """

        X = []
        y = []

        for rec in recordings:

            if rec.vel_cmd is None:
                raise ValueError(
                    f"{rec.path} has no vel register; "
                    "record with "
                    "`--float-register 1 vel 2 acc`"
                )

            if rec.acc_cmd is None:
                raise ValueError(
                    f"{rec.path} has no acc register; "
                    "record with "
                    "`--float-register 1 vel 2 acc`"
                )

            # ----------------------------------------------------------
            # Use target acceleration directly from the CSV.
            #
            # Shape:
            #
            #     (number_of_rows, 6)
            # ----------------------------------------------------------

            target_qdd = get_block(
                rec.df,
                "target_qdd",
            )

            # ----------------------------------------------------------
            # Create ALL-joint input vectors.
            # ----------------------------------------------------------

            features = self._row_features(
                rec.target_q,
                rec.target_qd,
                target_qdd,
                rec.target_current,
                rec.vel_cmd,
                rec.acc_cmd,
            )

            X.append(
                features
            )

            # ----------------------------------------------------------
            # Target is ACTUAL POSITION.
            #
            # Shape:
            #
            #     (number_of_rows, 6)
            # ----------------------------------------------------------

            targets = np.asarray(
                rec.actual_q,
                dtype=np.float32,
            )

            y.append(
                targets
            )

        # --------------------------------------------------------------
        # Combine recordings.
        # --------------------------------------------------------------

        X = np.vstack(X)

        y = np.vstack(y)

        print()
        print(
            f"Designed dataset:"
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

        One batch contains a number of CSV rows.

        The rows are shuffled before every epoch.
        """

        # --------------------------------------------------------------
        # Training settings
        # --------------------------------------------------------------

        epochs = 100
        batch_size = 100

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
            N_JOINTS * 4 + 2
        )

        if X.shape[1] != expected_input_size:

            raise ValueError(
                f"Expected "
                f"{expected_input_size} "
                f"input features, "
                f"but got {X.shape[1]}"
            )

        if y.shape[1] != N_JOINTS:

            raise ValueError(
                f"Expected "
                f"{N_JOINTS} output values, "
                f"but got {y.shape[1]}"
            )

        # --------------------------------------------------------------
        # Save velocity/acceleration ranges.
        #
        # They are the final two columns.
        # --------------------------------------------------------------

        self.vel_range = (
            float(
                X[:, -2].min()
            ),
            float(
                X[:, -2].max()
            ),
        )

        self.acc_range = (
            float(
                X[:, -1].min()
            ),
            float(
                X[:, -1].max()
            ),
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
        #
        # Prediction:
        #
        #     (batch, 6)
        #
        # Target:
        #
        #     (batch, 6)
        # --------------------------------------------------------------

        criterion = nn.MSELoss()

        # --------------------------------------------------------------
        # Optimizer.
        # --------------------------------------------------------------

        optimizer = torch.optim.Adam(
            self.modell.parameters(),
            lr=0.001,
        )

        # --------------------------------------------------------------
        # Automatically reduce learning rate when the loss stagnates.
        # --------------------------------------------------------------

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
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
                #
                # DO NOT unsqueeze y_batch.
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
                f"Loss: {average_loss:.8f} | "
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
        Predict actual joint positions.

        Input:

            n rows x 26 features

        Output:

            n rows x 6 actual positions
        """

        # --------------------------------------------------------------
        # Get all target data.
        # --------------------------------------------------------------

        target_current = get_block(
            df,
            "target_current",
        )

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

        # --------------------------------------------------------------
        # Get velocity and acceleration commands.
        # --------------------------------------------------------------

        vel = df[
            VEL_COL
        ].to_numpy(
            dtype=float
        )

        acc = df[
            ACC_COL
        ].to_numpy(
            dtype=float
        )

        # --------------------------------------------------------------
        # Construct exactly the same features used during training.
        # --------------------------------------------------------------

        X = self._row_features(
            target_q,
            target_qd,
            target_qdd,
            target_current,
            vel,
            acc,
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

        # --------------------------------------------------------------
        # Verify output shape.
        # --------------------------------------------------------------

        expected_shape = (
            len(df),
            N_JOINTS,
        )

        if prediction.shape != expected_shape:

            raise RuntimeError(
                f"Model returned "
                f"{prediction.shape}, "
                f"expected "
                f"{expected_shape}"
            )

        return {
            "actual_q": prediction
        }

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def bounds(self):

        if self.vel_range is None:
            return None

        if self.acc_range is None:
            return None

        return (
            self.vel_range,
            self.acc_range,
        )


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

def evaluate_holdout(
    model: OwnModel,
    recordings,
    holdout: float,
):
    """
    Evaluate the already-trained model.

    IMPORTANT:

    This is only a simple row-based holdout evaluation.

    For a proper robotics evaluation, it is better to hold out complete
    recordings rather than randomly selected rows because consecutive
    robot samples are highly correlated.
    """

    if not (
        0.0 < holdout < 1.0
    ):
        raise ValueError(
            "holdout must be between 0 and 1"
        )

    # --------------------------------------------------------------
    # Build dataset.
    # --------------------------------------------------------------

    X, y = model._design(
        recordings
    )

    # --------------------------------------------------------------
    # Deterministic row split.
    # --------------------------------------------------------------

    step = max(
        int(
            round(
                1.0 / holdout
            )
        ),
        2,
    )

    is_test = (
        np.arange(
            len(X)
        ) % step == 0
    )

    X_test = X[
        is_test
    ]

    y_test = y[
        is_test
    ]

    # --------------------------------------------------------------
    # Predict.
    # --------------------------------------------------------------

    model.modell.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():

        prediction = (
            model.modell(
                X_tensor
            )
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------------
    # Error.
    # --------------------------------------------------------------

    error = (
        prediction -
        y_test
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    denominator = np.sum(
        (
            y_test -
            y_test.mean()
        ) ** 2
    )

    if denominator > 0:

        r2 = float(
            1.0 -
            np.sum(
                error ** 2
            ) /
            denominator
        )

    else:

        r2 = float("nan")

    print(
        f"Held-out rows: "
        f"{is_test.sum()}"
    )

    print(
        f"actual_q RMSE: "
        f"{rmse:.6f}"
    )

    print(
        f"actual_q R2: "
        f"{r2:.6f}"
    )

    # --------------------------------------------------------------
    # Per-joint RMSE.
    # --------------------------------------------------------------

    print()
    print(
        "Per-joint RMSE:"
    )

    for j in range(
        N_JOINTS
    ):

        joint_error = (
            prediction[:, j] -
            y_test[:, j]
        )

        joint_rmse = float(
            np.sqrt(
                np.mean(
                    joint_error ** 2
                )
            )
        )

        print(
            f"  joint {j}: "
            f"{joint_rmse:.6f}"
        )


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

    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help=(
            "Fraction of rows used "
            "for holdout evaluation."
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
    # Evaluate.
    # --------------------------------------------------------------

    evaluate_holdout(
        model,
        recordings,
        args.holdout,
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