"""
Distilled neural-network model that predicts ``actual_current`` channels
for a commanded trajectory.

One CSV row = one training sample.

For every timestep, the model receives information about ALL joints:

    target_q0 ... target_q5
    target_qd0 ... target_qd5
    target_qdd0 ... target_qdd5
    target_current0 ... target_current5
    vel
    acc

For N_JOINTS = 6 this gives:

    6 + 6 + 6 + 6 + 1 + 1 = 26 input features

The network predicts:

    actual_current0 ... actual_current5

so the output has N_JOINTS values.

The model interface remains compatible with the rest of the pipeline:

    fit(recordings)
    predicts()
    predict(df)
    bounds()
    save(path)
"""

from __future__ import annotations

import argparse
import glob
import pickle
from abc import ABC, abstractmethod

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
# Base interface
# ============================================================================

class DistillModel(ABC):
    """
    Interface every distilled model must implement.
    """

    @abstractmethod
    def fit(self, recordings) -> "DistillModel":
        """Train on recorded real robot runs."""

    @abstractmethod
    def predicts(self) -> list[str]:
        """
        Return the actual_* channel bases that this model predicts.

        Example:
            ["actual_current"]
        """

    @abstractmethod
    def predict(self, df) -> dict:
        """
        Predict actual channels for a commanded-trajectory DataFrame.

        Returns:

            {
                "actual_current": array of shape
                    (n_rows, N_JOINTS)
            }
        """

    def bounds(self):
        """
        Optional training-data bounds.

        Returns:

            (
                (vel_low, vel_high),
                (acc_low, acc_high)
            )

        or None.
        """
        return None

    def save(self, path: str):
        """Save the complete model using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "DistillModel":
        """Load a saved model."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================================
# Original linear baseline
# ============================================================================

class LinearModel(DistillModel):
    """
    Original linear baseline.

    This is kept so the rest of the project can still import LinearModel,
    but OwnModel below is the new all-joints neural-network model.
    """

    FEATURE_NAMES = (
        ["target_current", "qd", "qdd", "pos", "vel", "acc"]
        + [f"is_{n}" for n in JOINT_NAMES]
    )

    def __init__(self):
        self.coef = None
        self.vel_range = None
        self.acc_range = None

    def predicts(self) -> list[str]:
        return ["actual_current"]

    def _row_features(
        self,
        joint: int,
        tgt_i,
        pos,
        qd,
        qdd,
        vel,
        acc,
    ) -> np.ndarray:

        tgt_i = np.asarray(tgt_i)
        pos = np.asarray(pos)
        qd = np.asarray(qd)
        qdd = np.asarray(qdd)
        vel = np.asarray(vel)
        acc = np.asarray(acc)

        onehot = np.zeros(
            (len(pos), N_JOINTS)
        )

        onehot[:, joint] = 1.0

        return np.column_stack([
            tgt_i,
            qd,
            qdd,
            pos,
            vel,
            acc,
            onehot,
        ])

    def _design(self, recordings):

        X = []
        y = []

        for rec in recordings:

            if rec.vel_cmd is None or rec.acc_cmd is None:
                raise ValueError(
                    f"{rec.path} has no vel/acc registers; "
                    "record with `--float-register 1 vel 2 acc`"
                )

            qdd = np.gradient(
                rec.target_qd,
                rec.dt,
                axis=0,
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

    def fit(self, recordings) -> "LinearModel":

        X, y = self._design(recordings)

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

    def predict(self, df) -> dict:

        dt = frame_dt(df)

        ti = get_block(
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

        qdd = np.gradient(
            qd,
            dt,
            axis=0,
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
                ti[:, j],
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

        input_size = (
            n_joints * 4 + 2
        )

        output_size = n_joints

        self.modell = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 6)
        )

    def forward(self, x):

        return self.modell(x)


# ============================================================================
# New neural-network model
# ============================================================================

class OwnModel(DistillModel):
    """
    Neural network using ALL joints as input.

    One row of the dataset corresponds to one timestep.

    Input for N_JOINTS = 6:

        [q0 ... q5,
         qd0 ... qd5,
         qdd0 ... qdd5,
         target_current0 ... target_current5,
         vel,
         acc]

    Total:

        6 + 6 + 6 + 6 + 1 + 1 = 26

    Output:

        [actual_current0 ... actual_current5]

    Total:

        6 outputs
    """

    FEATURE_NAMES = (
        [f"q{i}" for i in range(N_JOINTS)]
        + [f"qd{i}" for i in range(N_JOINTS)]
        + [f"qdd{i}" for i in range(N_JOINTS)]
        + [f"target_current{i}" for i in range(N_JOINTS)]
        + ["vel", "acc"]
    )

    def __init__(self):

        self.vel_range = None
        self.acc_range = None

        self.modell = SimpleNN(
            N_JOINTS
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def predicts(self) -> list[str]:

        return [
            "actual_current"
        ]

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _row_features(
        self,
        tgt_i,
        pos,
        qd,
        qdd,
        vel,
        acc,
    ) -> np.ndarray:
        """
        Construct features for ALL joints.

        Every row represents one timestep.

        For N_JOINTS = 6:

            pos       -> 6
            qd        -> 6
            qdd       -> 6
            tgt_i     -> 6
            vel       -> 1
            acc       -> 1

            total     -> 26
        """

        tgt_i = np.asarray(
            tgt_i,
            dtype=np.float32,
        )

        pos = np.asarray(
            pos,
            dtype=np.float32,
        )

        qd = np.asarray(
            qd,
            dtype=np.float32,
        )

        qdd = np.asarray(
            qdd,
            dtype=np.float32,
        )

        vel = np.asarray(
            vel,
            dtype=np.float32,
        ).reshape(-1, 1)

        acc = np.asarray(
            acc,
            dtype=np.float32,
        ).reshape(-1, 1)

        return np.column_stack([
            pos,
            qd,
            qdd,
            tgt_i,
            vel,
            acc,
        ]).astype(
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
        Convert recordings into a training dataset.

        IMPORTANT:

        One CSV row = one training sample.

        X:

            (number_of_rows, N_JOINTS * 4 + 2)

        y:

            (number_of_rows, N_JOINTS)
        """

        X = []
        y = []

        for rec in recordings:

            if rec.vel_cmd is None:
                raise ValueError(
                    f"{rec.path} has no vel register; "
                    "record with `--float-register 1 vel 2 acc`"
                )

            if rec.acc_cmd is None:
                raise ValueError(
                    f"{rec.path} has no acc register; "
                    "record with `--float-register 1 vel 2 acc`"
                )

            # ----------------------------------------------------------
            # Calculate commanded acceleration.
            #
            # Shape:
            #
            #     (n_rows, N_JOINTS)
            # ----------------------------------------------------------

            qdd = np.gradient(
                rec.target_qd,
                rec.dt,
                axis=0,
            )

            # ----------------------------------------------------------
            # Build one feature vector per CSV row.
            #
            # Shape:
            #
            #     (n_rows, 26)
            #
            # for N_JOINTS = 6.
            # ----------------------------------------------------------

            features = self._row_features(
                rec.target_current,
                rec.target_q,
                rec.target_qd,
                qdd,
                rec.vel_cmd,
                rec.acc_cmd,
            )

            X.append(
                features
            )

            # ----------------------------------------------------------
            # Target:
            #
            #     actual_current0 ... actual_current5
            #
            # Shape:
            #
            #     (n_rows, N_JOINTS)
            # ----------------------------------------------------------

            y.append(
                np.asarray(
                    rec.actual_current,
                    dtype=np.float32,
                )
            )

        # --------------------------------------------------------------
        # Combine recordings
        # --------------------------------------------------------------

        X = np.vstack(X)

        y = np.vstack(y)

        print(
            f"Designed dataset: "
            f"X={X.shape}, y={y.shape}"
        )

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

        batch_size refers to NUMBER OF CSV ROWS.

        It does NOT refer to number of recordings.
        """

        batch_size = 2048
        epochs = 250

        # --------------------------------------------------------------
        # Build complete dataset.
        #
        # This is done ONCE.
        # --------------------------------------------------------------

        X, y = self._design(
            recordings
        )

        print()
        print("Training dataset:")
        print(
            f"X = {X.shape}"
        )
        print(
            f"y = {y.shape}"
        )
        print()

        # --------------------------------------------------------------
        # Check dimensions
        # --------------------------------------------------------------

        expected_input_size = (
            N_JOINTS * 4 + 2
        )

        if X.shape[1] != expected_input_size:

            raise ValueError(
                f"Expected {expected_input_size} input features, "
                f"but got {X.shape[1]}"
            )

        if y.shape[1] != N_JOINTS:

            raise ValueError(
                f"Expected {N_JOINTS} output values, "
                f"but got {y.shape[1]}"
            )

        # --------------------------------------------------------------
        # Ranges for vel / acc
        #
        # They are the LAST two columns.
        # --------------------------------------------------------------

        self.vel_range = (
            float(X[:, -2].min()),
            float(X[:, -2].max()),
        )

        self.acc_range = (
            float(X[:, -1].min()),
            float(X[:, -1].max()),
        )

        # --------------------------------------------------------------
        # Convert to torch tensors
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
        # Network
        # --------------------------------------------------------------

        self.modell = SimpleNN(
            N_JOINTS
        )

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.modell.parameters(),
            lr=0.001,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------

        for epoch in range(epochs):

            self.modell.train()

            permutation = torch.randperm(len(X))

            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            total_loss = 0.0
            number_of_batches = 0

            for batch_start in range(
                0,
                len(X_shuffled),
                batch_size
            ):

                X_batch = X_shuffled[
                    batch_start:batch_start + batch_size
                ]

                y_batch = y_shuffled[
                    batch_start:batch_start + batch_size
                ]

                optimizer.zero_grad()

                prediction = self.modell(X_batch)

                loss = criterion(
                    prediction,
                    y_batch
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()
                number_of_batches += 1

            average_loss = (
                total_loss / number_of_batches
            )

            # Reduce LR if loss has stopped improving
            scheduler.step(average_loss)

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {average_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

            average_loss = (
                total_loss /
                max(number_of_batches, 1)
            )

            print(
                f"  Loss: {average_loss:.6f}"
            )

        print("DONE")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df,
    ) -> dict:
        """
        Predict actual_current for every row and every joint.

        Input:

            n rows x 26 features

        Output:

            n rows x 6 actual currents
        """

        # --------------------------------------------------------------
        # Time step
        # --------------------------------------------------------------

        dt = frame_dt(
            df
        )

        # --------------------------------------------------------------
        # Get all joint data
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

        # --------------------------------------------------------------
        # Commanded acceleration
        # --------------------------------------------------------------

        qdd = np.gradient(
            target_qd,
            dt,
            axis=0,
        )

        # --------------------------------------------------------------
        # MoveJ velocity / acceleration
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
        # Build ALL-joint input matrix.
        # --------------------------------------------------------------

        X = self._row_features(
            target_current,
            target_q,
            target_qd,
            qdd,
            vel,
            acc,
        )

        # --------------------------------------------------------------
        # Convert to torch
        # --------------------------------------------------------------

        X_tensor = torch.from_numpy(
            X
        ).float()

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        self.modell.eval()

        with torch.no_grad():

            prediction = self.modell(
                X_tensor
            )

        prediction = (
            prediction
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------------
        # Check result
        # --------------------------------------------------------------

        expected_shape = (
            len(df),
            N_JOINTS,
        )

        if prediction.shape != expected_shape:

            raise RuntimeError(
                f"Model returned shape "
                f"{prediction.shape}, "
                f"expected {expected_shape}"
            )

        return {
            "actual_current": prediction
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
    Overwrite actual_* columns with model predictions.

    The dataframe is transformed using the same preprocessing used during
    training, predictions are generated, and then preprocessing is reverted.
    """

    pre = pre or Identity()

    # --------------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------------

    df = pd.read_csv(
        csv
    )

    # --------------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------------

    df = pre.transform_distill(
        df
    )

    # --------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------

    preds = model.predict(
        df
    )

    # --------------------------------------------------------------
    # Write predictions into actual_* columns
    # --------------------------------------------------------------

    for base in model.predicts():

        set_block(
            df,
            base,
            preds[base],
        )

    # --------------------------------------------------------------
    # Convert back to original units
    # --------------------------------------------------------------

    df = pre.revert_distill(
        df
    )

    # --------------------------------------------------------------
    # Save
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
    Evaluate the neural network on a deterministic held-out subset.

    The split is performed on rows.

    IMPORTANT:

    We don't train another model here. We simply use the already-trained
    neural network to predict the held-out rows.
    """

    if not (
        0.0 < holdout < 1.0
    ):
        raise ValueError(
            "holdout must be between 0 and 1"
        )

    # --------------------------------------------------------------
    # Build complete dataset
    # --------------------------------------------------------------

    X, y = model._design(
        recordings
    )

    # --------------------------------------------------------------
    # Deterministic split
    # --------------------------------------------------------------

    step = max(
        int(round(1.0 / holdout)),
        2,
    )

    is_test = (
        np.arange(len(X)) % step == 0
    )

    X_test = X[
        is_test
    ]

    y_test = y[
        is_test
    ]

    # --------------------------------------------------------------
    # Predict
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
    # Error
    # --------------------------------------------------------------

    error = (
        prediction - y_test
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
            np.sum(error ** 2) /
            denominator
        )

    else:

        r2 = float("nan")

    print(
        f"held-out "
        f"({is_test.sum()} rows): "
        f"actual_current RMSE "
        f"{rmse:.3f} A   "
        f"R2 {r2:.3f}"
    )

    # --------------------------------------------------------------
    # Per-joint error
    # --------------------------------------------------------------

    print()
    print("Per-joint RMSE:")

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
            f"{joint_rmse:.4f} A"
        )


# ============================================================================
# Main
# ============================================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Train the neural-network "
            "distillation model."
        )
    )

    ap.add_argument(
        "--csvs",
        nargs="+",
        default=sorted(
            glob.glob(
                "data/test-*.csv"
            )
        ),
        help=(
            "recorded runs to train on"
        ),
    )

    ap.add_argument(
        "--out",
        default="models/distill.pkl",
        help="pickle path",
    )

    ap.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help=(
            "fraction of rows held out "
            "for the error report"
        ),
    )

    args = ap.parse_args()

    # --------------------------------------------------------------
    # Import using the real module name.
    #
    # This makes the saved pickle load correctly in train_rla.py
    # and run.py.
    # --------------------------------------------------------------

    from train_distillation_model_own import OwnModel
    from analysis import Recording

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------

    pre = default_preprocess()

    recordings = [
        Recording(
            r.path,
            df=pre.transform_distill(
                r.df
            ),
        )
        for r in (
            Recording(p)
            for p in args.csvs
        )
    ]

    print(
        f"Loaded {len(recordings)} recordings"
    )

    # --------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------

    model = OwnModel()

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    model.fit(
        recordings
    )

    # --------------------------------------------------------------
    # Holdout evaluation
    #
    # This happens AFTER training, using the neural network.
    # --------------------------------------------------------------

    evaluate_holdout(
        model,
        recordings,
        args.holdout,
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    model.save(
        args.out
    )

    print(
        f"saved {args.out}"
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()