"""
Neural-network distillation model.

One CSV row = one training sample.

The network uses multiple temporal frames:

    t-20
    t-10
    t-5
    t-2
    t-1
    t

The prediction is always made for the CURRENT frame t.

INPUTS PER TIMEFRAME:

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

Total per timeframe:

    44 input features

Temporal frames:

    t-20
    t-10
    t-5
    t-2
    t-1
    t

Total:

    44 * 6 = 264 input features

OUTPUTS:

    error_q0 ... error_q5

where:

    error_q = target_q - actual_q

Total outputs:

    6

Network:

    264 -> 80 -> 80 -> 80 -> 64 -> 32 -> 32 -> 32 -> 32 -> 6
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
# Temporal configuration
# ============================================================================

# Exact historical frames used by the model.
#
# For prediction at time t:
#
#     t-20
#     t-10
#     t-5
#     t-2
#     t-1
#     t
#
# The ordering is from oldest to newest.
#
TIME_OFFSETS = (
    20,
    10,
    5,
    2,
    1,
    0,
)

N_TIMEFRAMES = len(
    TIME_OFFSETS
)

MAX_HISTORY = max(
    TIME_OFFSETS
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
                384,
            ),

            nn.Tanh(),

            nn.Linear(
                384,
                384,
            ),

            nn.Tanh(),

            nn.Linear(
                384,
                200,
            ),

            nn.Tanh(),

            nn.Linear(
                200,
                150,
            ),

            nn.Tanh(),

            nn.Linear(
                150,
                64,
            ),

            nn.Tanh(),

            nn.Linear(
                64,
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

    The model uses six temporal frames:

        t-20
        t-10
        t-5
        t-2
        t-1
        t

Each timeframe contains:

        target_q
        target_qd
        target_qdd
        target_current
        target_moment
        vel
        acc

There are 44 features per timeframe.

Therefore:

        44 * 6 = 264 input features

Output:

        error_q

where:

        error_q(t) = target_q(t) - actual_q(t)
    """

    # ------------------------------------------------------------------
    # Feature names for ONE timeframe
    # ------------------------------------------------------------------

    FRAME_FEATURE_NAMES = (
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

    FRAME_INPUT_SIZE = len(
        FRAME_FEATURE_NAMES
    )

    # ------------------------------------------------------------------
    # Feature names for ALL temporal frames
    # ------------------------------------------------------------------

    FEATURE_NAMES = []

    for offset in TIME_OFFSETS:

        if offset == 0:
            prefix = "t"
        else:
            prefix = f"t-{offset}"

        for name in FRAME_FEATURE_NAMES:
            FEATURE_NAMES.append(f"{prefix}_{name}")
        

    FEATURE_NAMES = tuple(
        FEATURE_NAMES
    )

    # ------------------------------------------------------------------
    # Total input size
    # ------------------------------------------------------------------

    INPUT_SIZE = len(
        FEATURE_NAMES
    )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

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
    # Feature construction for ONE timeframe
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
        Build features for ONE timeframe.

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
    # Temporal feature construction
    # ------------------------------------------------------------------

    def _temporal_features(
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
        Construct temporal input windows.

        For every valid prediction at time t:

            [features(t-20),
             features(t-10),
             features(t-5),
             features(t-2),
             features(t-1),
             features(t)]

        Only rows with complete history are returned.

        Therefore the first MAX_HISTORY rows are skipped.

        Output:

            (n - MAX_HISTORY, 264)
        """

        # --------------------------------------------------------------
        # Build normal per-frame features.
        #
        # Shape:
        #
        #     (n, 44)
        # --------------------------------------------------------------

        frame_features = self._row_features(
            target_q,
            target_qd,
            target_qdd,
            target_current,
            target_moment,
            vel,
            acc,
        )

        n_rows = len(
            frame_features
        )

        # --------------------------------------------------------------
        # Not enough rows to create even one complete temporal window.
        # --------------------------------------------------------------

        if n_rows <= MAX_HISTORY:

            return np.empty(
                (
                    0,
                    self.INPUT_SIZE,
                ),
                dtype=np.float32,
            )

        # --------------------------------------------------------------
        # Build temporal windows.
        # --------------------------------------------------------------

        temporal = []

        # --------------------------------------------------------------
        # Start at MAX_HISTORY because t-20 is the oldest required
        # frame.
        # --------------------------------------------------------------

        for t in range(
            MAX_HISTORY,
            n_rows,
        ):

            window = []

            # ----------------------------------------------------------
            # Add:
            #
            #     t-20
            #     t-10
            #     t-5
            #     t-2
            #     t-1
            #     t
            # ----------------------------------------------------------

            for offset in TIME_OFFSETS:

                index = t - offset

                window.append(
                    frame_features[index]
                )

            # ----------------------------------------------------------
            # Flatten the six frames into one vector.
            # ----------------------------------------------------------

            temporal.append(
                np.concatenate(
                    window
                )
            )

        return np.asarray(
            temporal,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _design(
        self,
        recordings,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert recordings into one large temporal dataset.

        X:

            264 features:

                44 features at t-20
                44 features at t-10
                44 features at t-5
                44 features at t-2
                44 features at t-1
                44 features at t

        y:

            6 position errors at the CURRENT frame t.

        The first MAX_HISTORY rows of every recording are excluded
        because they do not have enough history.
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
            # Construct temporal input.
            #
            # IMPORTANT:
            #
            # This is performed independently for every recording.
            #
            # Therefore history NEVER crosses from one CSV into another.
            # ----------------------------------------------------------

            features = self._temporal_features(
                rec.target_q,
                rec.target_qd,
                target_qdd,
                rec.target_current,
                target_moment,
                rec.vel_cmd,
                rec.acc_cmd,
            )

            # ----------------------------------------------------------
            # Target:
            #
            #     target_q(t) - actual_q(t)
            #
            # Only rows from MAX_HISTORY onward are valid because
            # those are the rows represented in X.
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

            # ----------------------------------------------------------
            # Match y to X.
            #
            # X corresponds to:
            #
            #     t = MAX_HISTORY ... end
            # ----------------------------------------------------------

            error = error[
                MAX_HISTORY:
            ]

            # ----------------------------------------------------------
            # Make sure the recording actually produced valid samples.
            # ----------------------------------------------------------

            if len(features) == 0:

                print(
                    f"WARNING: "
                    f"recording {rec.path} "
                    f"does not contain enough rows "
                    f"for the temporal history."
                )

                continue

            if len(features) != len(error):

                raise ValueError(
                    f"Temporal feature/target length mismatch "
                    f"for {rec.path}: "
                    f"X={len(features)}, "
                    f"y={len(error)}"
                )

            X.append(
                features
            )

            y.append(
                error
            )

        # --------------------------------------------------------------
        # Make sure at least one recording produced data.
        # --------------------------------------------------------------

        if not X:

            raise ValueError(
                "No valid training samples were produced. "
                f"Each recording needs more than "
                f"{MAX_HISTORY} rows."
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
            f"  Features per timeframe: "
            f"{self.FRAME_INPUT_SIZE}"
        )

        print(
            f"  Number of timeframes: "
            f"{N_TIMEFRAMES}"
        )

        print(
            "  Temporal offsets: "
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
            f"  Total input features: "
            f"{X.shape[1]}"
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

        epochs = 350

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
                patience=6,
                min_lr=1e-10,
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
            #
            # The temporal windows have already been constructed, so
            # shuffling the complete windows is safe here.
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

            if epoch % 20 == 0:
                print("[TECHO] "
                f"Epoch {epoch:3d} | "
                f"Loss: {average_loss:.9f} | "
                f"LR: {current_lr:.8f}"
                )
            else:
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

        For every valid row t, the model uses:

            t-20
            t-10
            t-5
            t-2
            t-1
            t

The first MAX_HISTORY rows do not have enough history.

Therefore the returned prediction has shape:

    (number_of_rows, 6)

but the first MAX_HISTORY rows contain NaN.

Output:

    error_q
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
        # Build all valid temporal features.
        #
        # Output contains rows:
        #
        #     MAX_HISTORY ... n-1
        # --------------------------------------------------------------

        X = self._temporal_features(
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

        # --------------------------------------------------------------
        # Create full-length output.
        #
        # The first MAX_HISTORY rows cannot be predicted because
        # they don't have enough history.
        # --------------------------------------------------------------

        full_prediction = np.full(
            (
                len(df),
                N_JOINTS,
            ),
            np.nan,
            dtype=np.float32,
        )

        full_prediction[
            MAX_HISTORY:
        ] = prediction

        return {
            "error_q": full_prediction
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

        error_q(t) = target_q(t) - actual_q(t)

    using:

        t-20
        t-10
        t-5
        t-2
        t-1
        t

The prediction is stored in:

    error_q0 ... error_q5

The first MAX_HISTORY rows contain NaN because there is not enough
historical data to make a prediction.
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
            "for joint position error "
            "using temporal history."
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
        f"Features per timeframe: "
        f"{model.FRAME_INPUT_SIZE}"
    )

    print(
        f"Number of timeframes: "
        f"{N_TIMEFRAMES}"
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
        f"Total model inputs: "
        f"{model.INPUT_SIZE}"
    )

    print()

    for i, name in enumerate(
        model.FEATURE_NAMES
    ):

        print(
            f"  {i:3d}: {name}"
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