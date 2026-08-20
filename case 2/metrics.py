"""Score a recording's rows into one per-row value the optimizer minimizes.

``EvaluationMetric`` is the interface:

    needs()      -> per-joint channel bases this metric reads
    per_row(df)  -> (n,) score, one value per row

The default ``CurrentGapMetric`` is the current-tracking gap. A subclass can read
other channels (position error, jerk, a mix); each must be a channel the recording
carries or the distill model predicts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from utils import get_block, joint_cols

SCORE_COL = "score"


class EvaluationMetric(ABC):
    """Per-row score to minimize. Implement ``needs`` and ``per_row``."""

    @abstractmethod
    def needs(self) -> list[str]:
        """Per-joint channel bases this metric reads, e.g. ``["target_current",
        "actual_current"]``. Each must be a recording channel or one the distill
        model predicts, or ``add_score`` raises.
        """

    @abstractmethod
    def per_row(self, df) -> np.ndarray:
        """Score for every row of ``df``, shape ``(n,)``."""


class CurrentGapMetric(EvaluationMetric):
    """Current-tracking gap: ``|actual_current - target_current|`` summed over joints."""

    def needs(self) -> list[str]:
        return ["target_current", "actual_current"]

    def per_row(self, df) -> np.ndarray:
        gap = np.abs(get_block(df, "actual_current") - get_block(df, "target_current"))
        return gap.sum(axis=1)

class PositionErrorMetric(EvaluationMetric):
    """Position-tracking error: ``|actual_q - target_q|`` summed over joints.

    Per-row absolute position error (rad). Aggregating these rows over a move's
    settle window ``[i1:i2]`` with ``max()`` gives peak overshoot, with RMS gives
    the RMS position error — the two vibration metrics the case defines.
    """

    def needs(self) -> list[str]:
        return ["target_q", "actual_q"]

    def per_row(self, df) -> np.ndarray:
        err = np.abs(get_block(df, "actual_q") - get_block(df, "target_q"))
        return err.sum(axis=1)


class ErrorQMetric(EvaluationMetric):
    """Position-tracking error read straight from a predicted ``error_q`` channel.

    Same quantity as ``PositionErrorMetric``, but for distill models that predict
    the tracking error directly instead of the reached position. The distilled NNs
    (``train_distillation_model_error*.py``) are trained on

        error_q = target_q - actual_q

    so the per-row score is ``|error_q|`` summed over joints, with no conversion
    to ``actual_q`` and back. Taking the absolute value makes the sign convention
    irrelevant: a model trained on ``actual_q - target_q`` scores identically.

    Use this when ``model.predicts()`` contains ``error_q``; use
    ``PositionErrorMetric`` when it contains ``actual_q``.
    """

    def needs(self) -> list[str]:
        return ["error_q"]

    def per_row(self, df) -> np.ndarray:
        return np.abs(get_block(df, "error_q")).sum(axis=1)


def default_metric(model) -> EvaluationMetric:
    """The metric that reads what ``model`` predicts.

    A distill model and a metric have to agree on a channel: the model fills the
    ``actual_*``/``error_*`` columns the metric then scores. Pick the metric from
    the model's own ``predicts()`` so swapping the model does not silently leave a
    mismatched metric behind (``_MoveEnv`` raises on that).

        error_q         -> ErrorQMetric        (the distilled NNs)
        actual_q        -> PositionErrorMetric
        actual_current  -> CurrentGapMetric    (LinearModel, the baseline)
    """
    predicts = model.predicts()
    for base, metric in (("error_q", ErrorQMetric),
                         ("actual_q", PositionErrorMetric),
                         ("actual_current", CurrentGapMetric)):
        if base in predicts:
            return metric()
    raise ValueError(f"no metric reads any of {predicts}; add one to metrics.py "
                     f"and pick it here")


def add_score(df, metric: EvaluationMetric):
    """Return ``df`` with a ``score`` column from ``metric``.

    Checks the channels the metric needs are present, so a missing column raises
    here rather than later.
    """
    missing = [c for base in metric.needs() for c in joint_cols(base) if c not in df]
    if missing:
        raise ValueError(f"metric needs columns not in the recording: {missing}")
    df = df.copy()                          # defragment: recordings are very wide
    df[SCORE_COL] = metric.per_row(df)
    return df
