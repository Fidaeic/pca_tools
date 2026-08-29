"""Auditable Phase-I reference-data curation for PCA-MSPC."""

from dataclasses import dataclass
import logging
import math
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from .exceptions import NComponentsError, NotAListError, NotDataFrameError
from .model import PCA


Statistic = Literal["T2", "SPE", "both"]


@dataclass
class OptimizationIteration:
    """A single transparent decision made while curating Phase-I data."""

    iteration: int
    n_samples: int
    n_flagged: int
    flagged_fraction: float
    limits: dict[str, float]
    removed_positions: list[int]
    removed_index: list[object]


@dataclass
class OptimizationResult:
    """Result and audit trail of a Phase-I reference-data curation run."""

    in_control_data: pd.DataFrame
    removed_data: pd.DataFrame
    model: PCA
    history: list[OptimizationIteration]
    termination_reason: str
    chart_alpha: float
    max_outlier_fraction: float


class PCAOptimizer:
    """Iteratively curate a PCA Phase-I reference data set.

    The optimizer identifies observations outside Phase-I PCA control limits and
    removes only the most severe flagged observations per iteration. It is a
    reference-data curation aid, not an automatic diagnosis: inspect ``result_``
    before accepting the retained data as in control.

    ``alpha`` controls chart confidence. ``max_outlier_fraction`` is a separate
    operational policy stating how much residual contamination is acceptable in
    the final reference set.
    """

    def __init__(
        self,
        n_comps: int,
        alpha: float,
        numerical_features: list[str] | None = None,
        statistic: Statistic = "both",
        threshold: float | None = None,
        drop_percentage: float = 0.2,
        max_iterations: int = 50,
        *,
        max_outlier_fraction: float | None = None,
    ):
        statistic = statistic.upper()
        if statistic not in {"T2", "SPE", "BOTH"}:
            raise ValueError("statistic must be 'T2', 'SPE', or 'both'.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if numerical_features is not None and not isinstance(numerical_features, list):
            raise NotAListError(type(numerical_features).__name__)
        if not 0 < drop_percentage <= 1:
            raise ValueError("drop_percentage must be strictly between 0 and 1.")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative.")

        if threshold is not None:
            if threshold <= 0:
                raise ValueError("threshold must be positive.")
            warnings.warn(
                "threshold is deprecated; use max_outlier_fraction. It is no "
                "longer a control-limit multiplier.",
                DeprecationWarning,
                stacklevel=2,
            )
            if max_outlier_fraction is None:
                max_outlier_fraction = (1 - alpha) * threshold

        if max_outlier_fraction is None:
            max_outlier_fraction = 1 - alpha
        if not 0 <= max_outlier_fraction < 1:
            raise ValueError("max_outlier_fraction must be in [0, 1).")

        self.n_comps = n_comps
        self.alpha = alpha
        self.numerical_features = numerical_features or []
        self.statistic = statistic.lower()
        self.drop_percentage = drop_percentage
        self.max_iterations = max_iterations
        self.max_outlier_fraction = max_outlier_fraction
        # Bonferroni correction controls the family-wise rate when either T² or
        # SPE can flag an observation.
        self.chart_alpha = 1 - (1 - alpha) / 2 if self.statistic == "both" else alpha
        self.logger = logging.getLogger(__name__)

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise NotDataFrameError(type(X).__name__)
        if self.n_comps <= 0 or self.n_comps > min(X.shape):
            raise NComponentsError(min(X.shape))
        if len(X) < self.n_comps + 2:
            raise ValueError("At least n_comps + 2 reference observations are required.")
        if self.statistic in {"spe", "both"} and self.n_comps >= X.shape[1]:
            raise ValueError("SPE monitoring requires n_comps to be smaller than the number of features.")

    def _fit_pca(self, X: pd.DataFrame) -> PCA:
        pca = PCA(
            n_comps=self.n_comps,
            numerical_features=self.numerical_features,
            alpha=self.chart_alpha,
        )
        # Iterations need monitoring quantities only; SVI and plotting metrics
        # are deliberately deferred to the final model fit below.
        return pca.fit(X, compute_diagnostics=False)

    def _flagged_observations(self, pca: PCA) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        statistics = pca.phase1_statistics_
        limits = pca.control_limits_
        selected = ("T2", "SPE") if self.statistic == "both" else (self.statistic.upper(),)

        flagged = np.zeros(len(statistics["T2"]), dtype=bool)
        severity = np.zeros_like(statistics["T2"], dtype=float)
        selected_limits: dict[str, float] = {}
        for name in selected:
            limit_key = "T2_phase1" if name == "T2" else name
            limit = limits[limit_key]
            if not np.isfinite(limit):
                raise ValueError(f"{name} control limit is unavailable for this PCA model.")
            values = statistics[name]
            flagged |= values > limit
            ratio = np.divide(values, limit, out=np.full_like(values, np.inf, dtype=float), where=limit > 0)
            severity = np.maximum(severity, ratio)
            selected_limits[name] = float(limit)
        return flagged, severity, selected_limits

    def _select_removals(self, flagged: np.ndarray, severity: np.ndarray, capacity: int) -> np.ndarray:
        candidates = np.flatnonzero(flagged)
        n_remove = min(capacity, max(1, math.ceil(len(candidates) * self.drop_percentage)))
        if n_remove >= len(candidates):
            return candidates
        # Partial selection avoids sorting every flagged observation.
        selected = candidates[np.argpartition(severity[candidates], -n_remove)[-n_remove:]]
        return selected[np.argsort(severity[selected])[::-1]]

    def optimize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return curated data and store removed rows, history, and model in ``result_``."""
        self._validate_input(X)
        active = np.ones(len(X), dtype=bool)
        minimum_samples = self.n_comps + 2
        history: list[OptimizationIteration] = []
        termination_reason = "maximum iterations reached"

        for iteration in range(self.max_iterations + 1):
            positions = np.flatnonzero(active)
            current = X.iloc[positions]
            monitoring_model = self._fit_pca(current)
            flagged, severity, limits = self._flagged_observations(monitoring_model)
            flagged_fraction = float(flagged.mean())
            removed_relative = np.array([], dtype=int)

            if not flagged.any():
                termination_reason = "no observations exceed the selected Phase I limits"
            elif flagged_fraction <= self.max_outlier_fraction:
                termination_reason = "allowed out-of-control fraction reached"
            elif iteration >= self.max_iterations:
                termination_reason = "maximum iterations reached"
            else:
                capacity = len(current) - minimum_samples
                if capacity <= 0:
                    termination_reason = "minimum reference sample size reached"
                else:
                    removed_relative = self._select_removals(flagged, severity, capacity)
                    active[positions[removed_relative]] = False
                    self.logger.info(
                        "Iteration %d: removing %d of %d flagged observations (%.2f%% flagged).",
                        iteration + 1, len(removed_relative), flagged.sum(), 100 * flagged_fraction,
                    )

            removed_positions = positions[removed_relative]
            history.append(OptimizationIteration(
                iteration=iteration,
                n_samples=len(current),
                n_flagged=int(flagged.sum()),
                flagged_fraction=flagged_fraction,
                limits=limits,
                removed_positions=removed_positions.tolist(),
                removed_index=X.index[removed_positions].tolist(),
            ))
            if len(removed_relative) == 0:
                break

        in_control_data = X.iloc[np.flatnonzero(active)]
        removed_data = X.iloc[np.flatnonzero(~active)]
        # Fit the returned model once with full diagnostics on the final set.
        final_model = PCA(
            n_comps=self.n_comps,
            numerical_features=self.numerical_features,
            alpha=self.chart_alpha,
        ).fit(in_control_data)
        self.model_ = final_model
        self.result_ = OptimizationResult(
            in_control_data=in_control_data,
            removed_data=removed_data,
            model=final_model,
            history=history,
            termination_reason=termination_reason,
            chart_alpha=self.chart_alpha,
            max_outlier_fraction=self.max_outlier_fraction,
        )
        self.logger.info("Optimization finished: %s.", termination_reason)
        return in_control_data
