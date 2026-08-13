from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from sklearn.isotonic import IsotonicRegression
from tabulate import tabulate

from src.models.common.forecaster import Forecaster, QuantileConverter
from src.models.common.timecopilot_forecaster import TimeCopilotForecaster
from .metrics import (
    crps_fn,
    mae_fn,
    mse_fn,
    simple_mase_fn,
    smape_fn,
)

MetricName = Literal["mse", "mae", "smape", "crps", "mase"]


METRICS: dict[MetricName, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": mse_fn,
    "mae": mae_fn,
    "smape": smape_fn,
    "crps": crps_fn,
    "mase": simple_mase_fn,
}


@dataclass
class SLSQPOptimizationResult:
    weights: list[float]
    metric: MetricName
    metric_value: float
    success: bool
    message: str | None
    scipy_result: OptimizeResult | None = None


class SLSQPEnsemble(Forecaster):
    """Weighted ensemble with SLSQP constrained optimization.

    This ensemble computes non-negative weights that sum to 1 for the
    underlying models by minimizing a chosen error metric during
    cross-validation (CV) forecasts produced on the training data.

    Workflow (inside `forecast` when no explicit weights are supplied):
    1. Run model cross-validation (configurable number of windows).
    2. Collect each model's point forecasts and the actual `y`.
    3. Solve:  min_w  metric(y, X w)  subject to  w_i >= 0, sum w_i = 1.
    4. Apply the optimized weights to future (horizon) forecasts.

    Quantile & level forecasts:
    - If quantiles are requested, the same weights are applied per quantile.
    - Monotonicity across quantiles is enforced via isotonic regression.

    Parameters controlling weight optimization (all optional):
    - `weights`: manually pass weights (length = n_models) to skip
        optimization.
    - `metric`: one of {"mse", "mae", "smape", "crps", "mase"}. Default:
        "mse".
    - `opt_n_windows`: number of CV windows used for weighting (default 1).
    - `opt_step_size`: step size between CV windows (defaults to h if None).
    - `opt_h`: horizon to use for CV during weighting (defaults to `h`).

    Notes:
    - If optimization fails, the ensemble safely falls back to equal weights.
    - With a single model, the weight is trivially [1.0].
    """

    def __init__(
        self,
        models: list[Forecaster],
        metric: MetricName = "mse",
        batch_size: int = 128,
        n_windows: int = 1,
    ) -> None:
        self.tcf = TimeCopilotForecaster(models=models, fallback_model=None)
        self.opt_result_: SLSQPOptimizationResult | None = None
        self.batch_size = batch_size
        self.metric = metric
        self.n_windows = n_windows
        self.alias = self.format_alias()
        self.weights_df = pd.DataFrame(columns=self.model_aliases)

    def format_alias(self) -> str:
        num_models_str = f"{len(self.tcf.models)}-models"
        num_windows_str = f"{self.n_windows}-windows"
        metric_str = f"opt-{self.metric}"
        return f"SLSQPEnsemble_{num_models_str}_{metric_str}_{num_windows_str}"

    @property
    def model_aliases(self) -> list[str]:
        return [m.alias for m in self.tcf.models]

    def _optimize_weights(
        self,
        y: np.ndarray,
        model_matrix: np.ndarray,
        metric: MetricName,
        init: np.ndarray | None = None,
    ) -> SLSQPOptimizationResult:
        n_models = model_matrix.shape[1]
        if n_models == 1:
            return SLSQPOptimizationResult(
                weights=[1.0],
                metric=metric,
                metric_value=METRICS[metric](y, model_matrix[:, 0]),
                success=True,
                message="Single model - trivial weights",
            )
        if init is None:
            init = np.full(n_models, 1.0 / n_models)

        metric_fn = METRICS[metric]

        def objective(w: np.ndarray) -> float:
            yhat = model_matrix @ w
            return metric_fn(y, yhat)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n_models

        try:
            res = minimize(
                objective,
                init,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "disp": True},
            )
            if not res.success:
                warnings.warn(
                    f"SLSQP optimization failed: {res.message}. Defaulting to "
                    "equal weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                w = np.full(n_models, 1.0 / n_models)
                mv = metric_fn(y, model_matrix @ w)
                return SLSQPOptimizationResult(
                    weights=w.tolist(),
                    metric=metric,
                    metric_value=mv,
                    success=False,
                    message=str(res.message),
                    scipy_result=res,
                )
            w = res.x
            mv = metric_fn(y, model_matrix @ w)
            return SLSQPOptimizationResult(
                weights=w.tolist(),
                metric=metric,
                metric_value=mv,
                success=True,
                message=str(res.message),
                scipy_result=res,
            )
        except Exception as e:  # pragma: no cover (robustness)
            warnings.warn(
                f"SLSQP optimization failed: {res.message}. Defaulting to "
                "equal weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            w = np.full(n_models, 1.0 / n_models)
            mv = metric_fn(y, model_matrix @ w)
            return SLSQPOptimizationResult(
                weights=w.tolist(),
                metric=metric,
                metric_value=mv,
                success=False,
                message=str(e),
                scipy_result=None,
            )

    def _optimize_weights_quantile(
        self,
        y: np.ndarray,
        quantile_matrix: np.ndarray,
        metric: MetricName,
        init: np.ndarray | None = None,
    ) -> SLSQPOptimizationResult:
        """
        Optimize weights for quantile predictions (e.g., CRPS).

        Args:
            y: Ground truth values, shape (n_samples,)
            quantile_matrix: Quantile predictions, shape (n_samples, n_models,
                n_quantiles)
            metric: Metric name (should be "crps")
            init: Initial weights

        Returns:
            SLSQPOptimizationResult with optimized weights
        """
        _, n_models, _ = quantile_matrix.shape

        # Compute CRPS directly for single model
        if n_models == 1:
            single_model_quantiles = quantile_matrix[:, 0, :]
            metric_value = METRICS[metric](y, single_model_quantiles)
            return SLSQPOptimizationResult(
                weights=[1.0],
                metric=metric,
                metric_value=metric_value,
                success=True,
                message="Single model - trivial weights",
            )

        if init is None:
            init = np.full(n_models, 1.0 / n_models)

        metric_fn = METRICS[metric]

        def objective(w: np.ndarray) -> float:
            # Compute weighted ensemble quantile predictions.
            # shape: (n_samples, n_quantiles)
            yhat_quantiles = np.einsum("ijk,j->ik", quantile_matrix, w)
            return metric_fn(y, yhat_quantiles)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n_models

        res = minimize(
            objective,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "disp": False},
        )
        if not res.success:
            warnings.warn(
                f"SLSQP quantile optimization failed: {res.message}. "
                "Falling back to equal weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            w = np.full(n_models, 1.0 / n_models)
            yhat_quantiles = np.einsum("ijk,j->ik", quantile_matrix, w)
            mv = metric_fn(y, yhat_quantiles)
            return SLSQPOptimizationResult(
                weights=w.tolist(),
                metric=metric,
                metric_value=mv,
                success=False,
                message=str(res.message),
                scipy_result=res,
            )
        w = res.x
        yhat_quantiles = np.einsum("ijk,j->ik", quantile_matrix, w)
        mv = metric_fn(y, yhat_quantiles)
        return SLSQPOptimizationResult(
            weights=w.tolist(),
            metric=metric,
            metric_value=mv,
            success=True,
            message=str(res.message),
            scipy_result=res,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        weights: list[float] | None = None,
        opt_n_windows: int = 1,
        opt_step_size: int | None = None,
        opt_h: int | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)
        
        kwargs = {
            "attr": "cross_validation",
            "merge_on": ["unique_id", "ds", "cutoff"],
            "df": df,
            "h": h if opt_h is None else opt_h,
            "freq": freq,
            "level": None,
            "quantiles": qc.quantiles,
            "n_windows": opt_n_windows,
            "step_size": opt_step_size,
        }
        cv_df = self.tcf._call_models(**kwargs)
        
        print(f"DEBUG: CV dataframe shape: {cv_df.shape}")
        
        model_cols = [m.alias for m in self.tcf.models]

        # Use Toto's median forecasts
        model_cols = [
            col.replace("Toto", "Toto-q-50") if col == "Toto" else col
            for col in model_cols
        ]

        # Ensure all base models have cross-validation forecasts
        missing = [c for c in model_cols if c not in cv_df]
        if missing:
            raise RuntimeError(
                "Missing model columns in CV dataframe: " f"{missing}."
            )

        # Get the ground truth values
        y = cv_df["y"].to_numpy(dtype=float)

        # Handle special metrics that need additional processing
        if self.metric == "crps" and qc.quantiles is not None:
            # Get quantile columns for each model
            quantile_data = []
            model_cols = [m.alias for m in self.tcf.models]

            for model in model_cols:
                model_quantiles = []
                for q in sorted(qc.quantiles):
                    pct = int(q * 100)
                    q_col = f"{model}-q-{pct}"
                    if q_col not in cv_df:
                        raise RuntimeError(
                            f"Missing quantile column: {q_col}.",
                        )
                    model_quantiles.append(cv_df[q_col].to_numpy(dtype=float))

                # Stack quantiles for this model: shape (n_samples,
                # n_quantiles)
                model_quantiles = np.stack(model_quantiles, axis=1)
                quantile_data.append(model_quantiles)

            # Stack all models: shape (n_samples, n_models, n_quantiles)
            X_quantiles = np.stack(quantile_data, axis=1)

            # guard against NaNs
            mask = ~np.isnan(y) & ~np.any(
                np.isnan(X_quantiles.reshape(X_quantiles.shape[0], -1)),
                axis=1,
            )
            y_clean = y[mask]

            # shape: (n_clean_samples, n_models, n_quantiles)
            X_clean = X_quantiles[mask]

            if y_clean.size == 0:
                raise RuntimeError(
                    "No valid rows available for weight optimization."
                )

            # Optimize weights for quantile predictions
            opt_res = self._optimize_weights_quantile(
                y_clean,
                X_clean,
                self.metric,
            )

        elif self.metric == "mase":
            # Enhanced MASE calculation with training data context
            X = cv_df[model_cols].to_numpy(dtype=float)

            # Try to extract training data for better MASE calculation
            # Group by unique_id to get per-series training data
            enhanced_mase_data = []
            for uid in cv_df["unique_id"].unique():
                uid_data = cv_df[cv_df["unique_id"] == uid].sort_values(
                    "ds",
                )
                uid_y = uid_data["y"].values
                uid_X = uid_data[model_cols].values

                # Use the validation data as both training context and
                # target.
                # This is a limitation of the CV approach - we don't have
                # separate training data
                enhanced_mase_data.append((uid_y, uid_X))

            # For now, fall back to simple MASE since we don't have clean
            # training/validation split
            mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
            y_clean = y[mask]
            X_clean = X[mask]
            if y_clean.size == 0:
                raise RuntimeError(
                    "No valid rows available for weight optimization."
                )

            print(
                f"DEBUG: MASE optimization - y_clean.shape: "
                f"{y_clean.shape}, X_clean.shape: {X_clean.shape}"
            )
            opt_res = self._optimize_weights(y_clean, X_clean, self.metric)

        else:
            # Original logic for point predictions
            X = cv_df[model_cols].to_numpy(dtype=float)
            # guard against NaNs
            mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)

            y_clean = y[mask]  # Ground truth
            X_clean = X[mask]  # Predictions

            if y_clean.size == 0:
                raise RuntimeError(
                    "No valid rows available for weight optimization."
                )
            opt_res = self._optimize_weights(y_clean, X_clean, self.metric)

        # Every branch above (crps / mase / point) produces opt_res, but this
        # assignment used to live inside the `else`, so metric="mase" and
        # metric="crps" left `weights` at its None default and crashed on the
        # comprehension below with "'NoneType' object is not iterable".
        weights = opt_res.weights
        self.opt_result_ = opt_res

        self.weights_ = [float(w) for w in weights]

        # Save base models' weights
        new_weights = {
            m: w
            for m, w in zip(
                self.model_aliases,
                weights,
                strict=False,
            )
        }

        new_weights_df = pd.DataFrame(
            [new_weights],
            columns=self.model_aliases,
        )

        self.weights_df = pd.concat(
            [self.weights_df, new_weights_df],
            ignore_index=True,
        )

        model_cols = [m.alias for m in self.tcf.models]
        if verbose:
            weights_str = ", ".join(
                f"{m}:{w:.4f}"
                for m, w in zip(
                    model_cols,
                    self.weights_,
                    strict=False,
                )
            )
            if self.opt_result_ is not None:
                logging.info(
                    f"[{self.alias}] optimized weights ({self.metric}, "
                    f"value={self.opt_result_.metric_value:.6f}): "
                    f"{weights_str}"
                )
            else:
                logging.info(f"[{self.alias}] provided weights: {weights_str}")

        # Forecast future horizon for each base model
        _fcst_df = self.tcf._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=None,  # always use quantiles path then convert if needed
            quantiles=qc.quantiles,
        )

        out = _fcst_df[["unique_id", "ds"]].copy()
        # reuse model_cols from above
        W = np.array(self.weights_)
        # point forecasts
        out[self.alias] = (_fcst_df[model_cols].to_numpy(dtype=float) @ W).astype(float)

        if qc.quantiles is not None:
            qs = sorted(qc.quantiles)
            q_cols = []
            for q in qs:
                pct = int(q * 100)
                base_cols = [f"{col}-q-{pct}" for col in model_cols]
                missing = [c for c in base_cols if c not in _fcst_df]

                if missing:
                    raise RuntimeError(
                        f"Quantile columns missing for pct {pct}: {missing}."
                    )
                vals = _fcst_df[base_cols].to_numpy(dtype=float) @ W
                q_col = f"{self.alias}-q-{pct}"
                out[q_col] = vals
                q_cols.append(q_col)
            # enforce monotonicity across quantiles per row
            ir = IsotonicRegression(increasing=True)

            def _mono(row: pd.Series) -> np.ndarray:
                return ir.fit_transform(qs, row.values)

            mono_vals = out[q_cols].apply(_mono, axis=1, result_type="expand")
            out[q_cols] = mono_vals
            if 0.5 in qc.quantiles:
                out[self.alias] = out[f"{self.alias}-q-50"].values
            out = qc.maybe_convert_quantiles_to_level(out, models=[self.alias])

        return out

    def print_weights(self) -> None:
        """Prints the ensemble weights across all cross-validation windows."""
        print(
            tabulate(
                self.weights_df,
                headers="keys",
                tablefmt="grid",
                showindex=False,
                floatfmt=".4f",
            )
        )
