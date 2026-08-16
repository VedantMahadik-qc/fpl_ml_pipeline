"""
FPL ML Pipeline — Phase 3: Model Training
=========================================

Trains four position-specific regressors (GKP / DEF / MID / FWD) that forecast a
player's `total_points` for an upcoming gameweek, using only information that
existed before that gameweek's deadline.

Split
-----
  TRAIN  2023-24 + 2024-25   (last N gameweeks of 2024-25 held out for early stopping)
  TEST   2025-26
Strictly temporal. No random shuffling anywhere — a random split would let the
model see GW30 while predicting GW12 of the same season.

Usage
-----
    python fpl_train.py
    python fpl_train.py --train-seasons 2023-24 2024-25 --test-season 2025-26
    python fpl_train.py --model lightgbm --model-dir models
    python fpl_train.py --min-games-prior 3      # drop pure cold-start rows from training


On the target: why there is no .shift(-1)
-----------------------------------------
Phase 2 built every feature on row N from gameweeks N-1 and older. So the pair
(features_N, total_points_N) already *is* a next-gameweek forecasting example:
"given everything known before this fixture, how many points does he return?"
Shifting the target by -1 on top of that would predict GW N+1 from data ending at
N-1, silently throwing away a gameweek and misaligning double gameweeks. The
target is the row's own `total_points`.


On the feature allowlist
------------------------
Only three families of column are eligible:
  * anything matching `*_roll_<n>`  (Phase 2 rolling features, strictly causal)
  * pre-deadline knowns: gw_cost, was_home, games_prior, is_cold_start
  * position indicators
Everything else in the Phase 2 file is a post-match measurement — `minutes`,
`bps`, raw `expected_goals`, `bonus` and friends are only knowable after the
fixture is played. `assert_no_post_match_features()` fails the run if any of them
reaches the matrix, so the allowlist cannot silently rot as columns are added.


On the metrics
--------------
Roughly half of all player-gameweek rows are non-appearances worth 0 points. A
model that predicts a flat ~1.2 everywhere therefore scores a respectable-looking
MAE. So every model is reported against two naive baselines (position mean, and
"assume he repeats his 3-game average") and broken out for rows where the player
actually played. A model that cannot beat `roll_3` persistence is not a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:  # sklearn >= 1.4
    from sklearn.metrics import root_mean_squared_error

    def _rmse(y_true, y_pred) -> float:
        return float(root_mean_squared_error(y_true, y_pred))
except ImportError:  # pragma: no cover

    def _rmse(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


LOGGER = logging.getLogger("fpl.train")

POSITIONS: List[str] = ["GKP", "DEF", "MID", "FWD"]

ROLLING_PATTERN = re.compile(r"_roll_\d+$")

# Known before the deadline, so legal as features.
PRE_DEADLINE_FEATURES: List[str] = ["gw_cost", "was_home", "games_prior", "is_cold_start"]

# Only knowable after the fixture is played. Never features.
POST_MATCH_COLUMNS: set = {
    "total_points", "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence", "creativity",
    "threat", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "defensive_contribution",
    "clean_sheet_probability", "selected", "transfers_in", "transfers_out",
    "transfers_balance", "team_h_score", "team_a_score", "xp",
}

TARGET = "total_points"


@dataclass
class TrainConfig:
    input_path: Path = Path("fpl_ml_features.csv")
    model_dir: Path = Path("models")
    train_seasons: Sequence[str] = ("2023-24", "2024-25")
    test_season: str = "2025-26"
    valid_gameweeks: int = 10         # tail of the last training season, for early stopping
    model_type: str = "xgboost"       # xgboost | lightgbm | hist
    min_games_prior: int = 0          # drop rows with less real history than this
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 5
    early_stopping_rounds: int = 100
    permutation_repeats: int = 5
    permutation_sample: int = 5000
    random_state: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Data loading and feature selection
# --------------------------------------------------------------------------- #

def load_features(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fpl_features.py (Phase 2) first.")
    frame = pd.read_csv(path, low_memory=False)

    required = {TARGET, "season", "position", "gameweek"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame[frame[TARGET].notna()].copy()
    LOGGER.info("Loaded %s: %d rows x %d columns", path, len(frame), frame.shape[1])
    return frame


def add_position_indicators(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot the position. Constant inside a position model; kept for pooled use."""
    frame = frame.copy()
    names: List[str] = []
    for position in POSITIONS:
        column = f"pos_{position}"
        frame[column] = (frame["position"] == position).astype("int8")
        names.append(column)
    return frame, names


def select_feature_columns(frame: pd.DataFrame, position_columns: Sequence[str]) -> List[str]:
    """Allowlist: rolling features + pre-deadline knowns + position indicators."""
    rolling = sorted(c for c in frame.columns if ROLLING_PATTERN.search(c))
    if not rolling:
        raise RuntimeError("No `*_roll_<n>` columns found — is this the Phase 2 output?")

    pre_deadline = [c for c in PRE_DEADLINE_FEATURES if c in frame.columns]
    for absent in set(PRE_DEADLINE_FEATURES) - set(pre_deadline):
        LOGGER.warning("Pre-deadline feature `%s` not in data — skipping.", absent)

    features = rolling + pre_deadline + list(position_columns)
    LOGGER.info("Feature matrix: %d columns (%d rolling, %d pre-deadline, %d position).",
                len(features), len(rolling), len(pre_deadline), len(position_columns))
    return features


def assert_no_post_match_features(features: Sequence[str]) -> None:
    """Hard stop if a post-match measurement reached the feature matrix."""
    illegal = sorted(set(features) & POST_MATCH_COLUMNS)
    assert not illegal, (
        f"POST-MATCH LEAKAGE: {illegal} are only knowable after the fixture is played "
        f"and must not be features."
    )
    LOGGER.info("Leakage guard passed: no post-match columns in the feature matrix.")


def prepare_matrix(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Numeric, model-ready copy of the feature columns."""
    matrix = frame[list(features)].copy()
    for column in matrix.columns:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype("int8")
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def drop_constant_features(
    train_matrix: pd.DataFrame, features: Sequence[str]
) -> List[str]:
    """
    Remove zero-variance columns.

    Inside a position model the position indicators are constant by construction
    (every row in the GKP model is a GKP), so they carry no signal and only
    clutter the importance chart.
    """
    keep = [c for c in features if train_matrix[c].nunique(dropna=False) > 1]
    dropped = sorted(set(features) - set(keep))
    if dropped:
        LOGGER.debug("Dropped %d constant feature(s): %s", len(dropped), ", ".join(dropped))
    return keep


# --------------------------------------------------------------------------- #
# Temporal splitting
# --------------------------------------------------------------------------- #

def temporal_split(
    frame: pd.DataFrame, train_seasons: Sequence[str], test_season: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season. Never random — the ordering is the whole point."""
    seasons = set(frame["season"].unique())
    for season in list(train_seasons) + [test_season]:
        if season not in seasons:
            raise RuntimeError(
                f"Season '{season}' not present in the data. Available: {sorted(seasons)}"
            )

    train = frame[frame["season"].isin(train_seasons)].copy()
    test = frame[frame["season"] == test_season].copy()

    overlap = set(train.index) & set(test.index)
    assert not overlap, "Train and test overlap — split is broken."

    LOGGER.info("Temporal split — train %s: %d rows | test %s: %d rows",
                list(train_seasons), len(train), test_season, len(test))
    return train, test


def carve_validation(
    train: pd.DataFrame, valid_gameweeks: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out the tail of the last training season for early stopping.

    Taking the *tail* rather than a random sample keeps the validation set in the
    future relative to what the model fits on, mirroring the real test split.
    """
    if valid_gameweeks <= 0:
        return train, train.iloc[0:0]

    last_season = sorted(train["season"].unique())[-1]
    in_last = train["season"] == last_season
    max_gw = train.loc[in_last, "gameweek"].max()
    if pd.isna(max_gw):
        return train, train.iloc[0:0]

    cutoff = max_gw - valid_gameweeks
    valid_mask = in_last & (train["gameweek"] > cutoff)

    fit = train[~valid_mask].copy()
    valid = train[valid_mask].copy()
    LOGGER.info("Early-stopping split — fit: %d rows | valid: %s GW>%s, %d rows",
                len(fit), last_season, int(cutoff), len(valid))
    if 0 < len(valid) < 500:
        LOGGER.warning(
            "  validation slice is only %d rows — early stopping will be noisy and may "
            "halt the fit prematurely. Consider raising --valid-gameweeks.", len(valid)
        )
    return fit, valid


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #

def build_estimator(config: TrainConfig):
    """XGBoost by default, with LightGBM and a stdlib-only fallback."""
    if config.model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "xgboost not installed. `pip install xgboost` or use --model hist."
            ) from exc
        return XGBRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            # RMSE, not MAE, for early stopping: it matches the squared-error
            # objective, so the eval curve is smooth. MAE against a squared-error
            # objective plateaus early on small validation slices and stops the
            # fit after a couple of rounds. MAE is still what we *report*.
            eval_metric="rmse",
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            n_jobs=-1,
        )

    if config.model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "lightgbm not installed. `pip install lightgbm` or use --model xgboost."
            ) from exc
        return LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        early_stopping=True,
        random_state=config.random_state,
    )


def fit_estimator(estimator, X_fit, y_fit, X_valid, y_valid, config: TrainConfig):
    """Fit with early stopping where the backend supports it."""
    if config.model_type == "xgboost" and len(X_valid):
        estimator.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], verbose=False)
        best = getattr(estimator, "best_iteration", None)
        if best is not None:
            LOGGER.info("  early stopping at iteration %d/%d", best, config.n_estimators)
    elif config.model_type == "lightgbm" and len(X_valid):
        import lightgbm as lgb
        estimator.fit(
            X_fit, y_fit,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        )
    else:
        estimator.fit(X_fit, y_fit)
    return estimator


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def naive_baselines(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Three reference predictors the model must beat to be worth deploying.

    position_mean    average points for this position in training.
    position_median  the MAE-optimal constant. FPL points are zero-inflated and
                     right-skewed, so the median is a *harder* MAE floor than the
                     mean — omitting it flatters the model.
    roll_3           assume he repeats his 3-game average. Free to compute and a
                     surprisingly strong FPL heuristic.
    """
    baselines: Dict[str, np.ndarray] = {
        "baseline_position_mean": np.full(len(test), float(train[TARGET].mean())),
        "baseline_position_median": np.full(len(test), float(train[TARGET].median())),
    }
    roll3 = f"{TARGET}_roll_3"
    if roll3 in test.columns:
        values = pd.to_numeric(test[roll3], errors="coerce")
        baselines["baseline_roll_3"] = values.fillna(train[TARGET].mean()).to_numpy()
    return baselines


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
    }


def evaluate_predictions(
    test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, baselines: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Score the model and the baselines, overall and on appearances only.

    The `played` split matters: non-appearances are ~half the rows and are
    trivially predictable (0 points), so overall MAE flatters every model. The
    appearances-only number is the one that reflects real forecasting skill.
    """
    rows: List[Dict[str, Any]] = []
    segments: List[Tuple[str, np.ndarray]] = [("all_rows", np.ones(len(test), dtype=bool))]
    if "minutes" in test.columns:
        played = pd.to_numeric(test["minutes"], errors="coerce").fillna(0).to_numpy() > 0
        segments.append(("played_only", played))

    predictions = {"model": y_pred, **baselines}
    for segment_name, mask in segments:
        if mask.sum() == 0:
            continue
        for label, values in predictions.items():
            metrics = score(y_true[mask], values[mask])
            rows.append({"segment": segment_name, "predictor": label,
                         "n": int(mask.sum()), **metrics})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #

def compute_importance(
    estimator, X_test: pd.DataFrame, y_test: np.ndarray, features: Sequence[str],
    config: TrainConfig,
) -> pd.DataFrame:
    """
    Built-in (gain) importance plus permutation importance on the test season.

    Gain is cheap but biased toward high-cardinality features. Permutation
    importance measures the actual MAE cost of scrambling a column on unseen
    data, which is what "does this feature matter" should mean.
    """
    frame = pd.DataFrame({"feature": list(features)})

    native = getattr(estimator, "feature_importances_", None)
    frame["gain_importance"] = native if native is not None else np.nan

    if len(X_test) > 0 and config.permutation_repeats > 0:
        sample = min(len(X_test), config.permutation_sample)
        rng = np.random.default_rng(config.random_state)
        idx = rng.choice(len(X_test), size=sample, replace=False) if sample < len(X_test) \
            else np.arange(len(X_test))
        try:
            result = permutation_importance(
                estimator, X_test.iloc[idx], y_test[idx],
                n_repeats=config.permutation_repeats,
                random_state=config.random_state,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            frame["permutation_mae_increase"] = result.importances_mean
            frame["permutation_std"] = result.importances_std
        except Exception as exc:  # importance is diagnostic, never fatal
            LOGGER.warning("Permutation importance failed: %s", exc)
            frame["permutation_mae_increase"] = np.nan
            frame["permutation_std"] = np.nan
    else:
        frame["permutation_mae_increase"] = np.nan
        frame["permutation_std"] = np.nan

    sort_key = "permutation_mae_increase" if frame["permutation_mae_increase"].notna().any() \
        else "gain_importance"
    return frame.sort_values(sort_key, ascending=False).reset_index(drop=True)


def plot_importance(frame: pd.DataFrame, position: str, path: Path, top_n: int = 15) -> None:
    column = "permutation_mae_increase" if frame["permutation_mae_increase"].notna().any() \
        else "gain_importance"
    top = frame.nlargest(top_n, column).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.36 * len(top))))
    ax.barh(top["feature"], top[column], color="#38003c")
    ax.set_xlabel("Increase in MAE when shuffled" if column.startswith("permutation")
                  else "Gain importance")
    ax.set_title(f"{position} — top {len(top)} features ({column})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=140)
    except OSError as exc:
        LOGGER.warning("Could not write %s: %s", path, exc)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-position training
# --------------------------------------------------------------------------- #

def train_position_model(
    position: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    config: TrainConfig,
) -> Optional[Dict[str, Any]]:
    """Train, evaluate and package one position's model. None if data is too thin."""
    LOGGER.info("--- %s ---", position)

    train_pos = train[train["position"] == position]
    test_pos = test[test["position"] == position]

    if config.min_games_prior > 0 and "games_prior" in train_pos.columns:
        before = len(train_pos)
        train_pos = train_pos[train_pos["games_prior"] >= config.min_games_prior]
        LOGGER.info("  dropped %d training rows with games_prior < %d",
                    before - len(train_pos), config.min_games_prior)

    if len(train_pos) < 100 or len(test_pos) == 0:
        LOGGER.warning("  skipping %s: %d train / %d test rows is too few.",
                       position, len(train_pos), len(test_pos))
        return None

    fit_df, valid_df = carve_validation(train_pos, config.valid_gameweeks)

    X_all = prepare_matrix(train_pos, features)
    active = drop_constant_features(X_all, features)
    if len(active) < len(features):
        LOGGER.info("  using %d/%d features (%d constant within this position)",
                    len(active), len(features), len(features) - len(active))

    X_fit = prepare_matrix(fit_df, active)
    y_fit = fit_df[TARGET].to_numpy(dtype=float)
    X_valid = prepare_matrix(valid_df, active)
    y_valid = valid_df[TARGET].to_numpy(dtype=float)
    X_test = prepare_matrix(test_pos, active)
    y_test = test_pos[TARGET].to_numpy(dtype=float)

    estimator = build_estimator(config)
    estimator = fit_estimator(estimator, X_fit, y_fit, X_valid, y_valid, config)

    y_pred = estimator.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # FPL scores are rarely negative; never predict below 0

    metrics = evaluate_predictions(test_pos, y_test, y_pred, naive_baselines(train_pos, test_pos))
    metrics.insert(0, "position", position)

    importance = compute_importance(estimator, X_test, y_test, active, config)
    importance.insert(0, "position", position)

    headline = metrics[(metrics.segment == "all_rows") & (metrics.predictor == "model")].iloc[0]
    LOGGER.info("  test MAE %.4f | RMSE %.4f  (n=%d)",
                headline["mae"], headline["rmse"], int(headline["n"]))

    return {
        "position": position,
        "model": estimator,
        "features": active,
        "metrics": metrics,
        "importance": importance,
        "n_train": len(fit_df),
        "n_valid": len(valid_df),
        "n_test": len(test_pos),
    }


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

def save_artifacts(
    results: Dict[str, Dict[str, Any]], features: Sequence[str], config: TrainConfig
) -> Dict[str, Path]:
    """Persist models, per-position importance, and the metadata the optimizer needs."""
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    all_metrics, all_importance = [], []

    for position, result in results.items():
        bundle = {
            "position": position,
            "model": result["model"],
            "features": result["features"],
            "target": TARGET,
            "model_type": config.model_type,
            "train_seasons": list(config.train_seasons),
            "test_season": config.test_season,
            "metrics": result["metrics"].to_dict(orient="records"),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = model_dir / f"fpl_model_{position}.joblib"
        joblib.dump(bundle, path)
        written[position] = path

        result["importance"].to_csv(model_dir / f"feature_importance_{position}.csv", index=False)
        plot_importance(result["importance"], position,
                        model_dir / f"feature_importance_{position}.png")

        all_metrics.append(result["metrics"])
        all_importance.append(result["importance"])

    metadata = {
        "target": TARGET,
        "positions": list(results),
        "feature_columns": list(features),
        "per_position_features": {p: r["features"] for p, r in results.items()},
        "pre_deadline_features": PRE_DEADLINE_FEATURES,
        "post_match_blocklist": sorted(POST_MATCH_COLUMNS),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(config).items() if k != "metadata"},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(metadata, model_dir / "feature_metadata.joblib")
    (model_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2),
                                                     encoding="utf-8")

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            model_dir / "evaluation_summary.csv", index=False)
    if all_importance:
        pd.concat(all_importance, ignore_index=True).to_csv(
            model_dir / "feature_importance_all.csv", index=False)

    LOGGER.info("Saved %d model(s) + metadata to %s/", len(written), model_dir)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_training(config: TrainConfig) -> Dict[str, Any]:
    frame = load_features(config.input_path)
    frame, position_columns = add_position_indicators(frame)

    features = select_feature_columns(frame, position_columns)
    assert_no_post_match_features(features)

    train, test = temporal_split(frame, config.train_seasons, config.test_season)

    results: Dict[str, Dict[str, Any]] = {}
    for position in POSITIONS:
        result = train_position_model(position, train, test, features, config)
        if result is not None:
            results[position] = result

    if not results:
        raise RuntimeError("No position models could be trained — check the split and input data.")

    written = save_artifacts(results, features, config)
    summary = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
    return {"results": results, "summary": summary, "paths": written}


def print_report(summary: pd.DataFrame) -> None:
    """Model vs baselines, per position — the table that decides whether to ship."""
    print("\n" + "=" * 78)
    print("TEST-SEASON PERFORMANCE (model vs naive baselines)")
    print("=" * 78)

    for segment in summary["segment"].unique():
        block = summary[summary.segment == segment]
        pivot = block.pivot_table(index="position", columns="predictor",
                                  values=["mae", "rmse"])
        print(f"\n[{segment}]")
        print(pivot.round(4).to_string())

    # Report the verdict for EVERY segment. A model can beat the baselines
    # overall purely by predicting non-appearances well, while being worse than a
    # constant on the rows that actually decide a gameweek.
    print("\n" + "-" * 78)
    print("Model vs best baseline (MAE, lower is better):")
    regressions = 0
    for segment in summary["segment"].unique():
        print(f"\n  [{segment}]")
        for position in summary["position"].unique():
            block = summary[(summary.position == position) & (summary.segment == segment)]
            if block.empty or "model" not in set(block.predictor):
                continue
            model_mae = block[block.predictor == "model"]["mae"].iloc[0]
            baselines = block[block.predictor != "model"]
            if baselines.empty:
                continue
            best = baselines.loc[baselines["mae"].idxmin()]
            delta = best["mae"] - model_mae
            if delta > 0:
                verdict = "beats"
            else:
                verdict = "LOSES TO"
                regressions += 1
            print(f"    {position}: model {model_mae:.4f} {verdict} {best['predictor']} "
                  f"{best['mae']:.4f}  (delta {delta:+.4f})")

    if regressions:
        print(f"\n  WARNING: {regressions} position/segment combination(s) lose to a naive "
              f"baseline.\n  A model that cannot beat a constant on appearances is not ready "
              f"to drive transfers.")
    print("-" * 78)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Phase 3 position-specific model training.")
    parser.add_argument("-i", "--input", default="fpl_ml_features.csv")
    parser.add_argument("-d", "--model-dir", default="models")
    parser.add_argument("--train-seasons", nargs="+", default=["2023-24", "2024-25"])
    parser.add_argument("--test-season", default="2025-26")
    parser.add_argument("--model", dest="model_type", default="xgboost",
                        choices=["xgboost", "lightgbm", "hist"])
    parser.add_argument("--valid-gameweeks", type=int, default=6,
                        help="Tail gameweeks of the last training season used for early stopping")
    parser.add_argument("--min-games-prior", type=int, default=0,
                        help="Drop training rows with less real history than this")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainConfig(
        input_path=Path(args.input),
        model_dir=Path(args.model_dir),
        train_seasons=tuple(args.train_seasons),
        test_season=args.test_season,
        valid_gameweeks=args.valid_gameweeks,
        model_type=args.model_type,
        min_games_prior=args.min_games_prior,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        permutation_repeats=args.permutation_repeats,
        random_state=args.random_state,
    )

    try:
        outcome = run_training(config)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except AssertionError as exc:
        LOGGER.error("Leakage guard FAILED — refusing to train. %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    print_report(outcome["summary"])

    print("\nTop 8 features by position:")
    for position, result in outcome["results"].items():
        top = result["importance"].head(8)["feature"].tolist()
        print(f"  {position}: {', '.join(top)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
