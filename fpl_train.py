"""
FPL ML Pipeline — Phase 3 (V4): Two-Stage Position Models
=========================================================

Forecasts a player's `total_points` for an upcoming gameweek as the product of
two models per position:

    E[points]  =  P(plays)  x  E[points | plays]
                  ^^^^^^^^     ^^^^^^^^^^^^^^^^
                  XGBClassifier   XGBRegressor
                  binary:logistic  reg:squarederror
                  all rows         rows with minutes > 0

Why the product is exact
------------------------
    E[pts] = P(play)*E[pts|play] + (1 - P(play))*E[pts|not play]

A non-appearance scores exactly zero in FPL — no appearance point, no bonus, no
cards. The second term is identically 0, so this is the law of total
expectation, not an approximation.

Why V3's single regressor struggled
-----------------------------------
Roughly half of all player-gameweeks are non-appearances worth 0 points. A
single squared-error regressor spends most of its capacity learning who does not
play, which is why `minutes_roll_10` dominated every importance chart. Splitting
the problem lets the classifier own rotation risk and frees the regressor to
model returns on a target that is merely right-skewed rather than zero-inflated.

Calibration, not AUC
--------------------
Because the two stages MULTIPLY, the classifier's probabilities must be
calibrated, not merely well-ranked. ROC-AUC is invariant to any monotone
transform of the scores: a model can rank perfectly and still be systematically
overconfident, and that bias flows straight into every point estimate. Brier
score and log-loss are reported alongside AUC and are the ones to watch.

For the same reason `scale_pos_weight` is deliberately NOT set. It improves
apparent class balance and destroys calibration.

Split
-----
  TRAIN  2023-24 + 2024-25   (last 10 gameweeks of 2024-25 held out for early stopping)
  TEST   2025-26
Strictly temporal. No random shuffling anywhere.

Artifacts per position
----------------------
  models/fpl_model_{POS}_prob.joblib   classification stage
  models/fpl_model_{POS}_pts.joblib    regression stage
  models/fpl_model_{POS}.joblib        TwoStageModel wrapper, .predict() returns
                                       the product — keeps Phase 4 working unchanged

Usage
-----
    python fpl_train.py
    python fpl_train.py --model lightgbm --min-games-prior 3
    python fpl_train.py --train-seasons 2023-24 2024-25 --test-season 2025-26
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field, replace
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
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

# Defined in its own module, NOT here. A class pickled from a script is recorded
# as `__main__.TwoStageModel` and then fails to load from any other entry point.
from fpl_models import SeedAveragedClassifier, SeedAveragedRegressor, TwoStageModel

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

# V2 — fixture difficulty. Base `fdr_*` columns describe the UPCOMING opponent
# and are already causal at team level, so they are pre-deadline knowns. Picked
# up by prefix; the `*_roll_<n>` variants are caught by the rolling allowlist.
FDR_PREFIX: str = "fdr_"

# Bookmaker odds for the upcoming fixture (fpl_odds.py). Pre-deadline knowns,
# but only for the NEXT round: see `with_odds_masked`.
ODDS_PREFIX: str = "odds_"

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

TARGET = "total_points"          # regression stage
MINUTES = "minutes"              # classification stage derives from this


@dataclass
class TrainConfig:
    input_path: Path = Path("fpl_ml_features.csv")
    model_dir: Path = Path("models")
    train_seasons: Sequence[str] = ("2023-24", "2024-25")
    test_season: str = "2025-26"
    valid_gameweeks: int = 10         # tail of the last training season
    model_type: str = "xgboost"       # xgboost | lightgbm | hist
    min_games_prior: int = 0
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 5
    early_stopping_rounds: int = 100
    permutation_repeats: int = 5
    permutation_sample: int = 5000
    random_state: int = 42
    odds_dropout: bool = True         # also train on an odds-masked copy of each row
    ensemble_size: int = 1            # seeds per stage, predictions averaged
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Data loading and feature selection
# --------------------------------------------------------------------------- #

def load_features(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fpl_features.py (Phase 2) first.")
    frame = pd.read_csv(path, low_memory=False)

    required = {TARGET, MINUTES, "season", "position", "gameweek"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame[frame[TARGET].notna()].copy()
    frame["played"] = (pd.to_numeric(frame[MINUTES], errors="coerce").fillna(0) > 0).astype(int)
    rate = frame["played"].mean()
    LOGGER.info("Loaded %s: %d rows x %d columns | appearance rate %.1f%%",
                path, len(frame), frame.shape[1], 100 * rate)
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
    """Allowlist: rolling features + pre-deadline knowns + FDR + position indicators."""
    rolling = sorted(c for c in frame.columns if ROLLING_PATTERN.search(c))
    if not rolling:
        raise RuntimeError("No `*_roll_<n>` columns found — is this the Phase 2 output?")

    pre_deadline = [c for c in PRE_DEADLINE_FEATURES if c in frame.columns]
    for absent in set(PRE_DEADLINE_FEATURES) - set(pre_deadline):
        LOGGER.warning("Pre-deadline feature `%s` not in data — skipping.", absent)

    fdr = sorted(
        c for c in frame.columns
        if c.startswith(FDR_PREFIX)
        and not ROLLING_PATTERN.search(c)
        and not c.endswith("_observed")
        and frame[c].notna().any()
    )
    if fdr:
        LOGGER.info("Fixture-difficulty features found: %s", ", ".join(fdr))
    else:
        LOGGER.warning("No `fdr_*` columns — run Phase 2 with --strength-source derived.")

    odds = sorted(c for c in frame.columns
                  if c.startswith(ODDS_PREFIX) and frame[c].notna().any())
    if odds:
        LOGGER.info("Odds features found: %s", ", ".join(odds))

    features = rolling + pre_deadline + fdr + odds + list(position_columns)
    LOGGER.info("Feature matrix: %d columns (%d rolling, %d pre-deadline, %d FDR, %d odds, "
                "%d position).", len(features), len(rolling), len(pre_deadline), len(fdr),
                len(odds), len(position_columns))
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


def drop_constant_features(train_matrix: pd.DataFrame, features: Sequence[str]) -> List[str]:
    """Remove zero-variance columns (position indicators are constant per model)."""
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
                f"Season '{season}' not present in the data. Available: {sorted(seasons)}")

    train = frame[frame["season"].isin(train_seasons)].copy()
    test = frame[frame["season"] == test_season].copy()
    assert not (set(train.index) & set(test.index)), "Train and test overlap."

    LOGGER.info("Temporal split — train %s: %d rows | test %s: %d rows",
                list(train_seasons), len(train), test_season, len(test))
    return train, test


def carve_validation(train: pd.DataFrame, valid_gameweeks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the tail of the last training season for early stopping."""
    if valid_gameweeks <= 0:
        return train, train.iloc[0:0]

    last_season = sorted(train["season"].unique())[-1]
    in_last = train["season"] == last_season
    max_gw = train.loc[in_last, "gameweek"].max()
    if pd.isna(max_gw):
        return train, train.iloc[0:0]

    cutoff = max_gw - valid_gameweeks
    valid_mask = in_last & (train["gameweek"] > cutoff)
    fit, valid = train[~valid_mask].copy(), train[valid_mask].copy()

    LOGGER.info("Early-stopping split — fit: %d rows | valid: %s GW>%s, %d rows",
                len(fit), last_season, int(cutoff), len(valid))
    if 0 < len(valid) < 500:
        LOGGER.warning("  validation slice is only %d rows — early stopping will be noisy.",
                       len(valid))
    return fit, valid


# --------------------------------------------------------------------------- #
# Model construction — two stages
# --------------------------------------------------------------------------- #

def build_classifier(config: TrainConfig):
    """
    Stage 1: P(minutes > 0).

    NOTE: `scale_pos_weight` is deliberately unset. It would balance the classes
    and wreck calibration, and calibration is exactly what a multiplicative
    pipeline depends on.
    """
    if config.model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",     # proper scoring rule -> calibrated probabilities
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            n_jobs=-1,
        )

    if config.model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=config.n_estimators, learning_rate=config.learning_rate,
            max_depth=config.max_depth, subsample=0.8, colsample_bytree=0.8,
            random_state=config.random_state, n_jobs=-1, verbose=-1,
        )

    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=config.n_estimators, learning_rate=config.learning_rate,
        max_depth=config.max_depth, early_stopping=True, random_state=config.random_state,
    )


def build_regressor(config: TrainConfig):
    """Stage 2: E[points | plays]. Trained only on rows where minutes > 0."""
    if config.model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            # RMSE, not MAE: it matches the squared-error objective so the eval
            # curve is smooth. MAE against squared error plateaus early on thin
            # validation slices and halts the fit after a couple of rounds.
            eval_metric="rmse",
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            n_jobs=-1,
        )

    if config.model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=config.n_estimators, learning_rate=config.learning_rate,
            max_depth=config.max_depth, subsample=0.8, colsample_bytree=0.8,
            random_state=config.random_state, n_jobs=-1, verbose=-1,
        )

    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=config.n_estimators, learning_rate=config.learning_rate,
        max_depth=config.max_depth, early_stopping=True, random_state=config.random_state,
    )


def fit_stage(estimator, X_fit, y_fit, X_valid, y_valid, config: TrainConfig, label: str):
    """Fit with early stopping where the backend supports it."""
    if config.model_type == "xgboost" and len(X_valid):
        estimator.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], verbose=False)
        best = getattr(estimator, "best_iteration", None)
        if best is not None:
            LOGGER.info("    %s: early stop at iteration %d/%d",
                        label, best, config.n_estimators)
    elif config.model_type == "lightgbm" and len(X_valid):
        import lightgbm as lgb
        estimator.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)],
                      callbacks=[lgb.early_stopping(config.early_stopping_rounds,
                                                    verbose=False)])
    else:
        estimator.fit(X_fit, y_fit)
    return estimator


def fit_ensemble(build, X_fit, y_fit, X_valid, y_valid, config: TrainConfig, label: str):
    """
    Fit one stage `config.ensemble_size` times under different seeds.

    A size of 1 returns the plain estimator, exactly as before. Member seeds are
    `random_state * 100 + k`, so ensembles built from neighbouring base seeds do
    not share members.
    """
    if config.ensemble_size <= 1:
        return fit_stage(build(config), X_fit, y_fit, X_valid, y_valid, config, label)
    members = [
        fit_stage(build(replace(config, random_state=config.random_state * 100 + k)),
                  X_fit, y_fit, X_valid, y_valid, config, f"{label}[{k}]")
        for k in range(config.ensemble_size)
    ]
    wrapper = SeedAveragedClassifier if label == "clf" else SeedAveragedRegressor
    return wrapper(members)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """
    Rank correlation, without pulling in scipy.

    Spearman is just Pearson on the ranks, so `pandas.rank()` plus `corrcoef`
    gives the identical number (average ranks for ties, same as scipy default).
    """
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if len(ra) < 3 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ranking_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                    top_n: int = 15) -> Dict[str, float]:
    """
    Per-gameweek ranking quality — the metric the optimiser actually consumes.

    MAE and RMSE score the level of a prediction; the squad optimiser only ever
    compares players against each other within a gameweek. A model can be
    badly calibrated in level and still pick the right 15.
    """
    if "gameweek" not in frame.columns:
        return {"spearman_rho": float("nan"), "top15_capture": float("nan")}

    work = pd.DataFrame({"gameweek": frame["gameweek"].to_numpy(),
                         "actual": y_true, "pred": y_pred})
    rhos, captures = [], []
    for _, block in work.groupby("gameweek"):
        if len(block) < 10:
            continue
        rho = spearman_rho(block["pred"].to_numpy(), block["actual"].to_numpy())
        if not np.isnan(rho):
            rhos.append(rho)
        best = block.nlargest(top_n, "actual")["actual"].sum()
        got = block.nlargest(top_n, "pred")["actual"].sum()
        if best > 0:
            captures.append(got / best)

    return {
        "spearman_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "top15_capture": float(np.mean(captures)) if captures else float("nan"),
    }


def classifier_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """
    AUC plus the two that matter for a multiplicative pipeline.

    AUC is rank-only and invariant to monotone rescaling, so it cannot detect
    systematic over- or under-confidence. Brier and log-loss are proper scoring
    rules and will.
    """
    metrics: Dict[str, float] = {"base_rate": float(np.mean(y_true))}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")     # single-class slice
    try:
        metrics["brier"] = float(brier_score_loss(y_true, prob))
        metrics["logloss"] = float(log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7)))
    except ValueError:
        metrics["brier"] = metrics["logloss"] = float("nan")
    # Calibration in one number: are we over- or under-predicting on average?
    metrics["calibration_bias"] = float(np.mean(prob) - np.mean(y_true))
    return metrics


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": _rmse(y_true, y_pred)}


def naive_baselines(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Reference predictors the model must beat to be worth deploying."""
    baselines: Dict[str, np.ndarray] = {
        "baseline_position_mean": np.full(len(test), float(train[TARGET].mean())),
        # The MAE-optimal constant. FPL points are zero-inflated and right-skewed,
        # so the median is a harder MAE floor than the mean.
        "baseline_position_median": np.full(len(test), float(train[TARGET].median())),
    }
    roll3 = f"{TARGET}_roll_3"
    if roll3 in test.columns:
        values = pd.to_numeric(test[roll3], errors="coerce")
        baselines["baseline_roll_3"] = values.fillna(train[TARGET].mean()).to_numpy()
    return baselines


def evaluate_predictions(
    test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
    baselines: Dict[str, np.ndarray], clf: Dict[str, float],
) -> pd.DataFrame:
    """Score model and baselines, overall and on appearances only."""
    rows: List[Dict[str, Any]] = []
    segments: List[Tuple[str, np.ndarray]] = [("all_rows", np.ones(len(test), dtype=bool))]
    played = test["played"].to_numpy().astype(bool)
    segments.append(("played_only", played))

    predictions = {"model": y_pred, **baselines}
    for segment_name, mask in segments:
        if mask.sum() == 0:
            continue
        for label, values in predictions.items():
            entry = {"segment": segment_name, "predictor": label, "n": int(mask.sum()),
                     **score(y_true[mask], values[mask]),
                     **ranking_metrics(test[mask], y_true[mask], values[mask])}
            if label == "model" and segment_name == "all_rows":
                entry.update(clf)
            rows.append(entry)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #

def compute_importance(estimator, X_test, y_test, features, config: TrainConfig,
                       scoring: str) -> pd.DataFrame:
    """Gain importance plus permutation importance on the test season."""
    frame = pd.DataFrame({"feature": list(features)})
    native = getattr(estimator, "feature_importances_", None)
    frame["gain_importance"] = native if native is not None else np.nan

    if len(X_test) > 0 and config.permutation_repeats > 0:
        sample = min(len(X_test), config.permutation_sample)
        rng = np.random.default_rng(config.random_state)
        idx = (rng.choice(len(X_test), size=sample, replace=False)
               if sample < len(X_test) else np.arange(len(X_test)))
        try:
            result = permutation_importance(
                estimator, X_test.iloc[idx], y_test[idx],
                n_repeats=config.permutation_repeats, random_state=config.random_state,
                scoring=scoring, n_jobs=1)
            frame["permutation_importance"] = result.importances_mean
            frame["permutation_std"] = result.importances_std
        except Exception as exc:
            LOGGER.warning("Permutation importance failed: %s", exc)
            frame["permutation_importance"] = np.nan
            frame["permutation_std"] = np.nan
    else:
        frame["permutation_importance"] = np.nan
        frame["permutation_std"] = np.nan

    key = ("permutation_importance" if frame["permutation_importance"].notna().any()
           else "gain_importance")
    return frame.sort_values(key, ascending=False).reset_index(drop=True)


def plot_importance(frame: pd.DataFrame, position: str, stage: str,
                    path: Path, top_n: int = 15) -> None:
    column = ("permutation_importance" if frame["permutation_importance"].notna().any()
              else "gain_importance")
    top = frame.nlargest(top_n, column).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.36 * len(top))))
    ax.barh(top["feature"], top[column], color="#38003c" if stage == "pts" else "#00ff87")
    ax.set_xlabel(column.replace("_", " "))
    ax.set_title(f"{position} — {stage} stage — top {len(top)} features")
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

def with_odds_masked(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """
    Append a copy of every row with the odds features blanked out.

    Odds are only published for the next round. A model that has only ever
    seen them present would be scoring an unfamiliar input whenever they are
    missing: GW t+2 in a multi-week plan, or a live run without an odds feed.
    Training on both copies teaches XGBoost's missing-value branches to fall
    back on the FDR features, while rows with odds still use them.
    """
    odds = [c for c in features if c.startswith(ODDS_PREFIX)]
    if not odds or frame.empty:
        return frame
    masked = frame.copy()
    masked[odds] = np.nan
    return pd.concat([frame, masked], ignore_index=True)


def train_position_model(position: str, train: pd.DataFrame, test: pd.DataFrame,
                         features: Sequence[str], config: TrainConfig) -> Optional[Dict[str, Any]]:
    """Train both stages for one position, evaluate the product, package it."""
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
    active = drop_constant_features(prepare_matrix(train_pos, features), features)
    if len(active) < len(features):
        LOGGER.info("  using %d/%d features (%d constant within this position)",
                    len(active), len(features), len(features) - len(active))
    if config.odds_dropout:
        fit_df, valid_df = with_odds_masked(fit_df, active), with_odds_masked(valid_df, active)

    X_fit, X_valid = prepare_matrix(fit_df, active), prepare_matrix(valid_df, active)
    X_test = prepare_matrix(test_pos, active)
    y_test = test_pos[TARGET].to_numpy(dtype=float)

    # ---- Stage 1: P(plays), on ALL rows ----
    classifier = fit_ensemble(
        build_classifier,
        X_fit, fit_df["played"].to_numpy(),
        X_valid, valid_df["played"].to_numpy(),
        config, "clf")

    # ---- Stage 2: E[points | plays], on PLAYED rows only ----
    fit_played, valid_played = fit_df[fit_df["played"] == 1], valid_df[valid_df["played"] == 1]
    LOGGER.info("    reg: %d of %d fit rows are appearances (%.0f%%)",
                len(fit_played), len(fit_df), 100 * len(fit_played) / max(len(fit_df), 1))
    if len(fit_played) < 50:
        LOGGER.warning("  skipping %s: only %d appearances to regress on.",
                       position, len(fit_played))
        return None

    regressor = fit_ensemble(
        build_regressor,
        prepare_matrix(fit_played, active), fit_played[TARGET].to_numpy(dtype=float),
        prepare_matrix(valid_played, active), valid_played[TARGET].to_numpy(dtype=float),
        config, "reg")

    model = TwoStageModel(classifier, regressor, active, position)

    # ---- evaluate the PRODUCT, plus each stage on its own terms ----
    prob, conditional = model.predict_parts(X_test)
    y_pred = np.clip(prob * conditional, 0, None)

    clf = classifier_metrics(test_pos["played"].to_numpy(), prob)
    metrics = evaluate_predictions(test_pos, y_test, y_pred,
                                   naive_baselines(train_pos, test_pos), clf)
    metrics.insert(0, "position", position)

    played_mask = test_pos["played"].to_numpy().astype(bool)
    reg_only = (score(y_test[played_mask], np.clip(conditional[played_mask], 0, None))
                if played_mask.sum() else {"mae": np.nan, "rmse": np.nan})

    importance = pd.concat([
        compute_importance(classifier, X_test, test_pos["played"].to_numpy(), active,
                           config, "neg_log_loss").assign(stage="prob", position=position),
        compute_importance(regressor, X_test[played_mask], y_test[played_mask], active,
                           config, "neg_mean_absolute_error").assign(stage="pts",
                                                                     position=position),
    ], ignore_index=True)

    headline = metrics[(metrics.segment == "all_rows") & (metrics.predictor == "model")].iloc[0]
    LOGGER.info("  combined  MAE %.4f | RMSE %.4f | rho %.4f  (n=%d)",
                headline["mae"], headline["rmse"], headline["spearman_rho"], int(headline["n"]))
    LOGGER.info("  stage 1   AUC %.4f | Brier %.4f | logloss %.4f | bias %+.4f",
                clf["roc_auc"], clf["brier"], clf["logloss"], clf["calibration_bias"])
    LOGGER.info("  stage 2   MAE %.4f | RMSE %.4f  (appearances only)",
                reg_only["mae"], reg_only["rmse"])

    return {"position": position, "model": model, "classifier": classifier,
            "regressor": regressor, "features": active, "metrics": metrics,
            "importance": importance, "classifier_metrics": clf,
            "regressor_metrics": reg_only, "n_train": len(fit_df),
            "n_train_played": len(fit_played), "n_test": len(test_pos)}


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

def save_artifacts(results: Dict[str, Dict[str, Any]], features: Sequence[str],
                   config: TrainConfig) -> Dict[str, Path]:
    """Three files per position: the two stages, plus a combined wrapper."""
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    all_metrics, all_importance = [], []

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for position, result in results.items():
        common = {
            "position": position,
            "features": result["features"],
            "model_type": config.model_type,
            "train_seasons": list(config.train_seasons),
            "test_season": config.test_season,
            "trained_at": stamp,
        }

        joblib.dump({**common, "model": result["classifier"], "stage": "prob",
                     "target": "minutes > 0", "metrics": result["classifier_metrics"]},
                    model_dir / f"fpl_model_{position}_prob.joblib")
        joblib.dump({**common, "model": result["regressor"], "stage": "pts",
                     "target": f"{TARGET} | minutes > 0",
                     "metrics": result["regressor_metrics"]},
                    model_dir / f"fpl_model_{position}_pts.joblib")

        # Combined wrapper at the ORIGINAL path so Phase 4 and anything else
        # calling bundle["model"].predict(X) keeps working untouched.
        path = model_dir / f"fpl_model_{position}.joblib"
        joblib.dump({**common, "model": result["model"], "stage": "two_stage",
                     "target": TARGET,
                     "metrics": result["metrics"].to_dict(orient="records"),
                     "n_train": result["n_train"], "n_train_played": result["n_train_played"],
                     "n_test": result["n_test"]}, path)
        written[position] = path

        result["importance"].to_csv(
            model_dir / f"feature_importance_{position}.csv", index=False)
        for stage in ("prob", "pts"):
            block = result["importance"][result["importance"]["stage"] == stage]
            if not block.empty:
                plot_importance(block, position, stage,
                                model_dir / f"feature_importance_{position}_{stage}.png")

        all_metrics.append(result["metrics"])
        all_importance.append(result["importance"])

    metadata = {
        "architecture": "two_stage",
        "stages": {"prob": "P(minutes > 0), binary:logistic",
                   "pts": f"E[{TARGET} | minutes > 0], reg:squarederror"},
        "combination": "predicted_points = P(plays) * E[points | plays]",
        "target": TARGET,
        "positions": list(results),
        "feature_columns": list(features),
        "per_position_features": {p: r["features"] for p, r in results.items()},
        "pre_deadline_features": PRE_DEADLINE_FEATURES,
        "fdr_prefix": FDR_PREFIX,
        "post_match_blocklist": sorted(POST_MATCH_COLUMNS),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(config).items() if k != "metadata"},
        "trained_at": stamp,
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

    LOGGER.info("Saved %d position(s) x 3 files + metadata to %s/", len(written), model_dir)
    return written


# --------------------------------------------------------------------------- #
# Orchestration and reporting
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
        raise RuntimeError("No position models could be trained — check the split and input.")

    written = save_artifacts(results, features, config)
    summary = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
    return {"results": results, "summary": summary, "paths": written}


def print_report(summary: pd.DataFrame, results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 86)
    print("TEST-SEASON PERFORMANCE — TWO-STAGE  P(plays) x E[points | plays]")
    print("=" * 86)

    print("\n[stage 1 — classifier, P(minutes > 0)]")
    clf = pd.DataFrame({p: r["classifier_metrics"] for p, r in results.items()}).T
    print(clf[["base_rate", "roc_auc", "brier", "logloss", "calibration_bias"]]
          .round(4).to_string())
    print("  AUC is rank-only. Brier and log-loss are what a multiplicative model needs;")
    print("  calibration_bias is mean(predicted) - mean(actual), so 0 is perfectly centred.")

    print("\n[stage 2 — regressor, E[points | plays], appearances only]")
    reg = pd.DataFrame({p: r["regressor_metrics"] for p, r in results.items()}).T
    print(reg.round(4).to_string())

    for segment in summary["segment"].unique():
        block = summary[summary.segment == segment]
        print(f"\n[combined — {segment}]")
        print(block.pivot_table(index="position", columns="predictor",
                                values=["mae", "rmse"]).round(4).to_string())

    print("\n[ranking quality — what the optimiser actually consumes]")
    rank = summary[(summary.segment == "all_rows")].pivot_table(
        index="position", columns="predictor", values=["spearman_rho", "top15_capture"])
    print(rank.round(4).to_string())

    print("\n" + "-" * 86)
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
            verdict = "beats" if delta > 0 else "LOSES TO"
            regressions += delta <= 0
            print(f"    {position}: model {model_mae:.4f} {verdict} {best['predictor']} "
                  f"{best['mae']:.4f}  (delta {delta:+.4f})")

    if regressions:
        print(f"\n  NOTE: {regressions} position/segment combination(s) lose on MAE. Check RMSE")
        print("  and Spearman before concluding anything: a squared-error model predicts")
        print("  conditional MEANS, while MAE's optimal constant is the MEDIAN, and FPL")
        print("  points are right-skewed. Losing MAE to a median baseline is expected.")
    print("-" * 86)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Phase 3 (V4) two-stage model training.")
    parser.add_argument("-i", "--input", default="fpl_ml_features.csv")
    parser.add_argument("-d", "--model-dir", default="models")
    parser.add_argument("--train-seasons", nargs="+", default=["2023-24", "2024-25"])
    parser.add_argument("--test-season", default="2025-26")
    parser.add_argument("--model", dest="model_type", default="xgboost",
                        choices=["xgboost", "lightgbm", "hist"])
    parser.add_argument("--valid-gameweeks", type=int, default=10)
    parser.add_argument("--min-games-prior", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Seeds per stage, predictions averaged (default: 1)")
    parser.add_argument("--no-odds-dropout", action="store_true",
                        help="Train only on rows as they are, without the odds-masked copy")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    config = TrainConfig(
        input_path=Path(args.input), model_dir=Path(args.model_dir),
        train_seasons=tuple(args.train_seasons), test_season=args.test_season,
        valid_gameweeks=args.valid_gameweeks, model_type=args.model_type,
        min_games_prior=args.min_games_prior, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, max_depth=args.max_depth,
        permutation_repeats=args.permutation_repeats, random_state=args.random_state,
        odds_dropout=not args.no_odds_dropout, ensemble_size=args.ensemble,
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

    print_report(outcome["summary"], outcome["results"])

    print("\nTop 6 features per position and stage:")
    for position, result in outcome["results"].items():
        for stage in ("prob", "pts"):
            block = result["importance"]
            block = block[block["stage"] == stage].head(6)["feature"].tolist()
            print(f"  {position} [{stage}]: {', '.join(block)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
