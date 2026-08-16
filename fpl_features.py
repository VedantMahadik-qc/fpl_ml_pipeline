"""
FPL ML Pipeline — Phase 2: Feature Engineering
==============================================

Consumes `fpl_combined_history.csv` (Phase 1) and produces `fpl_ml_features.csv`:
leak-free rolling form features, ready for a gameweek-level points model.

Pipeline
--------
  1. Drop `xp`                      — post-match scrape, severe lookahead bias.
  2. Benchwarmers adjustments       — expected_assists x 1.4
                                      clean_sheet_probability = exp(-xGC)
  3. Sort + group by player         — chronological, one series per player.
  4. Rolling means (3/5/10)         — every window is .shift(1)'d, so gameweek N
                                      sees only N-1 and older. Never N itself.
  5. Cold-start imputation          — NaN rolling values filled with the median
                                      for that player's position. Rows are kept.
  6. Export                         — fpl_ml_features.csv

Usage
-----
    python fpl_features.py
    python fpl_features.py --input fpl_combined_history.csv --output fpl_ml_features.csv
    python fpl_features.py --windows 3 5 10 --median-groups position season
    python fpl_features.py --group-by element        # literal player_id grouping
    python fpl_features.py --partial-windows         # min_periods=1


IMPORTANT — grouping key
------------------------
The spec said "group by `player_id`". `player_id` is the FPL `element` id, which
is **re-issued every season**: element 328 is one player in 2023-24 and someone
else entirely in 2025-26. Grouping on it across seasons splices two unrelated
players into one time series at each season boundary, and the corruption lands
exactly on GW1-GW10 of each new season — the rows a transfer model cares about.

So the default grouping key here is `player_name`, which is stable across
seasons. `--group-by element` reproduces the literal spec, and
`--group-by season_element` is the paranoid option (correct, but every player
cold-starts at GW1). The chosen key is logged on every run.


Leakage notes worth reading before you trust the output
-------------------------------------------------------
* Rolling windows are strictly causal. `verify_no_leakage()` re-derives a sample
  of feature values from prior rows only and asserts they match; it runs by
  default and fails loudly.
* Position medians are fitted over the whole file. That is mild lookahead: a
  2023-24 row is filled from a median that saw 2025-26 data. For a clean
  backtest, fit the imputer on your training slice and reuse it — the imputer is
  a fit/transform class and `--medians-out` writes the fitted values to JSON.
* `defensive_contribution` only exists from 2025/26 onward. A global position
  median therefore fabricates it for 2023-24 and 2024-25. The script reports
  exactly how many rows that affects; `--median-groups position season` scopes
  medians per season instead, and `defensive_contribution_observed` flags rows
  where the raw stat genuinely existed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fpl.features")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Columns dropped before anything else touches the frame.
LOOKAHEAD_COLUMNS: List[str] = ["xp"]

# Metrics that get rolling means.
ROLLING_METRICS: List[str] = [
    "minutes",
    "total_points",
    "expected_goals",
    "expected_assists",
    "defensive_contribution",
    "clean_sheet_probability",
]

DEFAULT_WINDOWS: List[int] = [3, 5, 10]

# FPL Benchwarmers: FPL's raw xA understates fantasy assist value.
FANTASY_ASSIST_MULTIPLIER: float = 1.4

GROUP_KEY_CHOICES = {
    "name": "player_name",
    "element": "player_id",
    "season_element": "_season_element",
}


@dataclass
class FeatureConfig:
    input_path: Path = Path("fpl_combined_history.csv")
    output_path: Path = Path("fpl_ml_features.csv")
    medians_path: Optional[Path] = Path("fpl_position_medians.json")
    windows: Sequence[int] = tuple(DEFAULT_WINDOWS)
    metrics: Sequence[str] = tuple(ROLLING_METRICS)
    group_by: str = "name"
    median_groups: Sequence[str] = ("position",)
    partial_windows: bool = False      # True -> min_periods=1 instead of window
    assist_multiplier: float = FANTASY_ASSIST_MULTIPLIER
    verify: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 0. Load
# --------------------------------------------------------------------------- #

def load_history(path: Path) -> pd.DataFrame:
    """Read the Phase 1 combined CSV and check the columns we depend on exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fpl_ingest.py first to produce the combined history."
        )
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(f"Could not parse {path}: {exc}") from exc

    required = {"player_id", "player_name", "season", "gameweek", "position"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")

    LOGGER.info("Loaded %s: %d rows x %d columns", path, len(frame), frame.shape[1])
    return frame


# --------------------------------------------------------------------------- #
# 1. Drop lookahead columns
# --------------------------------------------------------------------------- #

def drop_lookahead_columns(
    frame: pd.DataFrame, columns: Iterable[str] = LOOKAHEAD_COLUMNS
) -> pd.DataFrame:
    """
    Remove post-match columns before any other step.

    `xp` is FPL's own expected-points figure as scraped by the archive *after*
    the fixture resolved. Any model that sees it is reading the answer sheet.
    """
    present = [c for c in columns if c in frame.columns]
    if present:
        LOGGER.info("Dropping lookahead column(s): %s", ", ".join(present))
        frame = frame.drop(columns=present)
    else:
        LOGGER.info("No lookahead columns present (already dropped upstream).")
    return frame


# --------------------------------------------------------------------------- #
# 2. FPL Benchwarmers adjustments
# --------------------------------------------------------------------------- #

def apply_benchwarmer_adjustments(
    frame: pd.DataFrame, assist_multiplier: float = FANTASY_ASSIST_MULTIPLIER
) -> pd.DataFrame:
    """
    Two community adjustments, applied to raw per-gameweek rows *before* rolling.

    Fantasy assists
        FPL's `expected_assists` counts the pass that creates a chance. Fantasy
        assists are awarded more loosely (deflections, rebounds, won penalties),
        so the community scales xA up by ~1.4.

    Poisson clean sheets
        Goals conceded are near-Poisson, so P(0 conceded) = exp(-lambda) with
        lambda = expected_goals_conceded. This turns a continuous defensive
        metric into a calibrated probability, which is what the CS points are.
    """
    frame = frame.copy()

    if "expected_assists" in frame.columns:
        frame["expected_assists"] = (
            pd.to_numeric(frame["expected_assists"], errors="coerce") * assist_multiplier
        )
        LOGGER.info("Scaled expected_assists by %.2fx (fantasy assist adjustment).",
                    assist_multiplier)
    else:
        LOGGER.warning("`expected_assists` not found — skipping fantasy assist adjustment.")

    if "expected_goals_conceded" in frame.columns:
        xgc = pd.to_numeric(frame["expected_goals_conceded"], errors="coerce")
        frame["clean_sheet_probability"] = np.exp(-xgc)
        observed = frame["clean_sheet_probability"].notna().sum()
        LOGGER.info("Built clean_sheet_probability = exp(-xGC) for %d/%d rows.",
                    observed, len(frame))
    else:
        LOGGER.warning("`expected_goals_conceded` not found — clean_sheet_probability is NaN.")
        frame["clean_sheet_probability"] = np.nan

    return frame


def flag_observed_metrics(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """
    Record whether each metric was genuinely measured on this row.

    Matters because `defensive_contribution` did not exist before 2025/26: after
    imputation you cannot tell a real 0.4 from a fabricated one without this.
    """
    frame = frame.copy()
    for metric in metrics:
        if metric in frame.columns:
            frame[f"{metric}_observed"] = frame[metric].notna().astype("int8")
    return frame


def report_absent_metrics(frame: pd.DataFrame, metrics: Iterable[str]) -> None:
    """Log any season x metric combination with no raw data at all."""
    if "season" not in frame.columns:
        return
    for metric in metrics:
        if metric not in frame.columns:
            continue
        coverage = frame.groupby("season")[metric].apply(lambda s: s.notna().mean())
        empty = coverage[coverage == 0.0]
        for season in empty.index:
            n = int((frame["season"] == season).sum())
            LOGGER.warning(
                "`%s` has ZERO raw values in %s (%d rows). Its rolling features there "
                "will be entirely imputed — consider --median-groups position season.",
                metric, season, n,
            )


# --------------------------------------------------------------------------- #
# 3. Sorting + grouping key
# --------------------------------------------------------------------------- #

def build_group_key(frame: pd.DataFrame, group_by: str) -> tuple[pd.DataFrame, str]:
    """Resolve the per-player series key. See the module docstring for why."""
    frame = frame.copy()

    if group_by not in GROUP_KEY_CHOICES:
        raise ValueError(f"group_by must be one of {sorted(GROUP_KEY_CHOICES)}")

    if group_by == "season_element":
        frame["_season_element"] = (
            frame["season"].astype(str) + ":" + frame["player_id"].astype(str)
        )
    key = GROUP_KEY_CHOICES[group_by]

    if group_by == "element":
        LOGGER.warning(
            "Grouping by raw player_id. FPL element ids are re-issued each season, so "
            "rolling windows will splice different players together at season boundaries."
        )
    LOGGER.info("Grouping player series by `%s` (%d distinct players).",
                key, frame[key].nunique())
    return frame, key


def sort_chronologically(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Order rows within each player series: season, then gameweek, then kickoff.

    `kickoff_time` is the tiebreaker that matters. In a double gameweek a player
    has two rows sharing one `gameweek`, and without a stable tiebreak `.shift(1)`
    could hand the later fixture's result to the earlier one.
    """
    sort_cols = [key, "season", "gameweek"]
    if "kickoff_time" in frame.columns:
        frame = frame.copy()
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)
        sort_cols.append("kickoff_time")
    else:
        LOGGER.warning("No `kickoff_time` column — double gameweeks may order non-deterministically.")

    return frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Rolling features (strictly causal)
# --------------------------------------------------------------------------- #

def rolling_column_name(metric: str, window: int) -> str:
    return f"{metric}_roll_{window}"


def add_rolling_features(
    frame: pd.DataFrame,
    key: str,
    metrics: Sequence[str],
    windows: Sequence[int],
    partial_windows: bool = False,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Rolling means over the previous N gameweeks, per player.

    The `.shift(1)` happens BEFORE `.rolling(...)`, so the window for gameweek N
    covers N-1, N-2, ... and never N. Shifting after rolling would leak the
    current row into its own feature.

    min_periods defaults to the full window: a player with 4 prior games gets NaN
    for the 10-game feature (and is imputed later), rather than a "10-game mean"
    silently computed from 4. `partial_windows=True` relaxes this to 1.
    """
    frame = frame.copy()
    available = [m for m in metrics if m in frame.columns]
    for missing in set(metrics) - set(available):
        LOGGER.warning("Metric `%s` not in data — skipping its rolling features.", missing)
    if not available:
        raise RuntimeError("None of the requested rolling metrics exist in the data.")

    for metric in available:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    grouped = frame.groupby(key, sort=False)

    # Shift once, then roll the shifted frame — cheaper than shifting per window.
    shifted = grouped[available].shift(1)
    shifted[key] = frame[key].values
    shifted_groups = shifted.groupby(key, sort=False)

    created: List[str] = []
    for window in windows:
        min_periods = 1 if partial_windows else window
        rolled = (
            shifted_groups[available]
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        for metric in available:
            name = rolling_column_name(metric, window)
            frame[name] = rolled[metric]
            created.append(name)
        LOGGER.info("Built %d rolling features for window=%d (min_periods=%d).",
                    len(available), window, min_periods)

    # How much real history backs each row — lets the model discount imputed values.
    frame["games_prior"] = grouped.cumcount()
    frame["is_cold_start"] = (frame["games_prior"] < max(windows)).astype("int8")

    return frame, created


# --------------------------------------------------------------------------- #
# 5. Cold-start imputation
# --------------------------------------------------------------------------- #

class PositionMedianImputer:
    """
    Fills NaN rolling features with the median for that player's position.

    Written as fit/transform rather than a one-shot function so the medians can
    be fitted on a training slice and reused at inference — otherwise every
    backtest silently imputes from data the model should not have seen. Medians
    round-trip through JSON via save()/load().

    Fallback chain per cell: group median -> global median -> 0.0.
    """

    def __init__(self, group_cols: Sequence[str] = ("position",)) -> None:
        self.group_cols = list(group_cols)
        self.medians_: Dict[str, Dict[str, float]] = {}
        self.global_medians_: Dict[str, float] = {}
        self.columns_: List[str] = []

    @staticmethod
    def _group_token(values: Sequence) -> str:
        return "|".join("NA" if pd.isna(v) else str(v) for v in values)

    def fit(self, frame: pd.DataFrame, columns: Sequence[str]) -> "PositionMedianImputer":
        missing = [c for c in self.group_cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Cannot group medians by missing column(s): {missing}")

        self.columns_ = [c for c in columns if c in frame.columns]
        self.global_medians_ = {
            c: float(v) if pd.notna(v := frame[c].median()) else 0.0 for c in self.columns_
        }

        grouped = frame.groupby(self.group_cols, dropna=False)[self.columns_].median()
        self.medians_ = {}
        for group_values, row in grouped.iterrows():
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            token = self._group_token(group_values)
            self.medians_[token] = {
                c: (float(row[c]) if pd.notna(row[c]) else self.global_medians_[c])
                for c in self.columns_
            }

        LOGGER.info("Fitted medians for %d columns across %d %s group(s).",
                    len(self.columns_), len(self.medians_), "+".join(self.group_cols))
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (filled frame, per-column NaN counts before/after)."""
        if not self.columns_:
            raise RuntimeError("Imputer not fitted — call fit() first.")

        frame = frame.copy()
        before = frame[self.columns_].isna().sum()

        tokens = frame[self.group_cols].astype(object).apply(
            lambda row: self._group_token(row.values), axis=1
        )

        for column in self.columns_:
            fill = tokens.map(
                lambda t: self.medians_.get(t, {}).get(column, self.global_medians_[column])
            )
            frame[column] = frame[column].fillna(fill)
            # Any group unseen at fit time still falls through to the global median.
            frame[column] = frame[column].fillna(self.global_medians_[column])

        after = frame[self.columns_].isna().sum()
        report = pd.DataFrame({"nan_before": before, "nan_after": after})
        report["imputed"] = report["nan_before"] - report["nan_after"]
        return frame, report

    def fit_transform(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.fit(frame, columns).transform(frame)

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "group_cols": self.group_cols,
            "columns": self.columns_,
            "medians": self.medians_,
            "global_medians": self.global_medians_,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved fitted medians to %s (reuse these at inference).", path)

    @classmethod
    def load(cls, path: Path) -> "PositionMedianImputer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        imputer = cls(payload["group_cols"])
        imputer.columns_ = payload["columns"]
        imputer.medians_ = payload["medians"]
        imputer.global_medians_ = payload["global_medians"]
        return imputer


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify_no_leakage(
    frame: pd.DataFrame,
    key: str,
    metrics: Sequence[str],
    windows: Sequence[int],
    sample_size: int = 400,
    tolerance: float = 1e-8,
) -> None:
    """
    Independently re-derive rolling values from prior rows and assert they match.

    Deliberately naive and slow: it recomputes from raw history rather than
    trusting the vectorised path, so a bug in `add_rolling_features` cannot hide
    behind the same bug here. Raises AssertionError on any mismatch.
    """
    checked = 0
    rng = np.random.default_rng(0)

    # Positional, not label-based. `Index.to_numpy()` can hand back a view of the
    # index's own cached array, and shuffling that in place corrupts the index
    # while it still *looks* intact — so work off a fresh positional array and
    # slice with .iloc. The frame is already sorted, so position == chronology.
    keys = frame[key].to_numpy(copy=True)
    positions = np.arange(len(frame))
    rng.shuffle(positions)

    for pos in positions:
        if checked >= sample_size:
            break
        prior_mask = keys[:pos] == keys[pos]
        if not prior_mask.any():
            continue
        prior = frame.iloc[:pos][prior_mask]

        for metric in metrics:
            if metric not in frame.columns:
                continue
            for window in windows:
                column = rolling_column_name(metric, window)
                if column not in frame.columns or f"{column}__preimpute" not in frame.columns:
                    continue
                actual = frame[f"{column}__preimpute"].iloc[pos]
                if pd.isna(actual):
                    continue
                expected = prior[metric].tail(window).mean()
                assert not pd.isna(expected), f"{column}@{pos}: expected NaN, got {actual}"
                assert abs(expected - actual) < tolerance, (
                    f"LEAKAGE/MISMATCH at position {pos}, {column}: "
                    f"expected {expected:.10f} from prior rows, got {actual:.10f}"
                )
        checked += 1

    LOGGER.info("Leakage check passed: %d sampled rows re-derived from prior gameweeks only.",
                checked)


def summarise(frame: pd.DataFrame, created: Sequence[str]) -> None:
    """Log a compact profile of what was built."""
    if "season" in frame.columns:
        per_season = (
            frame.groupby("season")
            .agg(rows=("player_id", "size"),
                 players=("player_id", "nunique"),
                 cold_start_rows=("is_cold_start", "sum"))
            .reset_index()
        )
        LOGGER.info("Per-season profile:\n%s", per_season.to_string(index=False))

    remaining = frame[list(created)].isna().sum().sum()
    LOGGER.info("Engineered %d rolling features; %d NaNs remain after imputation.",
                len(created), int(remaining))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_features(config: FeatureConfig) -> pd.DataFrame:
    """Full Phase 2 pipeline."""
    frame = load_history(config.input_path)

    # 1. Lookahead columns go first, before anything can depend on them.
    frame = drop_lookahead_columns(frame)

    # 2. Benchwarmers adjustments on raw rows, pre-rolling.
    frame = apply_benchwarmer_adjustments(frame, config.assist_multiplier)
    frame = flag_observed_metrics(frame, config.metrics)
    report_absent_metrics(frame, config.metrics)

    # 3. Chronological order within each player series.
    frame, key = build_group_key(frame, config.group_by)
    frame = sort_chronologically(frame, key)

    # 4. Strictly causal rolling means.
    frame, created = add_rolling_features(
        frame, key, config.metrics, config.windows, config.partial_windows
    )

    # Snapshot pre-imputation values so verification checks the real computation.
    for column in created:
        frame[f"{column}__preimpute"] = frame[column]

    if config.verify:
        verify_no_leakage(frame, key, config.metrics, config.windows)

    # 5. Cold-start imputation — fills NaNs, keeps every row.
    imputer = PositionMedianImputer(config.median_groups)
    frame, report = imputer.fit_transform(frame, created)
    LOGGER.info("Imputation summary (top 10 by count):\n%s",
                report.sort_values("imputed", ascending=False).head(10).to_string())
    if config.medians_path:
        imputer.save(config.medians_path)

    # Tidy: drop verification scaffolding and the internal group key.
    frame = frame.drop(columns=[c for c in frame.columns if c.endswith("__preimpute")])
    frame = frame.drop(columns=[c for c in ["_season_element"] if c in frame.columns])

    summarise(frame, created)
    return frame


def export(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {path}: {exc}") from exc
    LOGGER.info("Wrote %s: %d rows x %d columns", path, len(frame), frame.shape[1])
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Phase 2 feature engineering.")
    parser.add_argument("-i", "--input", default="fpl_combined_history.csv")
    parser.add_argument("-o", "--output", default="fpl_ml_features.csv")
    parser.add_argument("--medians-out", default="fpl_position_medians.json",
                        help="Where to save fitted medians (use '' to skip)")
    parser.add_argument("-w", "--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
                        help=f"Rolling windows (default: {' '.join(map(str, DEFAULT_WINDOWS))})")
    parser.add_argument("-m", "--metrics", nargs="+", default=ROLLING_METRICS,
                        help="Metrics to roll")
    parser.add_argument("-g", "--group-by", default="name",
                        choices=sorted(GROUP_KEY_CHOICES),
                        help="Player series key: name (default) | element | season_element")
    parser.add_argument("--median-groups", nargs="+", default=["position"],
                        help="Columns defining imputation groups (default: position)")
    parser.add_argument("--assist-multiplier", type=float, default=FANTASY_ASSIST_MULTIPLIER)
    parser.add_argument("--partial-windows", action="store_true",
                        help="min_periods=1 instead of the full window")
    parser.add_argument("--no-verify", action="store_true", help="Skip the leakage assertion")
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

    config = FeatureConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        medians_path=Path(args.medians_out) if args.medians_out else None,
        windows=tuple(args.windows),
        metrics=tuple(args.metrics),
        group_by=args.group_by,
        median_groups=tuple(args.median_groups),
        partial_windows=args.partial_windows,
        assist_multiplier=args.assist_multiplier,
        verify=not args.no_verify,
    )

    try:
        frame = build_features(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except AssertionError as exc:
        LOGGER.error("Leakage verification FAILED — refusing to write output. %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    export(frame, config.output_path)

    preview = ["season", "player_name", "position", "gameweek", "games_prior",
               "total_points", "total_points_roll_3", "total_points_roll_5",
               "expected_goals_roll_3", "clean_sheet_probability_roll_3"]
    print("\nPreview:")
    print(frame[[c for c in preview if c in frame.columns]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
