"""
FPL ML Pipeline — Bookmaker odds features
=========================================

Turns pre-match betting odds into per-fixture goal expectations. The market
prices team news, rotation and form that xG-derived ratings cannot see, so it is
the strongest public signal for goals and clean sheets.

Source: football-data.co.uk, one CSV per Premier League season. The `Avg*`
columns are the market average collected before the round (Friday afternoon
for weekend fixtures, Tuesday for midweek), i.e. around the FPL deadline. The
closing (`*C*`) columns are taken at kickoff, after the deadline, and are
deliberately not used.

From margin-free 1X2 and over/under-2.5 probabilities we fit the independent
Poisson rates (lambda_home, lambda_away) that reproduce them best:

    P(home win)   = sum_{h > a} Pois(h; lambda_home) * Pois(a; lambda_away)
    P(over 2.5)   = 1 - sum_{h + a <= 2} ...

Per-row features, from the player's team's point of view:

    odds_team_goals      lambda for his team
    odds_team_conceded   lambda for the opponent
    odds_cs_prob         exp(-odds_team_conceded)
    odds_win_prob        margin-free market probability his team wins

Odds exist only for the next round at any deadline. Phase 3 therefore trains on
a second, odds-masked copy of each row so the model also handles fixtures
further ahead (and live runs without odds) by falling back on the FDR features.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fpl.odds")

ODDS_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
ODDS_CACHE = Path(".fpl_cache/odds")

ODDS_FEATURES = ["odds_team_goals", "odds_team_conceded", "odds_cs_prob", "odds_win_prob"]
ODDS_PREFIX = "odds_"

# football-data.co.uk name -> FPL name, where they differ.
TEAM_NAMES: Dict[str, str] = {
    "Man United": "Man Utd",
    "Sheffield United": "Sheffield Utd",
    "Tottenham": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "Wolverhampton": "Wolves",
}

# Market-average pre-match prices first; a single bookmaker as the fallback.
PRICE_COLUMNS = [
    ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"),
    ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"),
]

MAX_GOALS = 10
LAMBDA_GRID = np.arange(0.05, 4.5, 0.025)


def season_code(season: str) -> str:
    """'2025-26' -> '2526'."""
    start, end = season.split("-")
    return start[-2:] + end[-2:]


def fetch_season_odds(season: str, cache_dir: Path = ODDS_CACHE) -> Optional[pd.DataFrame]:
    """Download (or read the cached) football-data CSV for one season."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"E0_{season_code(season)}.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")

    import requests

    url = ODDS_URL.format(code=season_code(season))
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "fpl-ml-pipeline"})
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Could not fetch odds for %s from %s: %s", season, url, exc)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")


def margin_free(prices: pd.DataFrame) -> np.ndarray:
    """Proportional de-vig: implied probabilities rescaled to sum to one."""
    implied = 1.0 / prices.to_numpy(dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)


def _grid_outcomes() -> Dict[str, np.ndarray]:
    """P(home), P(away), P(over 2.5) for every (lambda_home, lambda_away) on the grid."""
    goals = np.arange(MAX_GOALS + 1)
    from math import factorial
    log_fact = np.log([float(factorial(g)) for g in goals])
    lam = LAMBDA_GRID[:, None]
    pmf = np.exp(goals * np.log(lam) - lam - log_fact)            # (L, G)

    joint = pmf[:, None, :, None] * pmf[None, :, None, :]           # (Lh, La, h, a)
    h, a = np.meshgrid(goals, goals, indexing="ij")
    return {
        "home": (joint * (h > a)).sum(axis=(2, 3)).ravel(),
        "away": (joint * (h < a)).sum(axis=(2, 3)).ravel(),
        "over": 1.0 - (joint * (h + a <= 2)).sum(axis=(2, 3)).ravel(),
    }


def fit_poisson_rates(p_home: np.ndarray, p_away: np.ndarray,
                      p_over: np.ndarray) -> np.ndarray:
    """Least-squares (lambda_home, lambda_away) per match, by grid search."""
    grid = _grid_outcomes()
    n = len(LAMBDA_GRID)
    best = np.empty((len(p_home), 2))
    for start in range(0, len(p_home), 64):
        block = slice(start, start + 64)
        error = ((grid["home"][None] - p_home[block, None]) ** 2
                 + (grid["away"][None] - p_away[block, None]) ** 2
                 + (grid["over"][None] - p_over[block, None]) ** 2)
        index = error.argmin(axis=1)
        best[block, 0] = LAMBDA_GRID[index // n]
        best[block, 1] = LAMBDA_GRID[index % n]
    return best


def build_odds_table(seasons: Iterable[str], cache_dir: Path = ODDS_CACHE) -> pd.DataFrame:
    """One row per match: season, home, away, lambdas and win probabilities."""
    tables = []
    for season in seasons:
        raw = fetch_season_odds(season, cache_dir)
        if raw is None:
            continue
        columns = next((c for c in PRICE_COLUMNS if set(c).issubset(raw.columns)), None)
        if columns is None:
            LOGGER.warning("No usable price columns for %s; skipping.", season)
            continue
        raw = raw.dropna(subset=list(columns) + ["HomeTeam", "AwayTeam"])

        one_x_two = margin_free(raw[list(columns[:3])])
        over_under = margin_free(raw[list(columns[3:])])
        rates = fit_poisson_rates(one_x_two[:, 0], one_x_two[:, 2], over_under[:, 0])

        tables.append(pd.DataFrame({
            "season": season,
            "home_team": raw["HomeTeam"].map(lambda n: TEAM_NAMES.get(n, n)).to_numpy(),
            "away_team": raw["AwayTeam"].map(lambda n: TEAM_NAMES.get(n, n)).to_numpy(),
            "lambda_home": rates[:, 0],
            "lambda_away": rates[:, 1],
            "p_home": one_x_two[:, 0],
            "p_away": one_x_two[:, 2],
        }))
        LOGGER.info("Odds for %s: %d matches priced from %s.", season, len(raw), columns[0])

    if not tables:
        return pd.DataFrame(columns=["season", "home_team", "away_team", "lambda_home",
                                     "lambda_away", "p_home", "p_away"])
    return pd.concat(tables, ignore_index=True)


def attach_odds(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """
    Add the four odds_* features to player rows.

    Joined on (season, home club, away club), which identifies a Premier League
    fixture uniquely, so a rescheduled match still finds its prices.
    """
    frame = frame.copy()
    names = (frame.dropna(subset=["team_id"])
             .drop_duplicates(["season", "team_id"])
             .set_index(["season", "team_id"])["team"].to_dict())
    opponent = [names.get((s, o)) for s, o in zip(frame["season"], frame["opponent_team_id"])]
    home_flag = frame["was_home"].astype("boolean").fillna(False).to_numpy()

    home = np.where(home_flag, frame["team"], opponent)
    away = np.where(home_flag, opponent, frame["team"])
    keyed = table.set_index(["season", "home_team", "away_team"])
    keyed = keyed[~keyed.index.duplicated()]
    index = pd.MultiIndex.from_arrays([frame["season"].to_numpy(), home, away])
    matched = keyed.reindex(index)

    own = np.where(home_flag, matched["lambda_home"], matched["lambda_away"])
    opp = np.where(home_flag, matched["lambda_away"], matched["lambda_home"])
    frame["odds_team_goals"] = own
    frame["odds_team_conceded"] = opp
    frame["odds_cs_prob"] = np.exp(-opp)
    frame["odds_win_prob"] = np.where(home_flag, matched["p_home"], matched["p_away"])

    missing = int(pd.isna(own).sum())
    LOGGER.info("Odds attached to %d/%d rows%s.", len(frame) - missing, len(frame),
                f"; {missing} rows have no priced fixture" if missing else "")
    return frame
