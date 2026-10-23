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
further ahead by falling back on the FDR features. Live runs read the next
round's prices from football-data's upcoming-fixtures feed
(`fetch_upcoming_odds`); fixtures it does not list stay NaN.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fpl.odds")

ODDS_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
UPCOMING_URL = "https://www.football-data.co.uk/fixtures.csv"
ODDS_CACHE = Path(".fpl_cache/odds")

ODDS_FEATURES = ["odds_team_goals", "odds_team_conceded", "odds_cs_prob", "odds_win_prob"]
ODDS_PREFIX = "odds_"

# Every spelling we have met for a club, mapped to one key. The FPL archive,
# the live FPL API and football-data each name clubs differently ("Ipswich",
# "Ipswich Town"; "Man Utd", "Man United"), so both sides of a join go through
# team_key() rather than one side being renamed into the other.
TEAM_ALIASES: Dict[str, str] = {
    "man united": "man utd",
    "manchester united": "man utd",
    "manchester city": "man city",
    "tottenham": "spurs",
    "tottenham hotspur": "spurs",
    "sheffield united": "sheffield utd",
    "nottingham forest": "nott'm forest",
    "newcastle united": "newcastle",
    "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",
    "west ham united": "west ham",
    "brighton & hove albion": "brighton",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "luton town": "luton",
    "ipswich town": "ipswich",
    "coventry city": "coventry",
    "hull city": "hull",
    "norwich city": "norwich",
    "stoke city": "stoke",
    "swansea city": "swansea",
    "cardiff city": "cardiff",
    "west bromwich albion": "west brom",
    "afc bournemouth": "bournemouth",
}


def team_key(name) -> str:
    """Canonical lower-case key for a club name from any of our sources."""
    if not isinstance(name, str):
        return ""
    cleaned = " ".join(name.lower().split())
    return TEAM_ALIASES.get(cleaned, cleaned)


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


def season_finished(season: str, today: Optional[date] = None) -> bool:
    """A season is complete once its final summer has begun: '2025-26' after 1 July 2026."""
    end_year = int(season.split("-")[0][:2] + season.split("-")[1][-2:])
    return (today or date.today()) >= date(end_year, 7, 1)


def fetch_season_odds(season: str, cache_dir: Path = ODDS_CACHE,
                      max_age_hours: float = 12) -> Optional[pd.DataFrame]:
    """
    Download (or read the cached) football-data CSV for one season.

    A finished season's file never changes and is cached for good. The current
    season's grows every round, so its cached copy is refreshed once it is
    older than `max_age_hours`.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"E0_{season_code(season)}.csv"
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if season_finished(season) or age_hours < max_age_hours:
            return pd.read_csv(path, encoding="utf-8-sig")
        LOGGER.info("Cached odds for in-progress %s are %.0fh old; refreshing.",
                    season, age_hours)

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


def rates_from_prices(raw: pd.DataFrame, label: str) -> Optional[pd.DataFrame]:
    """One row per priced match: club names, Poisson rates, win probabilities."""
    columns = next((c for c in PRICE_COLUMNS if set(c).issubset(raw.columns)), None)
    if columns is None:
        LOGGER.warning("No usable price columns for %s; skipping.", label)
        return None
    raw = raw.dropna(subset=list(columns) + ["HomeTeam", "AwayTeam"])
    if raw.empty:
        return None

    one_x_two = margin_free(raw[list(columns[:3])])
    over_under = margin_free(raw[list(columns[3:])])
    rates = fit_poisson_rates(one_x_two[:, 0], one_x_two[:, 2], over_under[:, 0])
    LOGGER.info("Odds for %s: %d matches priced from %s.", label, len(raw), columns[0])
    return pd.DataFrame({
        "home_team": raw["HomeTeam"].to_numpy(),
        "away_team": raw["AwayTeam"].to_numpy(),
        "lambda_home": rates[:, 0],
        "lambda_away": rates[:, 1],
        "p_home": one_x_two[:, 0],
        "p_away": one_x_two[:, 2],
    })


def build_odds_table(seasons: Iterable[str], cache_dir: Path = ODDS_CACHE) -> pd.DataFrame:
    """One row per match: season, home, away, lambdas and win probabilities."""
    tables = []
    for season in seasons:
        raw = fetch_season_odds(season, cache_dir)
        table = rates_from_prices(raw, season) if raw is not None else None
        if table is not None:
            tables.append(table.assign(season=season))

    if not tables:
        return pd.DataFrame(columns=["season", "home_team", "away_team", "lambda_home",
                                     "lambda_away", "p_home", "p_away"])
    return pd.concat(tables, ignore_index=True)


def fetch_upcoming_odds(source: Optional[Path] = None) -> pd.DataFrame:
    """
    Prices for the Premier League fixtures football-data lists as upcoming.

    The feed covers roughly the next round and changes daily, so it is fetched
    fresh rather than cached. `source` reads a saved copy instead (offline runs,
    tests). Returns an empty table if the feed is unreachable: odds are an
    upgrade, and the models fall back on FDR without them.
    """
    if source is not None:
        raw = pd.read_csv(source, encoding="utf-8-sig")
    else:
        import requests
        try:
            response = requests.get(UPCOMING_URL, timeout=30,
                                    headers={"User-Agent": "fpl-ml-pipeline"})
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Could not fetch upcoming odds from %s: %s", UPCOMING_URL, exc)
            return pd.DataFrame()
        raw = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")

    if "Div" in raw.columns:
        raw = raw[raw["Div"] == "E0"]
    table = rates_from_prices(raw, "upcoming fixtures")
    return table if table is not None else pd.DataFrame()


def _odds_columns(frame: pd.DataFrame, home_flag: np.ndarray,
                  matched: pd.DataFrame) -> pd.DataFrame:
    """Orient matched match rates to each row's own club."""
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


def _keyed(table: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    keys = [table[c] for c in extra] + [table["home_team"].map(team_key),
                                         table["away_team"].map(team_key)]
    keyed = table.set_index(keys)
    return keyed[~keyed.index.duplicated()]


def attach_upcoming_odds(frame: pd.DataFrame, table: pd.DataFrame,
                         team_col: str = "team",
                         opponent_col: str = "opponent_name") -> pd.DataFrame:
    """
    Add odds_* to live fixture rows, matched on (home club, away club).

    A home/away pairing happens once per season, so the pair identifies the
    fixture whatever gameweek it sits in. Unpriced fixtures stay NaN, which the
    models were trained to handle.
    """
    frame = frame.copy()
    if table.empty:
        for column in ODDS_FEATURES:
            frame[column] = np.nan
        LOGGER.info("No upcoming odds available; scoring on the odds-free fallback.")
        return frame

    home_flag = frame["was_home"].astype("boolean").fillna(False).to_numpy()
    own, opp = frame[team_col].map(team_key), frame[opponent_col].map(team_key)
    index = pd.MultiIndex.from_arrays([np.where(home_flag, own, opp),
                                       np.where(home_flag, opp, own)])
    unknown = sorted(set(own) - set(map(team_key, table["home_team"]))
                     - set(map(team_key, table["away_team"])))
    if unknown:
        LOGGER.info("No upcoming prices for: %s.", ", ".join(unknown))
    return _odds_columns(frame, home_flag, _keyed(table).reindex(index))


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
    opponent = pd.Series([names.get((s, o)) for s, o in
                          zip(frame["season"], frame["opponent_team_id"])], index=frame.index)
    home_flag = frame["was_home"].astype("boolean").fillna(False).to_numpy()

    own, opp = frame["team"].map(team_key), opponent.map(team_key)
    index = pd.MultiIndex.from_arrays([frame["season"].to_numpy(),
                                       np.where(home_flag, own, opp),
                                       np.where(home_flag, opp, own)])
    return _odds_columns(frame, home_flag, _keyed(table, ["season"]).reindex(index))
