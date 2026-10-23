"""
FPL ML Pipeline — Phase 2: Feature Engineering
==============================================

Consumes `fpl_combined_history.csv` (Phase 1) and produces `fpl_ml_features.csv`:
leak-free rolling form features, ready for a gameweek-level points model.

Pipeline
--------
  1. Drop `xp`                      — post-match scrape, severe lookahead bias.
  2. Benchwarmers adjustments       — expected_assists x 1.4
                                      clean_sheet_probability = exp(-xGC),
                                      appearances only (NaN on 0 minutes)
     + fixture difficulty and bookmaker odds for the fixture itself
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
* Position medians are fitted over the whole file by default. That is mild
  lookahead: a 2023-24 row is filled from a median that saw 2025-26 data. For a
  clean backtest pass `--fit-seasons 2023-24 2024-25` so the held-out season is
  only ever filled from training data (the season simulator does this itself).
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
    # V2 — difficulty of the fixtures BEHIND the form above. Rolled (and so
    # shifted) like everything else, which lets the model discount xG piled up
    # against weak defences.
    "fdr_opp_defence",
    "fdr_opp_attack",
    "fdr_expected_team_goals",
]

DEFAULT_WINDOWS: List[int] = [3, 5, 10]

# Metrics that only mean something when the player was on the pitch. They are
# NaN on non-appearances and roll over a player's last N APPEARANCES rather than
# his last N gameweeks. clean_sheet_probability is the one that matters: a
# benchwarmer has xGC = 0, so exp(-xGC) would score every non-appearance as a
# certain clean sheet and turn the feature into an inverted minutes flag.
APPEARANCE_ONLY_METRICS: frozenset = frozenset({"clean_sheet_probability"})

# FPL Benchwarmers: FPL's raw xA understates fantasy assist value.
FANTASY_ASSIST_MULTIPLIER: float = 1.4

GROUP_KEY_CHOICES = {
    "name": "player_name",
    "element": "player_id",
    "season_element": "_season_element",
}


# --------------------------------------------------------------------------- #
# V2 — Fixture Difficulty Rating
# --------------------------------------------------------------------------- #

# Difficulty of THIS fixture. Known before the deadline, so NOT shifted at the
# player level (they are already causal — see compute_team_ratings). Phase 3 must
# add these to PRE_DEADLINE_FEATURES.
FDR_BASE_FEATURES: List[str] = [
    "fdr_team_attack",              # team's recent xG generated
    "fdr_team_defence",             # team's recent xG conceded
    "fdr_opp_attack",               # opponent's recent xG generated
    "fdr_opp_defence",              # opponent's recent xG conceded (high = weak defence)
    "fdr_expected_team_goals",      # Poisson rate: team attack x opponent defence
    "fdr_expected_team_conceded",   # Poisson rate: opponent attack x team defence
    "fdr_fixture_cs_prob",          # exp(-expected conceded) — opponent-aware clean sheet
    "fdr_overall_matchup",          # expected goals - expected conceded
    "fdr_opp_rating_known",         # 0 for promoted sides with no PL history
]

# Optional extras when --strength-source includes the FPL native ratings.
FDR_FPL_FEATURES: List[str] = [
    "fdr_fpl_opp_defence",
    "fdr_fpl_opp_attack",
    "fdr_fpl_opp_tier",
    "fdr_fpl_matchup",
]

# Team matches used per rolling rating, and the minimum before a rating counts.
TEAM_RATING_WINDOW: int = 10
TEAM_RATING_MIN_PERIODS: int = 3

# Reference scoring rate used to turn two ratings into a Poisson rate. A fixed
# constant, deliberately: a league mean computed from the data would peek at
# matches that have not been played yet.
LEAGUE_MEAN_XG: float = 1.35

VAASTAV_TEAMS_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master"
    "/data/{season}/teams.csv"
)


@dataclass
class FeatureConfig:
    input_path: Path = Path("fpl_combined_history.csv")
    output_path: Path = Path("fpl_ml_features.csv")
    medians_path: Optional[Path] = Path("fpl_position_medians.json")
    windows: Sequence[int] = tuple(DEFAULT_WINDOWS)
    metrics: Sequence[str] = tuple(ROLLING_METRICS)
    group_by: str = "name"
    median_groups: Sequence[str] = ("position",)
    fit_seasons: Optional[Sequence[str]] = None   # None -> fit medians on every row
    partial_windows: bool = False      # True -> min_periods=1 instead of window
    assist_multiplier: float = FANTASY_ASSIST_MULTIPLIER
    verify: bool = True
    # --- V2: fixture difficulty ---
    strength_source: str = "derived"   # derived | fpl | both | none
    team_rating_window: int = TEAM_RATING_WINDOW
    strength_csv: Optional[Path] = None
    fetch_strength: bool = True
    # --- bookmaker odds (fpl_odds.py) ---
    odds: bool = True
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

    # Files written before club names were canonicalised can hold both
    # "Ipswich" and "Ipswich Town"; fold them so team ratings see one club.
    if "team" in frame.columns:
        from fpl_ingest import canonical_club
        frame["team"] = frame["team"].map(canonical_club)
    frame = repair_club_labels(frame)

    LOGGER.info("Loaded %s: %d rows x %d columns", path, len(frame), frame.shape[1])
    return frame


def repair_club_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Give every row the club its player actually played for in that fixture.

    The live API's element-summary history carries no club, so Phase 1 stamps
    each row with the player's CURRENT club. A summer signing's early matches
    are therefore filed under his new club ("Man City vs Crystal Palace" in a
    gameweek City never played Palace), and each one becomes a phantom
    team-match in the ratings. Everyone on one side of a fixture shares the
    kickoff, opponent and venue, so the majority club on that side is the
    right one.
    """
    key = ["season", "kickoff_time", "opponent_team_id", "was_home"]
    if not {"team", *key}.issubset(frame.columns):
        return frame
    frame = frame.copy()
    columns = [c for c in ("team", "team_id") if c in frame.columns]
    for column in columns:
        majority = frame.groupby(key, dropna=False)[column].transform(
            lambda s: s.mode().iat[0] if s.notna().any() else s.iat[0])
        changed = majority.notna() & (majority != frame[column])
        if column == "team" and changed.any():
            LOGGER.info("Relabelled %d row(s) to the club the player was at for that "
                        "fixture (mid-window transfers): %s", int(changed.sum()),
                        ", ".join(sorted(frame.loc[changed, "player_name"].unique())[:6]))
        frame.loc[changed, column] = majority[changed]
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
        Only defined for appearances: with 0 minutes xGC is 0 and exp(0) = 1, a
        certain clean sheet for a player who never left the bench. Those rows
        are NaN instead (see APPEARANCE_ONLY_METRICS).
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
        minutes = pd.to_numeric(frame.get("minutes"), errors="coerce")
        frame["clean_sheet_probability"] = np.exp(-xgc).where(minutes > 0)
        observed = frame["clean_sheet_probability"].notna().sum()
        LOGGER.info("Built clean_sheet_probability = exp(-xGC) for %d/%d rows "
                    "(appearances only).", observed, len(frame))
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
# 2b. Fixture Difficulty Rating (V2)
# --------------------------------------------------------------------------- #

def build_team_name_map(frame: pd.DataFrame) -> Dict[tuple, str]:
    """
    (season, short_name) -> full team name.

    Phase 1 stores the player's own club as a full name (`team`) and the
    opponent as a short code (`opponent_team`) with no link between them, so we
    have to recover it. A club's own short code is the one code in the season
    that never appears in its own opponent list — you never play yourself.

    Cleaner long-term fix: have Phase 1 emit `team_id` next to `team`, then this
    function disappears.
    """
    mapping: Dict[tuple, str] = {}
    for season, block in frame.groupby("season"):
        shorts = set(block["opponent_team"].dropna().unique())
        for full_name, team_rows in block.groupby("team"):
            faced = set(team_rows["opponent_team"].dropna().unique())
            candidates = shorts - faced
            if len(candidates) == 1:
                mapping[(season, candidates.pop())] = full_name
            else:
                LOGGER.debug("[%s] Could not resolve short code for %s (%d candidates).",
                             season, full_name, len(candidates))

    expected = frame.groupby("season")["team"].nunique().sum()
    LOGGER.info("Resolved %d of %d (season, short_name) -> team mappings.",
                len(mapping), int(expected))
    return mapping


def build_team_match_table(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse player-gameweeks into one row per team per fixture.

    team_xg   = sum of the players' own expected_goals.
    team_xgc  = sum(xGC) / (sum(minutes) / 90). FPL charges every player on the
                pitch the full match xGC, so eleven players x 90 minutes sums to
                11 x the team figure; dividing by player-matches recovers it and
                handles substitutions correctly.
    """
    required = {"season", "team", "gameweek", "kickoff_time"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Cannot build team ratings; missing columns: {sorted(missing)}")

    work = frame.copy()
    for column in ("expected_goals", "expected_goals_conceded", "minutes"):
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    if "team_id" not in work.columns:
        work["team_id"] = pd.NA

    grouped = work.groupby(["season", "team", "team_id", "gameweek", "kickoff_time"],
                           dropna=False)
    table = grouped.agg(
        team_xg=("expected_goals", "sum"),
        xgc_sum=("expected_goals_conceded", "sum"),
        minutes_sum=("minutes", "sum"),
    ).reset_index()

    player_matches = (table["minutes_sum"] / 90.0).replace(0, np.nan)
    table["team_xgc"] = table["xgc_sum"] / player_matches

    # Actual goals as a fallback where xG is absent (older archives).
    if {"team_h_score", "team_a_score", "was_home"}.issubset(work.columns):
        goals = grouped.agg(
            home=("was_home", "first"),
            h=("team_h_score", "first"),
            a=("team_a_score", "first"),
        ).reset_index()
        home_flag = goals["home"].astype("boolean").fillna(False).to_numpy()
        table["team_goals_for"] = np.where(home_flag, goals["h"], goals["a"])
        table["team_goals_against"] = np.where(home_flag, goals["a"], goals["h"])
        table["team_xg"] = table["team_xg"].fillna(pd.to_numeric(table["team_goals_for"],
                                                                 errors="coerce"))
        table["team_xgc"] = table["team_xgc"].fillna(pd.to_numeric(table["team_goals_against"],
                                                                   errors="coerce"))

    table = table.drop(columns=["xgc_sum", "minutes_sum"])
    LOGGER.info("Team-match table: %d team-fixtures across %d club-seasons.",
                len(table), table.groupby(["season", "team"]).ngroups)
    return table


def compute_team_ratings(
    team_matches: pd.DataFrame, window: int = TEAM_RATING_WINDOW
) -> pd.DataFrame:
    """
    Causal rolling attack/defence ratings per team-fixture.

    Two properties that matter:

    * `.shift(1)` before `.rolling(...)`, exactly as at the player level, so a
      team's rating going into a match reflects only matches already played.
    * Grouped by TEAM, not by (team, season). A club's rating therefore carries
      across the summer, so GW1 inherits last season's closing form instead of
      cold-starting. That is causal — last season really did already happen —
      and it removes the main reason people reach for a static preseason rating.

    Promoted clubs have no Premier League history, so they get a league-average
    prior and `rating_known = 0` for the model to key off.
    """
    table = team_matches.sort_values(["team", "kickoff_time", "season", "gameweek"],
                                     kind="mergesort").reset_index(drop=True)
    grouped = table.groupby("team", sort=False)

    for source, target in (("team_xg", "attack_rating"), ("team_xgc", "defence_rating")):
        shifted = grouped[source].shift(1)
        table[target] = (
            shifted.groupby(table["team"], sort=False)
            .rolling(window=window, min_periods=TEAM_RATING_MIN_PERIODS)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    table["rating_known"] = table[["attack_rating", "defence_rating"]].notna().all(axis=1)
    unknown = int((~table["rating_known"]).sum())
    if unknown:
        LOGGER.info("%d team-fixture(s) have too little history for a rating "
                    "(season openers / promoted clubs) — using the league prior %.2f.",
                    unknown, LEAGUE_MEAN_XG)

    table["attack_rating"] = table["attack_rating"].fillna(LEAGUE_MEAN_XG)
    table["defence_rating"] = table["defence_rating"].fillna(LEAGUE_MEAN_XG)
    table["rating_known"] = table["rating_known"].astype("int8")

    return table[["season", "team", "team_id", "gameweek", "kickoff_time",
                  "attack_rating", "defence_rating", "rating_known"]]


def latest_team_ratings(
    team_matches: pd.DataFrame, window: int = TEAM_RATING_WINDOW
) -> pd.DataFrame:
    """
    Each team's rating going into its NEXT, unplayed fixture.

    The forward analogue of `compute_team_ratings`, and the same argument as
    Phase 5's `compute_forward_form`: the historical ratings are shifted, so a
    team's value on its last row excludes that very match. For a fixture that
    has not happened yet we want the window INCLUSIVE of everything played so
    far, which is simply the tail of the team's match history.

    Grouped by team NAME, so the rating carries across the summer. Team ids are
    re-issued each season and would break that.
    """
    table = team_matches.sort_values(["team", "kickoff_time", "season", "gameweek"],
                                     kind="mergesort")
    rows = []
    for team, group in table.groupby("team", sort=False):
        attack = group["team_xg"].dropna().tail(window)
        defence = group["team_xgc"].dropna().tail(window)
        known = len(attack) >= TEAM_RATING_MIN_PERIODS and len(defence) >= TEAM_RATING_MIN_PERIODS
        rows.append({
            "team": team,
            "attack_rating": float(attack.mean()) if len(attack) else LEAGUE_MEAN_XG,
            "defence_rating": float(defence.mean()) if len(defence) else LEAGUE_MEAN_XG,
            "rating_known": int(known),
            "matches_used": int(min(len(attack), len(defence))),
        })

    frame = pd.DataFrame(rows)
    LOGGER.info("Forward team ratings for %d club(s); %d with a full history.",
                len(frame), int(frame["rating_known"].sum()))
    return frame


def derive_fdr_matchups(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Turn four raw ratings into the Poisson matchup features.

    Shared by Phase 2 (historical rows) and Phase 5 (the upcoming fixture) so
    the arithmetic cannot drift between train and serve. If this function
    changes, both sides change together.
    """
    frame = frame.copy()
    for column, default in (("fdr_team_attack", LEAGUE_MEAN_XG),
                            ("fdr_team_defence", LEAGUE_MEAN_XG),
                            ("fdr_opp_attack", LEAGUE_MEAN_XG),
                            ("fdr_opp_defence", LEAGUE_MEAN_XG),
                            ("fdr_opp_rating_known", 0)):
        if column not in frame.columns:
            frame[column] = default
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

    # Multiplicative Poisson rates, the standard Dixon-Coles shape:
    #   E[goals] = own attack x opponent defence / league reference
    frame["fdr_expected_team_goals"] = (
        frame["fdr_team_attack"] * frame["fdr_opp_defence"] / LEAGUE_MEAN_XG)
    frame["fdr_expected_team_conceded"] = (
        frame["fdr_opp_attack"] * frame["fdr_team_defence"] / LEAGUE_MEAN_XG)

    # Opponent-aware clean sheet probability. Unlike the player-level
    # clean_sheet_probability (which is exp(-his own recent xGC) and knows
    # nothing about who he is facing), this is conditioned on the fixture.
    frame["fdr_fixture_cs_prob"] = np.exp(-frame["fdr_expected_team_conceded"])
    frame["fdr_overall_matchup"] = (
        frame["fdr_expected_team_goals"] - frame["fdr_expected_team_conceded"])
    return frame


def attach_derived_fdr(
    frame: pd.DataFrame, ratings: pd.DataFrame, name_map: Dict[tuple, str]
) -> pd.DataFrame:
    """
    Join each row's own-team and opponent ratings, then build the matchups.

    Opponent resolution prefers `opponent_team_id` against `team_id` — an exact
    id-to-id match within the season. It only falls back to the short-code name
    map for files produced before Phase 1 started emitting `team_id`.
    """
    frame = frame.copy()

    own = ratings.rename(columns={"attack_rating": "fdr_team_attack",
                                  "defence_rating": "fdr_team_defence"})
    frame = frame.merge(own.drop(columns=["rating_known", "team_id"]),
                        on=["season", "team", "gameweek", "kickoff_time"],
                        how="left", validate="many_to_one")

    use_ids = ("team_id" in frame.columns and "opponent_team_id" in frame.columns
               and frame["team_id"].notna().any() and frame["opponent_team_id"].notna().any())

    opponent = ratings.rename(columns={
        "attack_rating": "fdr_opp_attack",
        "defence_rating": "fdr_opp_defence",
        "rating_known": "fdr_opp_rating_known",
    })

    if use_ids:
        LOGGER.info("Resolving opponents by team_id (exact).")
        frame["_opp_key"] = pd.to_numeric(frame["opponent_team_id"], errors="coerce")
        opponent = opponent.rename(columns={"team_id": "_opp_key"}).drop(columns=["team"])
        opponent["_opp_key"] = pd.to_numeric(opponent["_opp_key"], errors="coerce")
    else:
        LOGGER.info("No team_id available — resolving opponents by derived short-code map.")
        frame["_opp_key"] = [
            name_map.get((season, short))
            for season, short in zip(frame["season"], frame["opponent_team"])
        ]
        opponent = opponent.rename(columns={"team": "_opp_key"}).drop(columns=["team_id"])

    # kickoff_time is in the key so a double gameweek resolves per fixture.
    frame = frame.merge(opponent, on=["season", "_opp_key", "gameweek", "kickoff_time"],
                        how="left", validate="many_to_one")

    unresolved = int(frame["fdr_opp_attack"].isna().sum())
    if unresolved:
        LOGGER.warning("%d row(s) could not be matched to an opponent rating — "
                       "falling back to the league prior.", unresolved)

    frame = derive_fdr_matchups(frame).drop(columns=["_opp_key"])
    LOGGER.info("Derived FDR attached: expected goals %.2f-%.2f, fixture CS prob %.2f-%.2f.",
                frame["fdr_expected_team_goals"].min(), frame["fdr_expected_team_goals"].max(),
                frame["fdr_fixture_cs_prob"].min(), frame["fdr_fixture_cs_prob"].max())
    return frame


# -- optional: FPL's own published strength ratings -------------------------- #

def load_fpl_strength_table(
    seasons: Iterable[str], config: FeatureConfig
) -> Optional[pd.DataFrame]:
    """
    Load FPL's published `strength_*` fields, from CSV or the archive.

    Read the warning in `attach_fpl_strength` before using these in training.
    """
    if config.strength_csv and Path(config.strength_csv).exists():
        table = pd.read_csv(config.strength_csv)
        LOGGER.info("Loaded FPL strength table from %s (%d rows).",
                    config.strength_csv, len(table))
        return table

    if not config.fetch_strength:
        LOGGER.warning("No strength CSV and --no-fetch-strength set — skipping FPL ratings.")
        return None

    try:
        from fpl_ingest import FPLClient, IngestConfig
    except ImportError:
        LOGGER.warning("fpl_ingest not importable — cannot fetch FPL strength ratings.")
        return None

    client = FPLClient(IngestConfig(cache_dir=Path(".fpl_cache")))
    frames = []
    for season in seasons:
        text = client.get_text(VAASTAV_TEAMS_URL.format(season=season),
                               cache_key=f"vaastav_teams_{season}")
        if not text:
            LOGGER.warning("[%s] teams.csv unavailable — no FPL strength for this season.",
                           season)
            continue
        import io
        block = pd.read_csv(io.StringIO(text))
        block["season"] = season
        frames.append(block)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def attach_fpl_strength(
    frame: pd.DataFrame, table: Optional[pd.DataFrame], name_map: Dict[tuple, str]
) -> pd.DataFrame:
    """
    Join FPL's own strength ratings, z-scored within season.

    WARNING — READ THIS BEFORE TRAINING ON THESE COLUMNS
    ----------------------------------------------------
    The archive's `teams.csv` is an END-OF-SEASON snapshot, not a preseason one.
    Nott'm Forest are rated 3 in the 2023-24 file and upgraded to 4 in the
    2024-25 file, after finishing 17th; Man Utd move the other way, matching
    their 15th-place finish. The ratings encode how each season turned out, so
    using them on a GW1 row leaks that season's outcome.

    They are fine at LIVE inference (bootstrap-static returns a current value)
    and fine for the season currently in progress. They are not fine as training
    features on completed seasons, which is why --strength-source defaults to
    `derived`. Absolute values also drift year to year (Arsenal 1350 in 2023-24
    vs 1300 in 2025-26), so everything is z-scored within season.
    """
    frame = frame.copy()
    if table is None or table.empty:
        for column in FDR_FPL_FEATURES:
            frame[column] = np.nan
        return frame

    LOGGER.warning(
        "Attaching FPL native strength ratings. The archive teams.csv is an "
        "end-of-season snapshot — these leak on completed seasons. See the "
        "docstring of attach_fpl_strength()."
    )

    strength_columns = ["strength_attack_home", "strength_attack_away",
                        "strength_defence_home", "strength_defence_away"]
    available = [c for c in strength_columns if c in table.columns]
    if not available or "short_name" not in table.columns:
        LOGGER.warning("Strength table lacks the expected columns — skipping.")
        for column in FDR_FPL_FEATURES:
            frame[column] = np.nan
        return frame

    # Z-score within season: raw values are not comparable across years.
    table = table.copy()
    for column in available:
        values = pd.to_numeric(table[column], errors="coerce")
        table[f"z_{column}"] = values.groupby(table["season"]).transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))

    lookup = table.set_index(["season", "short_name"])
    was_home = frame["was_home"].astype("boolean").fillna(False).to_numpy()

    def pull(column: str) -> np.ndarray:
        index = pd.MultiIndex.from_arrays([frame["season"], frame["opponent_team"]])
        return lookup[column].reindex(index).to_numpy()

    # The opponent's venue is the mirror of ours.
    frame["fdr_fpl_opp_defence"] = np.where(
        was_home, pull("z_strength_defence_away"), pull("z_strength_defence_home"))
    frame["fdr_fpl_opp_attack"] = np.where(
        was_home, pull("z_strength_attack_away"), pull("z_strength_attack_home"))
    frame["fdr_fpl_opp_tier"] = pull("strength") if "strength" in lookup.columns else np.nan
    frame["fdr_fpl_matchup"] = frame["fdr_fpl_opp_attack"] - frame["fdr_fpl_opp_defence"]

    filled = int(frame["fdr_fpl_opp_defence"].notna().sum())
    LOGGER.info("FPL strength attached to %d of %d rows.", filled, len(frame))
    return frame


def add_fixture_difficulty(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Entry point for the whole FDR block."""
    if config.strength_source == "none":
        LOGGER.info("Fixture difficulty disabled (--strength-source none).")
        for column in FDR_BASE_FEATURES + FDR_FPL_FEATURES:
            frame[column] = np.nan
        return frame

    if "opponent_team" not in frame.columns:
        LOGGER.warning("No `opponent_team` column — cannot build fixture difficulty.")
        for column in FDR_BASE_FEATURES + FDR_FPL_FEATURES:
            frame[column] = np.nan
        return frame

    name_map = build_team_name_map(frame)

    if config.strength_source in ("derived", "both"):
        team_matches = build_team_match_table(frame)
        ratings = compute_team_ratings(team_matches, config.team_rating_window)
        frame = attach_derived_fdr(frame, ratings, name_map)
    else:
        for column in FDR_BASE_FEATURES:
            frame[column] = np.nan

    if config.strength_source in ("fpl", "both"):
        seasons = sorted(frame["season"].dropna().unique())
        frame = attach_fpl_strength(frame, load_fpl_strength_table(seasons, config), name_map)
    else:
        for column in FDR_FPL_FEATURES:
            frame[column] = np.nan

    return frame


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


def _rolling_over_appearances(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Mean of the last `window` non-NaN values strictly before each row.

    Roll the observed values inclusively, carry the latest result forward over
    the gaps, then shift by one row. Row N therefore sees appearances up to and
    including N-1 and never N itself, exactly like the per-gameweek path.
    """
    observed = series.dropna()
    rolled = observed.rolling(window=window, min_periods=min_periods).mean()
    return rolled.reindex(series.index).ffill().shift(1)


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

    APPEARANCE_ONLY_METRICS roll over the last N non-NaN values instead, so a
    starter who missed one match keeps his clean-sheet form rather than going
    NaN for N gameweeks.
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

    per_gameweek = [m for m in available if m not in APPEARANCE_ONLY_METRICS]
    per_appearance = [m for m in available if m in APPEARANCE_ONLY_METRICS]

    created: List[str] = []
    for window in windows:
        min_periods = 1 if partial_windows else window
        if per_gameweek:
            rolled = (
                shifted_groups[per_gameweek]
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
                .sort_index()
            )
            for metric in per_gameweek:
                frame[rolling_column_name(metric, window)] = rolled[metric]

        for metric in per_appearance:
            frame[rolling_column_name(metric, window)] = grouped[metric].transform(
                _rolling_over_appearances, window=window, min_periods=min_periods)

        for metric in available:
            created.append(rolling_column_name(metric, window))
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
                history = prior[metric]
                if metric in APPEARANCE_ONLY_METRICS:
                    history = history.dropna()
                expected = history.tail(window).mean()
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

    # 2b. Fixture difficulty. Must precede the rolling step so that the FDR
    #     metrics in ROLLING_METRICS exist by the time windows are built.
    frame = add_fixture_difficulty(frame, config)

    # 2c. Bookmaker odds for THIS fixture. Pre-deadline knowns, like the base
    #     FDR columns, so they are attached as-is and never rolled.
    if config.odds:
        from fpl_odds import attach_odds, build_odds_table
        table = build_odds_table(sorted(frame["season"].dropna().unique()))
        frame = attach_odds(frame, table)

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
    fit_rows = frame
    if config.fit_seasons:
        fit_rows = frame[frame["season"].isin(config.fit_seasons)]
        if fit_rows.empty:
            raise RuntimeError(f"--fit-seasons {list(config.fit_seasons)} matched no rows.")
        LOGGER.info("Fitting imputation medians on %s only (%d rows).",
                    list(config.fit_seasons), len(fit_rows))
    frame, report = imputer.fit(fit_rows, created).transform(frame)
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
    parser.add_argument("--fit-seasons", nargs="+", default=None,
                        help="Fit imputation medians on these seasons only, so a "
                             "held-out season is never filled from its own data")
    parser.add_argument("--assist-multiplier", type=float, default=FANTASY_ASSIST_MULTIPLIER)
    parser.add_argument("--partial-windows", action="store_true",
                        help="min_periods=1 instead of the full window")
    parser.add_argument("--strength-source", default="derived",
                        choices=["derived", "fpl", "both", "none"],
                        help="Fixture difficulty source. `derived` (default) builds causal "
                             "team ratings from our own xG data; `fpl` uses the published "
                             "strength_* fields, which LEAK on completed seasons")
    parser.add_argument("--team-rating-window", type=int, default=TEAM_RATING_WINDOW,
                        help=f"Team matches per rolling rating (default: {TEAM_RATING_WINDOW})")
    parser.add_argument("--strength-csv", default=None,
                        help="Pre-fetched FPL strength table (season,short_name,strength_*)")
    parser.add_argument("--no-fetch-strength", action="store_true",
                        help="Never hit the network for the FPL strength table")
    parser.add_argument("--no-odds", action="store_true",
                        help="Skip the bookmaker odds features (fpl_odds.py)")
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
        fit_seasons=tuple(args.fit_seasons) if args.fit_seasons else None,
        partial_windows=args.partial_windows,
        assist_multiplier=args.assist_multiplier,
        verify=not args.no_verify,
        strength_source=args.strength_source,
        team_rating_window=args.team_rating_window,
        strength_csv=Path(args.strength_csv) if args.strength_csv else None,
        fetch_strength=not args.no_fetch_strength,
        odds=not args.no_odds,
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
