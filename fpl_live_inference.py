"""
FPL ML Pipeline — Phase 5: Live Inference & Forward Projection
==============================================================

Turns the backtest into something you can actually transfer on:

  1. Pull live `bootstrap-static` -> current `now_cost` and `status` per player.
     Keep only status == 'a' (available). No more picking long-term absentees.
  2. Pull `/fixtures/?future=1` -> identify the next gameweek and each player's
     upcoming fixture(s): `was_home`, `opponent_team` (id -> name via team_lookup).
  3. Build a genuine "GW N+1" feature row per player-fixture by recomputing their
     rolling form over their most recent gameweeks.
  4. Score with the Phase 3 position models.
  5. Hand the scored frame straight to the Phase 4 `SquadOptimizer`.

Usage
-----
    python fpl_live_inference.py
    python fpl_live_inference.py --budget 100 --bench-weight 0.15
    python fpl_live_inference.py --allow-status a d --min-chance 75
    python fpl_live_inference.py --gameweek 12          # force a specific GW
    python fpl_live_inference.py --reuse-last-row       # see note below


Why the rolling features are RECOMPUTED, not copied
---------------------------------------------------
The obvious move is to reuse the rolling features on each player's most recent
Phase 2 row. That is subtly wrong. Phase 2 shifts every window by one, so the
values on the row for GW N summarise GW N-1 and older — they deliberately
exclude GW N itself. Copying them forward to predict GW N+1 therefore discards
the single most recent (and most informative) gameweek, and shows the model a
window one step staler than the one it was trained on.

The correct forward row averages the player's last k gameweeks *inclusive* of
GW N. That is precisely what Phase 2's shifted window would have contained had a
row for GW N+1 existed. `--reuse-last-row` restores the naive behaviour if you
want to measure the difference.

Cold starts are filled with the medians Phase 2 fitted (`fpl_position_medians.json`),
not medians refitted on live data — same numbers at train and serve time.


Known gaps (deliberate, not oversights)
---------------------------------------
* `clean_sheet_probability` is still derived from the player's own recent xGC,
  so it encodes his team's recent defensive form but knows nothing about the
  specific opponent he is about to face. Fixture difficulty (`team_h_difficulty`
  / `team_a_difficulty`) is fetched and reported, but is NOT a model feature —
  Phase 3 never trained on it, and quietly appending an untrained column would
  be worse than leaving it out. That is the next modelling upgrade.
* Single-period only. A multi-period transfer optimiser needs projections for
  GW N+1..N+k plus transfer/hit accounting; this produces the N+1 layer it
  would sit on.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Phase 1 — HTTP client with retries/caching, and the bootstrap decoders.
from fpl_ingest import (
    BASE_URL,
    BOOTSTRAP_URL,
    FPLClient,
    IngestConfig,
    build_position_lookup,
    build_team_lookup,
    normalise_position,
)

# Phase 4 — reuse the solver rather than reimplementing the constraints.
from fpl_optimizer import (
    MAX_PER_CLUB,
    POSITIONS,
    OptimizerConfig,
    SquadOptimizer,
    aggregate_double_gameweeks,
    check_feasibility,
    formation_string,
    load_models,
    prepare_matrix,
)

LOGGER = logging.getLogger("fpl.live")

FIXTURES_URL = f"{BASE_URL}/fixtures/?future=1"
ALL_FIXTURES_URL = f"{BASE_URL}/fixtures/"

ROLLING_PATTERN = re.compile(r"^(?P<metric>.+)_roll_(?P<window>\d+)$")

# FPL availability codes.
STATUS_MEANINGS = {
    "a": "available", "d": "doubtful", "i": "injured",
    "s": "suspended", "u": "unavailable", "n": "not in squad",
}


@dataclass
class LiveConfig:
    features_path: Path = Path("fpl_ml_features.csv")
    model_dir: Path = Path("models")
    medians_path: Path = Path("fpl_position_medians.json")
    budget: float = 100.0
    gameweek: Optional[int] = None        # default: the API's "next" event
    allow_status: Sequence[str] = ("a",)
    min_chance: int = 0                   # chance_of_playing_next_round threshold
    bench_weight: float = 0.0
    reuse_last_row: bool = False
    partial_windows: bool = False
    exclude: Sequence[str] = ()
    cache_dir: Optional[Path] = None      # None -> always hit the live API
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Live availability
# --------------------------------------------------------------------------- #

def make_client(config: LiveConfig) -> FPLClient:
    """Phase 1's client: retry/backoff on 429+5xx, optional on-disk cache."""
    return FPLClient(IngestConfig(cache_dir=config.cache_dir, timeout=20, max_retries=4))


def fetch_future_fixtures(client: FPLClient) -> List[Dict[str, Any]]:
    """`/fixtures/?future=1`, falling back to filtering the full fixture list."""
    payload = client.get_json(FIXTURES_URL, cache_key="fixtures_future")
    if isinstance(payload, list) and payload:
        LOGGER.info("Fetched %d future fixture(s).", len(payload))
        return payload

    LOGGER.warning("future=1 returned nothing — falling back to the full fixture list.")
    payload = client.get_json(ALL_FIXTURES_URL, cache_key="fixtures_all")
    if not isinstance(payload, list):
        raise RuntimeError("Could not fetch fixtures from the FPL API.")
    upcoming = [f for f in payload if not f.get("finished") and f.get("event") is not None]
    LOGGER.info("Derived %d unplayed fixture(s) from the full list.", len(upcoming))
    return upcoming


def resolve_next_gameweek(
    bootstrap: Dict[str, Any], fixtures: List[Dict[str, Any]], override: Optional[int]
) -> int:
    """Prefer bootstrap's `is_next` event; fall back to the earliest future fixture."""
    if override is not None:
        LOGGER.info("Gameweek overridden to GW%d.", override)
        return int(override)

    for event in bootstrap.get("events", []):
        if event.get("is_next"):
            LOGGER.info("Next gameweek per bootstrap: GW%s (deadline %s).",
                        event.get("id"), event.get("deadline_time"))
            return int(event["id"])

    events = [f["event"] for f in fixtures if f.get("event") is not None]
    if not events:
        raise RuntimeError("No upcoming gameweek could be identified — is the season over?")
    gameweek = int(min(events))
    LOGGER.info("No `is_next` flag; using earliest future fixture: GW%d.", gameweek)
    return gameweek


def build_live_players(
    bootstrap: Dict[str, Any],
    team_lookup: Dict[int, Dict[str, str]],
    position_lookup: Dict[int, str],
) -> pd.DataFrame:
    """One row per FPL player with live price, status and club."""
    rows = []
    for element in bootstrap["elements"]:
        team = team_lookup.get(element.get("team"), {})
        rows.append({
            "element_id": element["id"],
            "player_name": f"{element.get('first_name', '')} "
                           f"{element.get('second_name', '')}".strip(),
            "web_name": element.get("web_name", ""),
            "team_id": element.get("team"),
            "team": team.get("name", "Unknown"),
            "team_short": team.get("short_name", "UNK"),
            "position": position_lookup.get(element.get("element_type"), "UNK"),
            "now_cost": element.get("now_cost", 0) / 10.0,
            "status": element.get("status", "u"),
            "chance_of_playing": element.get("chance_of_playing_next_round"),
            "news": (element.get("news") or "").strip(),
        })
    frame = pd.DataFrame(rows)
    LOGGER.info("Live pool: %d players across %d clubs.", len(frame), frame["team"].nunique())
    return frame


def filter_available(frame: pd.DataFrame, config: LiveConfig) -> pd.DataFrame:
    """
    Keep only players the API says can play.

    Default is status == 'a' exactly, per spec. Worth knowing what that costs:
    'd' (doubtful) covers knocks with a 75% chance of playing, and those players
    are often the best value picks of the week. `--allow-status a d --min-chance 75`
    is the pragmatic setting.
    """
    before = len(frame)
    allowed = {s.strip().lower() for s in config.allow_status}

    counts = frame["status"].value_counts().to_dict()
    LOGGER.info("Status breakdown: %s", ", ".join(
        f"{code}={n} ({STATUS_MEANINGS.get(code, '?')})" for code, n in sorted(counts.items())))

    frame = frame[frame["status"].isin(allowed)].copy()
    LOGGER.info("Availability filter (status in %s): %d of %d players kept.",
                sorted(allowed), len(frame), before)

    if config.min_chance > 0:
        chance = pd.to_numeric(frame["chance_of_playing"], errors="coerce")
        # Null means "no doubt reported" for an available player — treat as 100.
        keep = chance.isna() | (chance >= config.min_chance)
        dropped = int((~keep).sum())
        if dropped:
            LOGGER.info("chance_of_playing >= %d%%: dropped a further %d player(s).",
                        config.min_chance, dropped)
        frame = frame[keep].copy()

    if config.exclude:
        excluded = {n.strip().lower() for n in config.exclude if n.strip()}
        mask = (frame["player_name"].str.lower().isin(excluded)
                | frame["web_name"].str.lower().isin(excluded))
        if mask.any():
            LOGGER.info("Manually excluded %d player(s).", int(mask.sum()))
        frame = frame[~mask].copy()

    if frame.empty:
        raise RuntimeError("No players survived the availability filter.")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Forward fixture construction
# --------------------------------------------------------------------------- #

def build_fixture_rows(
    players: pd.DataFrame,
    fixtures: List[Dict[str, Any]],
    team_lookup: Dict[int, Dict[str, str]],
    gameweek: int,
) -> pd.DataFrame:
    """
    Expand players x their fixtures in the target gameweek.

    Produces one row per (player, fixture): a blank gameweek yields no rows for
    that player (correctly excluding him), a double yields two rows which are
    scored separately and summed later. `was_home` and the opponent are read off
    the fixture, not guessed.
    """
    relevant = [f for f in fixtures if f.get("event") == gameweek]
    if not relevant:
        raise RuntimeError(f"No fixtures found for GW{gameweek}.")

    LOGGER.info("GW%d has %d fixture(s).", gameweek, len(relevant))

    # team_id -> list of (was_home, opponent_id, difficulty, kickoff, fixture_id)
    by_team: Dict[int, List[Dict[str, Any]]] = {}
    for fixture in relevant:
        home, away = fixture.get("team_h"), fixture.get("team_a")
        if home is None or away is None:
            continue
        by_team.setdefault(home, []).append({
            "fixture_id": fixture.get("id"), "was_home": True, "opponent_team_id": away,
            "difficulty": fixture.get("team_h_difficulty"),
            "kickoff_time": fixture.get("kickoff_time"),
        })
        by_team.setdefault(away, []).append({
            "fixture_id": fixture.get("id"), "was_home": False, "opponent_team_id": home,
            "difficulty": fixture.get("team_a_difficulty"),
            "kickoff_time": fixture.get("kickoff_time"),
        })

    rows: List[Dict[str, Any]] = []
    blanks = 0
    for player in players.to_dict(orient="records"):
        team_fixtures = by_team.get(player["team_id"], [])
        if not team_fixtures:
            blanks += 1
            continue
        for fixture in team_fixtures:
            opponent = team_lookup.get(fixture["opponent_team_id"], {})
            rows.append({
                **player,
                **fixture,
                "gameweek": gameweek,
                "opponent_team": opponent.get("short_name", "UNK"),
                "opponent_name": opponent.get("name", "Unknown"),
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No player has a fixture in GW{gameweek}.")

    doubles = frame["element_id"].value_counts()
    n_double = int((doubles > 1).sum())
    LOGGER.info("Built %d player-fixture row(s): %d blanking, %d on a double gameweek.",
                len(frame), blanks, n_double)
    return frame


# --------------------------------------------------------------------------- #
# 3. Forward rolling features
# --------------------------------------------------------------------------- #

def normalise_name(value: Any) -> str:
    """Accent- and case-insensitive key for joining live players to history."""
    if not isinstance(value, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", "", stripped.lower()).strip()


def required_rolling_features(bundles: Dict[str, Dict[str, Any]]) -> Dict[str, List[int]]:
    """Parse `{metric}_roll_{w}` out of the saved models — never hardcode the list."""
    needed: Dict[str, set] = {}
    for bundle in bundles.values():
        for feature in bundle.get("features", []):
            match = ROLLING_PATTERN.match(feature)
            if match:
                needed.setdefault(match.group("metric"), set()).add(int(match.group("window")))
    resolved = {metric: sorted(windows) for metric, windows in needed.items()}
    LOGGER.info("Models require rolling features: %s",
                ", ".join(f"{m}({','.join(map(str, w))})" for m, w in sorted(resolved.items())))
    return resolved


def load_history(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fpl_features.py (Phase 2) first.")
    frame = pd.read_csv(path, low_memory=False)
    if "player_name" not in frame.columns:
        raise RuntimeError(f"{path} has no `player_name` column to join on.")
    frame["_name_key"] = frame["player_name"].map(normalise_name)
    LOGGER.info("Loaded history: %d rows, %d distinct players.",
                len(frame), frame["_name_key"].nunique())
    return frame


def sort_history(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["_name_key", "season", "gameweek"]
    if "kickoff_time" in frame.columns:
        frame = frame.copy()
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)
        keys.append("kickoff_time")
    return frame.sort_values(keys, kind="mergesort")


def compute_forward_form(
    history: pd.DataFrame,
    needed: Dict[str, List[int]],
    partial_windows: bool = False,
) -> pd.DataFrame:
    """
    Per player, average the last w gameweeks INCLUSIVE of the most recent one.

    This is the forward analogue of Phase 2's shifted window: for a hypothetical
    row at GW N+1, `.shift(1).rolling(w)` would cover exactly GW N back to
    N-w+1 — which is the tail of the player's actual history.
    """
    history = sort_history(history)
    grouped = history.groupby("_name_key", sort=False)

    records: Dict[str, Dict[str, Any]] = {}
    for name_key, group in grouped:
        row: Dict[str, Any] = {"_name_key": name_key, "games_prior": len(group)}
        for metric, windows in needed.items():
            if metric not in group.columns:
                for window in windows:
                    row[f"{metric}_roll_{window}"] = np.nan
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            for window in windows:
                tail = values.tail(window)
                enough = len(tail) >= (1 if partial_windows else window)
                row[f"{metric}_roll_{window}"] = float(tail.mean()) if enough and len(tail) \
                    else np.nan
        records[name_key] = row

    frame = pd.DataFrame(list(records.values()))
    max_window = max((w for windows in needed.values() for w in windows), default=10)
    frame["is_cold_start"] = (frame["games_prior"] < max_window).astype("int8")
    LOGGER.info("Recomputed forward form for %d player(s) (windows inclusive of the latest GW).",
                len(frame))
    return frame


def reuse_last_row_form(history: pd.DataFrame, needed: Dict[str, List[int]]) -> pd.DataFrame:
    """Naive alternative: copy the most recent row's (shifted, hence staler) features."""
    history = sort_history(history)
    columns = [f"{m}_roll_{w}" for m, windows in needed.items() for w in windows]
    available = [c for c in columns if c in history.columns]
    keep = ["_name_key", *available]
    for extra in ("games_prior", "is_cold_start"):
        if extra in history.columns:
            keep.append(extra)

    frame = history.groupby("_name_key", sort=False).tail(1)[keep].copy()
    for missing in set(columns) - set(available):
        frame[missing] = np.nan
    LOGGER.warning("Using --reuse-last-row: features exclude the most recent gameweek.")
    return frame.reset_index(drop=True)


def impute_cold_starts(
    frame: pd.DataFrame, columns: Sequence[str], config: LiveConfig, history: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill NaN form with Phase 2's *fitted* medians — same numbers at train and serve.

    Refitting medians on live data would mean the model sees a slightly different
    imputation distribution than it trained against. That is the whole reason
    Phase 2's imputer is a fit/transform class with JSON persistence.

    Schema reconciliation
    ---------------------
    The imputer's column set and this frame's column set are built from different
    places and legitimately differ in BOTH directions:

      imputer has extra   Phase 2 fits medians for every rolling metric it built.
                          Phase 3 then drops zero-variance features, so a metric
                          that was constant in training (notably
                          `defensive_contribution`, which Phase 2 fabricated from
                          a single global median for the two seasons where the
                          raw stat did not exist) never reaches the saved models,
                          and Phase 5 never computes it. `transform()` indexes
                          `self.columns_` directly, so the absent columns raise
                          KeyError.

      frame has extra     A model could require a rolling feature the imputer was
                          never fitted on. Those cannot be filled and would reach
                          the model as NaN, so warn rather than fail silently.

    Fix: pad the missing columns with NaN before `transform()`, then drop the
    padding again so the feature matrix contains only what the models asked for.
    """
    columns = [c for c in columns if c in frame.columns]
    missing_before = int(frame[columns].isna().sum().sum()) if columns else 0
    if missing_before == 0:
        LOGGER.info("No NaN form values — cold-start imputation not needed.")
        return frame

    imputer = None
    path = Path(config.medians_path)
    if path.exists():
        try:
            from fpl_features import PositionMedianImputer
            imputer = PositionMedianImputer.load(path)
            LOGGER.info("Loaded Phase 2 medians from %s.", path)
        except Exception as exc:
            LOGGER.warning("Could not load %s (%s) — refitting from history.", path, exc)

    if imputer is None:
        try:
            from fpl_features import PositionMedianImputer
        except ImportError as exc:
            raise RuntimeError("fpl_features.py must be importable for imputation.") from exc
        LOGGER.warning("Refitting medians on historical data (train/serve parity not guaranteed).")
        imputer = PositionMedianImputer(("position",)).fit(history, columns)

    frame = reconcile_imputer_schema(frame, imputer, columns)

    expected = list(getattr(imputer, "columns_", []) or [])
    padded = [c for c in expected if c not in columns]

    frame, report = imputer.transform(frame)

    # Drop the padding: these columns exist only to satisfy transform(), and no
    # model asked for them. Leaving them would silently widen the matrix.
    if padded:
        frame = frame.drop(columns=[c for c in padded if c in frame.columns])

    filled = int(report.loc[report.index.isin(columns), "imputed"].sum())
    LOGGER.info("Cold-start imputation: filled %d of %d NaN cell(s) across %d column(s).",
                filled, missing_before, len(columns))

    still_nan = int(frame[columns].isna().sum().sum())
    if still_nan:
        LOGGER.warning("%d NaN cell(s) remain after imputation — they reach the model as NaN.",
                       still_nan)
    return frame


def reconcile_imputer_schema(
    frame: pd.DataFrame, imputer: Any, columns: Sequence[str]
) -> pd.DataFrame:
    """
    Make `frame` safe to pass to `imputer.transform()`.

    `PositionMedianImputer.transform` does `frame[self.columns_]`, which raises
    KeyError if any fitted column is absent. Pad those with NaN. Also flag the
    reverse mismatch, where a model needs a column the imputer cannot fill.
    """
    expected = list(getattr(imputer, "columns_", []) or [])
    if not expected:
        LOGGER.warning("Imputer reports no fitted columns — skipping reconciliation.")
        return frame

    frame = frame.copy()

    padded = [c for c in expected if c not in frame.columns]
    if padded:
        LOGGER.info(
            "Padding %d column(s) the imputer expects but the models never asked for "
            "(they are dropped again straight after): %s",
            len(padded), ", ".join(sorted(padded)),
        )
        for column in padded:
            frame[column] = np.nan

    unfillable = [c for c in columns if c not in expected]
    if unfillable:
        LOGGER.warning(
            "%d model feature(s) have no fitted median and cannot be imputed: %s. "
            "Re-run fpl_features.py to refresh %s if this is unexpected.",
            len(unfillable), ", ".join(sorted(unfillable)), "fpl_position_medians.json",
        )

    # `transform` groups by these; a NaN group would fall through to the global median.
    for group_column in getattr(imputer, "group_cols", []):
        if group_column not in frame.columns:
            raise RuntimeError(
                f"Imputer groups medians by `{group_column}`, which is missing from the "
                f"live feature matrix."
            )

    return frame


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble_feature_matrix(
    fixture_rows: pd.DataFrame,
    history: pd.DataFrame,
    needed: Dict[str, List[int]],
    config: LiveConfig,
) -> pd.DataFrame:
    """Join live fixture context to recomputed form, then fill any gaps."""
    fixture_rows = fixture_rows.copy()
    fixture_rows["_name_key"] = fixture_rows["player_name"].map(normalise_name)

    form = reuse_last_row_form(history, needed) if config.reuse_last_row \
        else compute_forward_form(history, needed, config.partial_windows)

    merged = fixture_rows.merge(form, on="_name_key", how="left", validate="many_to_one")

    unmatched = merged["games_prior"].isna()
    if unmatched.any():
        names = merged.loc[unmatched, "web_name"].unique()[:8]
        LOGGER.warning(
            "%d player-row(s) have no history (new signings / name mismatch): %s%s",
            int(unmatched.sum()), ", ".join(map(str, names)),
            " ..." if len(names) == 8 else "",
        )
        merged["games_prior"] = merged["games_prior"].fillna(0)
        merged["is_cold_start"] = merged["is_cold_start"].fillna(1).astype("int8")

    # Live price is authoritative for both the model feature and the budget.
    merged["gw_cost"] = merged["now_cost"]
    merged["position"] = merged["position"].map(normalise_position)
    merged["was_home"] = merged["was_home"].astype(bool)

    rolling_columns = [f"{m}_roll_{w}" for m, windows in needed.items() for w in windows]
    merged = impute_cold_starts(merged, rolling_columns, config, history)

    LOGGER.info("Feature matrix ready: %d rows x %d columns.", len(merged), merged.shape[1])
    return merged


def predict_points(frame: pd.DataFrame, bundles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Score each position with its own Phase 3 model."""
    frame = frame.copy()
    for position in POSITIONS:
        frame[f"pos_{position}"] = (frame["position"] == position).astype("int8")
    frame["predicted_points"] = np.nan

    for position, bundle in bundles.items():
        mask = frame["position"] == position
        if not mask.any():
            continue
        matrix = prepare_matrix(frame.loc[mask], bundle["features"])
        try:
            predictions = bundle["model"].predict(matrix)
        except Exception as exc:
            LOGGER.warning("Prediction failed for %s: %s", position, exc)
            continue
        frame.loc[mask, "predicted_points"] = np.clip(predictions, 0.0, None)
        LOGGER.info("  %s: %d rows scored (mean %.2f, max %.2f)",
                    position, int(mask.sum()),
                    float(np.mean(predictions)), float(np.max(predictions)))

    unscored = frame["predicted_points"].isna()
    if unscored.any():
        LOGGER.warning("Dropping %d row(s) with no model for their position.",
                       int(unscored.sum()))
        frame = frame[~unscored]
    return frame.reset_index(drop=True)


def collapse_to_players(frame: pd.DataFrame) -> pd.DataFrame:
    """
    One decision variable per player.

    Doubles are summed (he genuinely plays twice); the opponent column becomes
    "ARS + CHE" so the report stays honest about why his projection is high.
    """
    frame = frame.copy()
    frame["cost"] = frame["now_cost"]

    fixture_labels = (
        frame.assign(_label=frame["opponent_team"].astype(str)
                     + np.where(frame["was_home"], " (H)", " (A)"))
        .groupby("element_id")["_label"].apply(lambda s: " + ".join(s))
    )

    collapsed = aggregate_double_gameweeks(
        frame.rename(columns={"element_id": "player_id"})
    )
    collapsed["fixtures"] = collapsed["player_id"].map(fixture_labels)
    return collapsed


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _rows(block: pd.DataFrame) -> str:
    lines = []
    for _, row in block.iterrows():
        marker = "(C)" if row["is_captain"] else ("(V)" if row.get("is_vice") else "")
        name = f"{row['player_name']} {marker}".strip()
        lines.append(
            f"  {row['position']:<4} {name:<24.24} {str(row['team']):<13.13} "
            f"{str(row.get('fixtures', '')):<18.18} {row['cost']:>5.1f} {row['predicted_points']:>7.2f}"
        )
    return "\n".join(lines)


def print_live_report(squad: pd.DataFrame, gameweek: int, config: LiveConfig,
                      deadline: Optional[str] = None) -> None:
    starters = squad[squad["is_starter"]]
    bench = squad[~squad["is_starter"]]
    captain = squad[squad["is_captain"]].iloc[0]

    xi_points = float(starters["predicted_points"].sum())
    total = xi_points + float(captain["predicted_points"])
    header = f"{'POS':<4} {'PLAYER':<24} {'TEAM':<13} {'FIXTURE':<18} {'COST':>5} {'xPTS':>7}"

    print("\n" + "=" * 82)
    print(f"OPTIMAL SQUAD — UPCOMING GAMEWEEK {gameweek}".center(82))
    if deadline:
        print(f"deadline {deadline}".center(82))
    print("=" * 82)

    print(f"\nSTARTING XI  ({formation_string(starters)})")
    print("  " + header)
    print("  " + "-" * 78)
    print(_rows(starters))
    print("  " + "-" * 78)
    print(f"  {'':<4} {'XI subtotal':<24} {'':<13} {'':<18} "
          f"{starters['cost'].sum():>5.1f} {xi_points:>7.2f}")

    print("\nBENCH")
    print("  " + header)
    print("  " + "-" * 78)
    print(_rows(bench))
    print("  " + "-" * 78)
    print(f"  {'':<4} {'Bench subtotal':<24} {'':<13} {'':<18} "
          f"{bench['cost'].sum():>5.1f} {bench['predicted_points'].sum():>7.2f}")

    print("\n" + "-" * 82)
    print(f"  Captain            {captain['player_name']} "
          f"(+{captain['predicted_points']:.2f} doubled)")
    vice = squad[squad["is_vice"]]
    if not vice.empty:
        print(f"  Vice-captain       {vice.iloc[0]['player_name']}")
    print(f"  Squad cost         {squad['cost'].sum():.1f} / {config.budget:.1f} "
          f"  (remaining {config.budget - squad['cost'].sum():.1f})")
    print(f"  Clubs used         {squad['team'].nunique()} "
          f"(max {squad['team'].value_counts().max()} of {MAX_PER_CLUB} allowed)")
    print(f"  PROJECTED POINTS   {total:.2f}")
    print("\n  All 15 have status 'a' as of this run. Re-run after the press conferences —")
    print("  FPL flags injuries right up to the deadline.")
    print("-" * 82)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_live_inference(config: LiveConfig) -> Dict[str, Any]:
    client = make_client(config)

    bootstrap = client.fetch_bootstrap()
    team_lookup = build_team_lookup(bootstrap["teams"])
    position_lookup = build_position_lookup(bootstrap["element_types"])

    fixtures = fetch_future_fixtures(client)
    gameweek = resolve_next_gameweek(bootstrap, fixtures, config.gameweek)
    deadline = next((e.get("deadline_time") for e in bootstrap.get("events", [])
                     if e.get("id") == gameweek), None)

    players = build_live_players(bootstrap, team_lookup, position_lookup)
    players = filter_available(players, config)

    fixture_rows = build_fixture_rows(players, fixtures, team_lookup, gameweek)

    bundles, _ = load_models(config.model_dir)
    needed = required_rolling_features(bundles)

    history = load_history(config.features_path)
    matrix = assemble_feature_matrix(fixture_rows, history, needed, config)

    scored = predict_points(matrix, bundles)
    candidates = collapse_to_players(scored)

    LOGGER.info("Candidate pool: %d available players across %d clubs.",
                len(candidates), candidates["team"].nunique())

    optimizer_config = OptimizerConfig(budget=config.budget, bench_weight=config.bench_weight)
    check_feasibility(candidates, config.budget)
    optimizer = SquadOptimizer(candidates, optimizer_config)
    optimizer.build()
    optimizer.solve()
    squad = optimizer.extract()
    squad["fixtures"] = squad["player_id"].map(
        candidates.set_index("player_id")["fixtures"])

    if config.output_path:
        path = Path(config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        squad.to_csv(path, index=False)
        LOGGER.info("Wrote squad to %s", path)

    return {"squad": squad, "gameweek": gameweek, "deadline": deadline,
            "candidates": candidates, "optimizer_config": optimizer_config}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FPL Phase 5: live inference and forward projection.")
    parser.add_argument("-i", "--input", default="fpl_ml_features.csv")
    parser.add_argument("-d", "--model-dir", default="models")
    parser.add_argument("--medians", default="fpl_position_medians.json")
    parser.add_argument("-b", "--budget", type=float, default=100.0)
    parser.add_argument("--gameweek", type=int, default=None,
                        help="Override the target gameweek (default: the API's next)")
    parser.add_argument("--allow-status", nargs="+", default=["a"],
                        help="FPL status codes to keep (default: a)")
    parser.add_argument("--min-chance", type=int, default=0,
                        help="Minimum chance_of_playing_next_round percent")
    parser.add_argument("--bench-weight", type=float, default=0.0)
    parser.add_argument("--reuse-last-row", action="store_true",
                        help="Copy the last row's rolling features instead of recomputing")
    parser.add_argument("--partial-windows", action="store_true",
                        help="Allow windows shorter than w instead of imputing")
    parser.add_argument("--exclude", default="", help="Comma-separated names to exclude")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache API responses here (default: always live)")
    parser.add_argument("-o", "--output", default=None, help="Optional CSV path for the squad")
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

    config = LiveConfig(
        features_path=Path(args.input),
        model_dir=Path(args.model_dir),
        medians_path=Path(args.medians),
        budget=args.budget,
        gameweek=args.gameweek,
        allow_status=tuple(args.allow_status),
        min_chance=args.min_chance,
        bench_weight=args.bench_weight,
        reuse_last_row=args.reuse_last_row,
        partial_windows=args.partial_windows,
        exclude=[n for n in args.exclude.split(",") if n.strip()],
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        output_path=Path(args.output) if args.output else None,
    )

    try:
        outcome = run_live_inference(config)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except AssertionError as exc:
        LOGGER.error("Solution violated an FPL rule: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    print_live_report(outcome["squad"], outcome["gameweek"], config, outcome["deadline"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
