"""
FPL ML Pipeline — V3: Multi-Period Transfer Solver
==================================================

Extends the single-gameweek knapsack of Phase 4 into a rolling H-week plan that
respects the real FPL transfer economy: a free-transfer bank that rolls up to 5,
a -4 point penalty per excess transfer, and a cash bank that must stay solvent
every week.

Decision variables, per player i and gameweek t
    x[i,t]    in the 15-man squad
    s[i,t]    in the starting XI
    c[i,t]    captain (points doubled)
    in[i,t]   transferred in this week
    out[i,t]  transferred out this week
    FT[t]     free transfers available entering week t   (integer, 0..5)
    h[t]      point hits taken in week t                 (integer, >= 0)
    b[t]      cash in the bank after week t's transfers  (continuous, >= 0)

Objective
    max  sum_t  gamma^(t-1) * [ sum_i p[i,t]*(s[i,t]+c[i,t])
                                + w_b * sum_i p[i,t]*(x[i,t]-s[i,t])
                                - 4*h[t] ]
         - epsilon * sum_t h[t]

Usage
-----
    # plan from your live squad, pulled straight from the FPL API
    python fpl_transfers.py --team-id 1234567 --horizon 5

    # or supply the squad manually
    python fpl_transfers.py --squad 1,15,233,...,610 --bank 1.4 --free-transfers 2

    # wildcard: no starting squad, free rein in week 1
    python fpl_transfers.py --wildcard --horizon 5

    # offline / testing against a precomputed projection table
    python fpl_transfers.py --projections-csv projections.csv --squad-json squad.json


The free-transfer recursion
---------------------------
The real rule nests two nonlinear operators:

    h[t]   = max(0, n[t] - FT[t])                       where n[t] = sum_i in[i,t]
    FT[t+1] = min(5, max(0, FT[t] - n[t]) + 1)

Both linearise without indicator binaries, because the objective already pushes
each variable to the right bound:

    h[t]    >= n[t] - FT[t]          h[t] only ever costs points, so the solver
    h[t]    >= 0                     drives it to this lower bound

    FT[t+1] <= 5                     FT[t+1] only ever relaxes next week's hit
    FT[t+1] <= FT[t] - n[t] + h[t] + 1   constraint, so the solver drives it up

The residual `FT[t] - n[t] + h[t]` equals `max(0, FT[t] - n[t])` exactly when
h[t] is tight, which it is at the optimum.

One degeneracy: h[t] carries a +1 coefficient in the FT recursion, so in
principle the solver could buy a free transfer for 4 points. That is strictly
unprofitable for any decay < 1 (pay 4*gamma^(t-1) now, save at most 4*gamma^t
later), but at gamma = 1 it is an exact tie and a solver may return either. A
tiny secondary penalty `epsilon * h[t]` breaks the tie for far less machinery
than the big-M indicator an exact formulation would need.


Modelling simplifications, stated plainly
-----------------------------------------
* PRICES ARE STATIC across the horizon. Real prices drift nightly with transfer
  volume; here they are frozen at today's. With --team-id, owned players are
  valued at their real selling price (half of any rise since purchase, from the
  public transfer log); with a manual squad they sell at today's price.
* FORM IS FROZEN at the latest observation. Rolling features for GW N+3 would
  need GW N+1 and N+2 results, which do not exist. Only the fixture varies
  across the horizon — opponent, venue, and the resulting difficulty. This is
  standard practice and is the main reason for the decay factor.
* CHIPS ARE PLANNED WITHIN THE HORIZON ONLY. With --team-id the solver knows
  which copies you still hold and plays one when its gain beats a fixed reserve
  value (--chip-reserve), or when the copy expires inside the horizon. It cannot
  see a double gameweek beyond the horizon, which is what the reserve stands in
  for. `--wildcard` is separate: it means "build from scratch in week 1".
* AVAILABILITY IS A WEEK-1 SNAPSHOT. A player injured today is excluded for the
  whole horizon even if he would return in week 4. An injured player you OWN
  stays in the pool at 0 points, so the plan can sell or bench him.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pulp is required: pip install pulp") from exc

# Rules and shapes only — no model loading from here. `fpl_optimizer.load_models`
# reads the combined `fpl_model_{POS}.joblib` bundles, which hold a pickled
# TwoStageModel; see the note on model loading below.
from fpl_optimizer import (
    FORMATION_BOUNDS,
    MAX_PER_CLUB,
    POSITIONS,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    STARTING_XI,
    ensure_printable_console,
)

# Two-stage model loading (V4). Imported at MODULE level, deliberately:
#
#   * `load_two_stage_models` prefers `fpl_model_{POS}_prob.joblib` and
#     `_pts.joblib`, which contain plain XGBClassifier / XGBRegressor objects.
#     Their classes live in an installed package, so they unpickle from any
#     entry point — no `__main__` dependency at all.
#   * `TwoStageModel` is imported so the combined-bundle fallback still works.
#     Importing it here also binds the name into this process's `__main__` when
#     the script is run directly, which is what rescues model directories
#     written by an older fpl_train.py that pickled the class from `__main__`.
#     The import in `fpl_models` registers the same alias for the case where
#     this file is imported rather than executed.
from fpl_live_inference import load_two_stage_models
from fpl_models import TwoStageModel  # noqa: F401

LOGGER = logging.getLogger("fpl.transfers")

BASE_URL = "https://fantasy.premierleague.com/api"
ENTRY_PICKS_URL = f"{BASE_URL}/entry/{{entry_id}}/event/{{event}}/picks/"
ENTRY_URL = f"{BASE_URL}/entry/{{entry_id}}/"
ENTRY_HISTORY_URL = f"{BASE_URL}/entry/{{entry_id}}/history/"
ENTRY_TRANSFERS_URL = f"{BASE_URL}/entry/{{entry_id}}/transfers/"

MAX_FREE_TRANSFERS = 5          # FPL rolls up to 5 from 2024/25
HIT_COST = 4.0                  # points per excess transfer

# Two tie-breakers, both far below the scale of a real decision (~0.1-4 points).
# HIT: stops the solver inflating hits to manufacture a free transfer.
# FT:  makes the solver PREFER a larger bank, so the <= constraints on FT bind
#      and reproduce min() exactly. Without it FT is objective-free whenever no
#      transfer is made, and the solver returns an arbitrary feasible value.
# HIT must dominate FT, so buying an FT with a hit is never profitable even at
# decay 1.0: cost 4*gamma + 1e-3 vs gain 4*gamma + 1e-4.
HIT_TIEBREAK_EPSILON = 1e-3
FT_TIEBREAK_EPSILON = 1e-4

# FPL's chip names, as the API spells them.
CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc")
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost",
               "3xc": "Triple Captain"}

# Since 2025-26 every manager gets two of each chip, one per half of the season.
# Live runs read the windows from bootstrap-static; this is the fallback and
# what the season simulator plays with.
STANDARD_CHIP_WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("wildcard", 2, 19), ("wildcard", 20, 38),
    ("freehit", 2, 19), ("freehit", 20, 38),
    ("bboost", 1, 19), ("bboost", 20, 38),
    ("3xc", 1, 19), ("3xc", 20, 38),
)

# What holding a chip is worth, in projected points: the solver only plays one
# when the gain beats keeping it for a better week. Set before looking at any
# replay, from the scale of a typical projected week: a bench of ~10, a captain
# of ~5. A copy whose window ends inside the horizon is free to use.
DEFAULT_CHIP_RESERVE: Dict[str, float] = {
    "wildcard": 20.0, "freehit": 15.0, "bboost": 12.0, "3xc": 6.0,
}


@dataclass
class TransferConfig:
    features_path: Path = Path("fpl_ml_features.csv")
    model_dir: Path = Path("models")
    medians_path: Path = Path("fpl_position_medians.json")
    projections_csv: Optional[Path] = None
    squad_json: Optional[Path] = None
    entry_id: Optional[int] = None
    current_squad: Sequence[int] = ()
    current_bank: float = 0.0
    free_transfers: Optional[int] = None   # None -> derived from --team-id, else 1
    wildcard: bool = False
    horizon: int = 5
    decay: float = 0.85
    bench_weight: float = 0.10       # a 5-week plan needs a usable bench
    budget: float = 100.0            # only used when --wildcard
    max_transfers_per_week: int = 3  # keeps the search tractable
    candidate_pool: int = 40         # top-N per position, plus the current squad
    team_rating_window: int = 10
    allow_status: Sequence[str] = ("a",)
    min_chance: int = 0
    time_limit: Optional[int] = 120
    solver_msg: bool = False
    output_path: Optional[Path] = None
    # Unused chip copies as (name, first GW, last GW); empty means no chips.
    chip_windows: Sequence[Tuple[str, int, int]] = ()
    use_chips: bool = True            # with --team-id, plan chips you still hold
    # Experimental: time chips by their best remaining week. Off by default: over
    # 15 replays of 2025-26 it scored 42 points fewer than the fixed reserves.
    chip_calendar: bool = False
    # Per-copy reserves from the chip calendar; overrides chip_reserve when set.
    chip_window_reserve: Optional[Dict[Tuple[str, int, int], float]] = None
    chip_reserve: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CHIP_RESERVE))
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 2. Initial squad state (t = 0)
# --------------------------------------------------------------------------- #

def load_squad_state(config: TransferConfig, client=None
                     ) -> Tuple[List[int], float, int, Dict[int, float],
                                List[Tuple[str, int]]]:
    """
    Resolve (squad element ids, bank in £m, free transfers, purchase prices, chips
    played) for t = 0.

    Sources, in priority order: --wildcard, --squad-json, --team-id (live API),
    then the explicit --squad / --bank / --free-transfers flags. Purchase prices
    are only known for --team-id; elsewhere owned players sell at today's price.
    An explicit --free-transfers always wins over a derived count.
    """
    if config.wildcard:
        LOGGER.info("Wildcard mode: no starting squad, week 1 is a free selection.")
        return [], config.budget, MAX_FREE_TRANSFERS, {}, []

    if config.squad_json:
        payload = json.loads(Path(config.squad_json).read_text(encoding="utf-8"))
        squad = [int(i) for i in payload.get("squad", payload.get("picks", []))]
        bank = float(payload.get("bank", config.current_bank))
        free = int(payload.get("free_transfers", config.free_transfers or 1))
        LOGGER.info("Loaded squad state from %s: %d players, bank %.1f, %d FT.",
                    config.squad_json, len(squad), bank, free)
        return squad, bank, free, {}, []

    if config.entry_id and client is not None:
        state = fetch_entry_state(client, config.entry_id, config.features_path)
        if config.free_transfers is not None and config.free_transfers != state["free"]:
            LOGGER.info("Using --free-transfers %d instead of the derived %d.",
                        config.free_transfers, state["free"])
            state["free"] = config.free_transfers
        return (state["squad"], state["bank"], state["free"], state["purchase_prices"],
                state["chips_used"])

    squad = [int(i) for i in config.current_squad]
    free = config.free_transfers if config.free_transfers is not None else 1
    if squad:
        LOGGER.info("Using squad from --squad: %d players, bank %.1f, %d FT.",
                    len(squad), config.current_bank, free)
    return squad, config.current_bank, free, {}, []


def chip_rules(bootstrap: Optional[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    """Every chip copy this season as (name, first GW, last GW), from bootstrap-static."""
    rules = [(c["name"], int(c["start_event"]), int(c["stop_event"]))
             for c in (bootstrap or {}).get("chips", [])
             if c.get("name") in CHIP_NAMES and c.get("start_event") and c.get("stop_event")]
    return rules or list(STANDARD_CHIP_WINDOWS)


def available_chip_windows(rules: Sequence[Tuple[str, int, int]],
                           used: Sequence[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
    """Copies not yet played: each chip played uses the copy whose window holds its GW."""
    remaining = list(rules)
    for name, event in used:
        spent = next((w for w in remaining if w[0] == name and w[1] <= event <= w[2]), None)
        if spent is not None:
            remaining.remove(spent)
    return remaining


def replay_free_transfers(history: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Free transfers available for the next deadline, replayed from the season.

    The API never states the count, but `entry/{id}/history/` records every
    gameweek's transfers and hit cost, which is enough to replay the rule: one
    free transfer after the opening squad, then
    FT[t+1] = min(5, max(0, FT[t] - transfers[t]) + 1). Wildcard and Free Hit
    weeks report zero transfers and keep the bank, which the same formula
    handles. Each week's hit cost is checked against the replay; a mismatch
    means a rule we do not model (a one-off FPL top-up, say) and is reported.
    """
    weeks = sorted(history.get("current", []), key=lambda g: g["event"])
    if not weeks:
        return 1, ["no gameweeks played yet"]

    problems: List[str] = []
    free = 1                                   # after the opening squad
    for week in weeks[1:]:
        made = int(week.get("event_transfers", 0))
        hits = int(week.get("event_transfers_cost", 0)) // int(HIT_COST)
        expected = max(0, made - free)
        if hits != expected:
            problems.append(f"GW{week['event']}: {hits} hit(s) charged, replay expected "
                            f"{expected} with {free} FT")
        free = min(MAX_FREE_TRANSFERS, max(0, free - made) + 1)
    return free, problems


def purchase_prices(transfers: List[Dict[str, Any]], squad: Sequence[int],
                    free_hit_weeks: Iterable[int],
                    opening_prices: Dict[int, float]) -> Dict[int, float]:
    """
    What the manager paid for each owned player, in £m.

    The latest transfer-in of each player gives his price; Free Hit transfers
    are skipped because that squad reverts. Anyone never transferred in was
    in the opening squad and cost his first price of the season.
    """
    skip = set(free_hit_weeks)
    paid: Dict[int, float] = {}
    for transfer in sorted(transfers, key=lambda t: t.get("time", "")):
        if transfer.get("event") in skip:
            continue
        paid[int(transfer["element_in"])] = transfer["element_in_cost"] / 10.0
    return {pid: paid.get(pid, opening_prices.get(pid)) for pid in squad
            if paid.get(pid, opening_prices.get(pid)) is not None}


def selling_value(paid: float, current: float) -> float:
    """FPL keeps half of any rise, rounded down to £0.1m; falls are taken in full."""
    paid_t, current_t = int(round(paid * 10)), int(round(current * 10))
    if current_t <= paid_t:
        return current_t / 10.0
    return (paid_t + (current_t - paid_t) // 2) / 10.0


def opening_prices_from_history(features_path: Path) -> Dict[int, float]:
    """Each player's first recorded price of the latest season in the feature file."""
    try:
        frame = pd.read_csv(features_path, usecols=["season", "player_id", "gameweek",
                                                    "gw_cost"])
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Could not read opening prices from %s: %s", features_path, exc)
        return {}
    latest = frame[frame["season"] == frame["season"].max()].sort_values("gameweek")
    first = latest.dropna(subset=["gw_cost"]).groupby("player_id").head(1)
    return dict(zip(first["player_id"].astype(int), first["gw_cost"].astype(float)))


def fetch_entry_state(client, entry_id: int,
                      features_path: Path = Path("fpl_ml_features.csv")) -> Dict[str, Any]:
    """
    Pull the manager's squad, bank, free transfers and purchase prices.

    Picks give the squad and bank; history gives the free-transfer replay and
    chips; the transfer log gives what was paid. If the latest gameweek was a
    Free Hit, the squad that carries forward is the one from the week before.
    """
    entry = client.get_json(ENTRY_URL.format(entry_id=entry_id))
    if not entry:
        raise RuntimeError(f"Could not fetch entry {entry_id}. Is the team id correct?")
    event = entry.get("current_event")
    if event is None:
        raise RuntimeError(f"Entry {entry_id} has no current gameweek — season not started?")

    history = client.get_json(ENTRY_HISTORY_URL.format(entry_id=entry_id)) or {}
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    squad_event = event - 1 if chips.get(event) == "freehit" and event > 1 else event
    if squad_event != event:
        LOGGER.info("GW%d was a Free Hit; planning from the GW%d squad it reverts to.",
                    event, squad_event)

    picks = client.get_json(ENTRY_PICKS_URL.format(entry_id=entry_id, event=squad_event))
    if not picks or "picks" not in picks:
        raise RuntimeError(f"Could not fetch picks for entry {entry_id}, GW{squad_event}.")
    squad = [int(p["element"]) for p in picks["picks"]]
    current = next((g for g in history.get("current", []) if g["event"] == event), None)
    bank = float((current or picks.get("entry_history", {})).get("bank", 0)) / 10.0

    free, problems = replay_free_transfers(history)
    for problem in problems:
        LOGGER.warning("Free-transfer replay: %s. Pass --free-transfers if the count "
                       "below is wrong.", problem)

    transfers = client.get_json(ENTRY_TRANSFERS_URL.format(entry_id=entry_id)) or []
    pending = [t for t in transfers if t.get("event", 0) > event]
    if pending:
        LOGGER.warning("%d transfer(s) already made for GW%d are not in the picks yet; the "
                       "plan starts from the squad before them.", len(pending), event + 1)
    free_hits = [e for e, name in chips.items() if name == "freehit"]
    paid = purchase_prices([t for t in transfers if t.get("event", 0) <= event], squad,
                           free_hits, opening_prices_from_history(features_path))

    LOGGER.info("Entry %s (GW%s): %d players, bank %.1f, %d free transfer(s) for the "
                "next deadline; purchase prices known for %d/%d.",
                entry_id, event, len(squad), bank, free, len(paid), len(squad))
    chips_used = [(c["name"], int(c["event"])) for c in history.get("chips", [])]
    if chips_used:
        LOGGER.info("Chips played: %s", ", ".join(
            f"{CHIP_LABELS.get(n, n)} GW{e}" for n, e in sorted(chips_used, key=lambda c: c[1])))
    return {"squad": squad, "bank": bank, "free": free, "purchase_prices": paid,
            "chips_used": chips_used, "event": event}


# --------------------------------------------------------------------------- #
# Horizon projections
# --------------------------------------------------------------------------- #

def build_horizon_projections(config: TransferConfig,
                              owned: Sequence[int] = ()) -> pd.DataFrame:
    """
    Score every available player for each of the next H gameweeks.

    Reuses the Phase 5 machinery: live availability, real fixtures, forward form
    and forward fixture difficulty. Form and team ratings are computed ONCE and
    held constant across the horizon; only the fixture changes week to week.

    Returns tidy rows: player_id, gameweek, predicted_points, cost, position,
    team, player_name, fixtures.
    """
    from fpl_features import build_team_match_table, derive_fdr_matchups, latest_team_ratings
    from fpl_ingest import build_position_lookup, build_team_lookup
    from fpl_live_inference import (
        LiveConfig,
        assemble_feature_matrix,
        build_fixture_rows,
        build_live_players,
        collapse_to_players,
        fetch_future_fixtures,
        filter_available,
        load_history,
        make_client,
        predict_points,
        required_rolling_features,
        resolve_next_gameweek,
    )

    live_config = LiveConfig(
        features_path=config.features_path,
        model_dir=config.model_dir,
        medians_path=config.medians_path,
        allow_status=config.allow_status,
        min_chance=config.min_chance,
        fdr=True,
    )
    client = make_client(live_config)

    bootstrap = client.fetch_bootstrap()
    team_lookup = build_team_lookup(bootstrap["teams"])
    position_lookup = build_position_lookup(bootstrap["element_types"])
    fixtures = fetch_future_fixtures(client)
    first_gw = resolve_next_gameweek(bootstrap, fixtures, None)

    everyone = build_live_players(bootstrap, team_lookup, position_lookup)
    players = filter_available(everyone, live_config)

    # V4: loads the split _prob/_pts XGBoost files natively. Falls back to the
    # combined TwoStageModel wrapper, then to a V3 single regressor.
    bundles, _ = load_two_stage_models(config.model_dir)
    needed = required_rolling_features(bundles)
    history = load_history(config.features_path)

    # Team ratings are computed ONCE and reused for every gameweek in the
    # horizon. They are frozen going forward anyway (we cannot know GW N+1's
    # results when projecting GW N+2), so recomputing per week would be identical
    # work for an identical answer. Passing them through keeps `fdr=True`, so
    # every future gameweek matrix carries full opponent ratings via the shared
    # derive_fdr_matchups().
    ratings = latest_team_ratings(build_team_match_table(history),
                                  config.team_rating_window)
    # One download for the whole horizon. The feed only prices the coming round,
    # so later gameweeks mostly score on the odds-free fallback, as in training.
    from fpl_odds import fetch_upcoming_odds
    odds_table = fetch_upcoming_odds()
    LOGGER.info("Forward team ratings built once for %d club(s); reused across the horizon.",
                len(ratings))

    frames: List[pd.DataFrame] = []
    for offset in range(config.horizon):
        gameweek = first_gw + offset
        try:
            rows = build_fixture_rows(players, fixtures, team_lookup, gameweek)
        except RuntimeError as exc:
            LOGGER.warning("Stopping horizon at GW%d: %s", gameweek, exc)
            break

        matrix = assemble_feature_matrix(rows, history, needed, live_config, ratings,
                                         odds_table)
        _assert_fdr_present(matrix, gameweek)
        scored = predict_points(matrix, bundles)
        collapsed = collapse_to_players(scored)
        collapsed["gameweek"] = gameweek
        frames.append(collapsed)
        LOGGER.info("GW%d: %d players scored (mean %.2f).",
                    gameweek, len(collapsed), collapsed["predicted_points"].mean())

    if not frames:
        raise RuntimeError("No gameweeks could be projected.")
    projections = pd.concat(frames, ignore_index=True)
    return add_unavailable_owned(projections, everyone, owned)


def add_unavailable_owned(projections: pd.DataFrame, everyone: pd.DataFrame,
                          owned: Sequence[int]) -> pd.DataFrame:
    """
    Keep owned players the availability filter removed, at 0 points.

    An injured or suspended player still fills a squad slot until he is sold.
    Dropping him leaves the continuity constraints with nothing to bind to and
    the plan infeasible, so he goes back in projecting nothing, which lets the
    solver sell him or leave him on the bench.
    """
    missing = sorted(set(int(i) for i in owned) - set(projections["player_id"].astype(int)))
    if not missing:
        return projections
    meta = everyone.set_index("element_id").reindex(missing)
    first = int(projections["gameweek"].min())
    rows = pd.DataFrame({
        "player_id": missing, "gameweek": first, "predicted_points": 0.0,
        "cost": meta["now_cost"].to_numpy(), "position": meta["position"].to_numpy(),
        "team": meta["team"].to_numpy(), "player_name": meta["player_name"].to_numpy(),
        "fixtures": "unavailable",
    })
    news = "; ".join(f"{n} ({s})" for n, s in zip(meta["web_name"], meta["status"]))
    LOGGER.info("Kept %d unavailable owned player(s) at 0 points: %s", len(missing), news)
    return pd.concat([projections, rows], ignore_index=True)


def _assert_fdr_present(matrix: pd.DataFrame, gameweek: int) -> None:
    """
    Fail loudly if a future gameweek reaches the model without opponent ratings.

    Models trained with `fdr_*` will happily accept NaN (XGBoost treats it as a
    missing branch) and return plausible-looking numbers that ignore the fixture
    entirely — a silent, expensive failure. Cheaper to crash here.
    """
    base = [c for c in matrix.columns
            if c.startswith("fdr_") and "_roll_" not in c and not c.endswith("_observed")]
    if not base:
        raise RuntimeError(
            f"GW{gameweek}: no fixture-difficulty columns in the feature matrix. "
            f"Phase 2 must be run with --strength-source derived."
        )
    empty = [c for c in base if matrix[c].isna().all()]
    if empty:
        raise RuntimeError(
            f"GW{gameweek}: fixture-difficulty columns are entirely NaN: "
            f"{', '.join(sorted(empty))}."
        )
    LOGGER.debug("GW%d: %d FDR columns populated.", gameweek, len(base))


def load_projections(config: TransferConfig, owned: Sequence[int] = ()) -> pd.DataFrame:
    """Read a precomputed projection table, or build one from the live API."""
    if config.projections_csv:
        frame = pd.read_csv(config.projections_csv)
        required = {"player_id", "gameweek", "predicted_points", "cost", "position", "team"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{config.projections_csv} missing columns: {sorted(missing)}")
        if "player_name" not in frame.columns:
            frame["player_name"] = frame["player_id"].astype(str)

        # Honour --horizon here too. A projections file may hold more gameweeks
        # than asked for, and silently planning past the horizon makes the run
        # slower and the report disagree with the flag.
        available = sorted(int(g) for g in frame["gameweek"].unique())
        wanted = available[: config.horizon]
        if len(wanted) < len(available):
            LOGGER.info("Truncating %d available gameweeks to the %d-week horizon.",
                        len(available), len(wanted))
            frame = frame[frame["gameweek"].isin(wanted)].copy()
        elif len(wanted) < config.horizon:
            LOGGER.warning("Only %d gameweek(s) available; horizon shortened from %d.",
                           len(wanted), config.horizon)

        LOGGER.info("Loaded projections from %s: %d rows, %d gameweeks.",
                    config.projections_csv, len(frame), frame["gameweek"].nunique())
        return frame
    return build_horizon_projections(config, owned)


def prune_candidates(
    projections: pd.DataFrame, current_squad: Sequence[int], top_n: int,
    cheap_per_position: int = 8, club_enablers: int = 3,
) -> pd.DataFrame:
    """
    Shrink the player pool before it reaches the solver.

    Full pool x horizon x five variable families is tens of thousands of
    binaries, which CBC will chew on for a very long time. Pruning cuts that by
    an order of magnitude — but pruning on projected points ALONE makes the
    problem infeasible, which is not obvious until it happens.

    Top-N by points is a pool of expensive players. A wildcard then has to build
    a legal 15 inside £100m from it and cannot, so CBC returns `Infeasible` with
    no hint as to why. Real squads are barbells: premiums funded by £4.0m
    enablers who will never appear in a top-N-by-points list.

    So three slices per position:
      * top_n by total projected points        — the players you actually want
      * top_n // 2 by points per £m            — value, the mid-price engine
      * cheap_per_position cheapest            — enablers that make the budget work
    plus every player currently owned, who must stay representable or the
    continuity constraints have nothing to bind to.
    """
    totals = (projections.groupby(["player_id", "position", "team"], as_index=False)
              .agg(points=("predicted_points", "sum"), cost=("cost", "first")))
    totals["value"] = totals["points"] / totals["cost"].clip(lower=0.1)

    # 1. PIN the current squad first, unconditionally. Every owned player must
    #    stay representable or the continuity constraints have nothing to bind
    #    to and the model is infeasible from t=1.
    pinned = {int(i) for i in current_squad}
    keep = set(pinned)

    for position in POSITIONS:
        block = totals[totals["position"] == position]
        keep.update(block.nlargest(top_n, "points")["player_id"].astype(int))
        keep.update(block.nlargest(max(1, top_n // 2), "value")["player_id"].astype(int))
        keep.update(block.nsmallest(cheap_per_position, "cost")["player_id"].astype(int))

    # 2. Enablers from EVERY club. The 3-per-club rule is a constraint on
    #    breadth, not depth: a pool of 140 players drawn from eight clubs cannot
    #    field a legal 15. Keeping the cheapest few at each club guarantees the
    #    solver can always spread out, and they are exactly the £4.0m bench
    #    fodder a real squad is built on.
    for club, block in totals.groupby("team"):
        keep.update(block.nsmallest(club_enablers, "cost")["player_id"].astype(int))

    pruned = projections[projections["player_id"].isin(keep)].copy()

    missing_pins = pinned - set(pruned["player_id"].astype(int))
    if missing_pins:
        LOGGER.warning(
            "%d owned player(s) have no projection and cannot be pinned: %s. "
            "They are most likely filtered by --allow-status (injured/suspended) "
            "or absent from the feature file.",
            len(missing_pins), sorted(missing_pins))

    per_player = pruned.drop_duplicates("player_id")
    floor = sum(float(per_player[per_player["position"] == pos]["cost"].nsmallest(q).sum())
                for pos, q in SQUAD_QUOTA.items())
    LOGGER.info("Candidate pool: %d players (from %d) across %d clubs x %d gameweeks; "
                "%d pinned; cheapest legal 15 costs %.1f.",
                pruned["player_id"].nunique(), projections["player_id"].nunique(),
                per_player["team"].nunique(), pruned["gameweek"].nunique(),
                len(pinned & set(per_player["player_id"].astype(int))), floor)
    return pruned


def diagnose_infeasibility(
    projections: pd.DataFrame, squad: Sequence[int], bank: float, config: TransferConfig
) -> List[str]:
    """
    Name the constraint that cannot be satisfied, before handing CBC the model.

    `Infeasible` from a solver is close to useless on a model this size — it
    tells you nothing about which of 4,000 constraints failed. These are the
    conditions that actually break in practice, checked per gameweek.
    """
    problems: List[str] = []
    per_player = projections.drop_duplicates("player_id")
    wildcard = not squad

    # Enough players per position, in every gameweek.
    for position, quota in SQUAD_QUOTA.items():
        available = int((per_player["position"] == position).sum())
        if available < quota:
            problems.append(f"only {available} {position} in the pool, need {quota}")

    # Enough clubs to spread 15 players at 3 per club.
    clubs = per_player["team"].nunique()
    if clubs * MAX_PER_CLUB < SQUAD_SIZE:
        problems.append(f"pool spans {clubs} club(s) x max {MAX_PER_CLUB} < {SQUAD_SIZE}")

    # Budget: only binds on a wildcard/fresh build, where there is no squad to
    # inherit and the whole 15 must fit the budget.
    if wildcard:
        floor = sum(float(per_player[per_player["position"] == pos]["cost"].nsmallest(q).sum())
                    for pos, q in SQUAD_QUOTA.items())
        if floor > config.budget:
            problems.append(f"cheapest legal 15 costs {floor:.1f} > budget {config.budget:.1f}")
    else:
        # The existing squad must itself be legal, or "make no transfers" is not
        # a feasible fallback and everything downstream fails.
        owned = per_player[per_player["player_id"].isin(squad)]
        if len(owned) < SQUAD_SIZE:
            problems.append(
                f"only {len(owned)} of your {len(squad)} players have projections")
        else:
            counts = owned["position"].value_counts().to_dict()
            for position, quota in SQUAD_QUOTA.items():
                if counts.get(position, 0) != quota:
                    problems.append(
                        f"your squad has {counts.get(position, 0)} {position}, expected {quota}")
            # Over the club quota is fixable — a mid-season move can put a 4th
            # player at one club, and FPL makes you sell one — so it is not an
            # infeasibility as long as a transfer is allowed this week.
            worst = owned["team"].value_counts()
            if not worst.empty and worst.max() > MAX_PER_CLUB:
                if config.max_transfers_per_week < worst.max() - MAX_PER_CLUB:
                    problems.append(
                        f"your squad has {worst.max()} players from {worst.idxmax()}, "
                        f"max {MAX_PER_CLUB}")
                else:
                    LOGGER.warning("Squad has %d players from %s; this week's plan must "
                                   "sell down to %d.", worst.max(), worst.idxmax(),
                                   MAX_PER_CLUB)

    return problems


# --------------------------------------------------------------------------- #
# The multi-period MILP
# --------------------------------------------------------------------------- #

class MultiPeriodOptimizer:
    """
    Rolling H-week squad plan with a real transfer economy and the four chips.

    Chips, when `config.chip_windows` lists unused copies:
      * Wildcard     unlimited transfers that week, no hits, FT bank kept
      * Free Hit     a separate one-week squad z[i,t], legal and within the
                     squad's selling value plus bank; the persistent squad
                     x[i,t] stands still and returns the next week
      * Bench Boost  the four bench players score in full
      * Triple Capt. the captain scores three times instead of twice
    At most one chip per gameweek, each copy only inside its window. Playing a
    chip costs `config.chip_reserve[name]`, the value of keeping it for a later
    week, except where its window ends inside the horizon (use it or lose it).
    """

    def __init__(self, projections: pd.DataFrame, squad: Sequence[int],
                 bank: float, free_transfers: int, config: TransferConfig) -> None:
        self.config = config
        self.gameweeks = sorted(int(g) for g in projections["gameweek"].unique())
        self.horizon = len(self.gameweeks)

        # One row per player with static attributes, plus a points lookup.
        meta_columns = ["player_id", "player_name", "position", "team", "cost"]
        self.players = (projections[meta_columns]
                        .drop_duplicates("player_id")
                        .set_index("player_id")
                        .sort_index())
        self.ids = list(self.players.index)

        self.points = {
            (int(row.player_id), int(row.gameweek)): float(row.predicted_points)
            for row in projections.itertuples(index=False)
        }
        self.fixtures = {
            (int(row.player_id), int(row.gameweek)): getattr(row, "fixtures", "")
            for row in projections.itertuples(index=False)
        } if "fixtures" in projections.columns else {}

        self.initial_squad = set(int(i) for i in squad)
        unknown = self.initial_squad - set(self.ids)
        if unknown:
            raise RuntimeError(
                f"{len(unknown)} player(s) in your current squad have no projection: "
                f"{sorted(unknown)}. They may be injured (filtered by --allow-status) "
                f"or missing from the feature file."
            )
        self.bank0 = float(bank)
        self.ft0 = int(np.clip(free_transfers, 0, MAX_FREE_TRANSFERS))

        # A fresh build (no starting squad) is not the Wildcard chip; it is how
        # GW1 or a manual --wildcard starts, and it cannot also play chips in
        # its first week.
        self.fresh_build = not self.initial_squad
        self.windows = [w for w in (tuple(w) for w in config.chip_windows)
                        if w[0] in CHIP_NAMES
                        and any(w[1] <= t <= w[2] for t in self.gameweeks)]

        self.problem: Optional[pulp.LpProblem] = None
        self.x: Dict = {}
        self.z: Dict = {}
        self.s: Dict = {}
        self.c: Dict = {}
        self.tin: Dict = {}
        self.tout: Dict = {}
        self.ft: Dict = {}
        self.hits: Dict = {}
        self.hit_active: Dict = {}
        self.bank: Dict = {}
        self.chip: Dict[str, Dict[int, Any]] = {name: {} for name in CHIP_NAMES}

    # -- chips ---------------------------------------------------------------
    def _chip_weeks(self, name: str) -> List[int]:
        weeks = [t for t in self.gameweeks
                 if any(w[0] == name and w[1] <= t <= w[2] for w in self.windows)]
        # A Wildcard or Free Hit on top of a from-scratch build means nothing.
        if (self.fresh_build and name in ("wildcard", "freehit")
                and weeks and weeks[0] == self.gameweeks[0]):
            weeks = weeks[1:]
        # A Free Hit needs a second, one-week squad, which doubles the hardest
        # part of the model. Only week 1 of a plan is ever executed, so it is
        # offered there alone and reconsidered at every deadline.
        if name == "freehit":
            weeks = [t for t in weeks if t == self.gameweeks[0]]
        return weeks

    def _reserve(self, name: str, t: int) -> float:
        """Opportunity cost of playing `name` in week t: zero if its copy expires in view."""
        if self.config.chip_window_reserve:
            window = next((w for w in self.windows if w[0] == name and w[1] <= t <= w[2]), None)
            if window in self.config.chip_window_reserve:
                return float(self.config.chip_window_reserve[window])
        last = self.gameweeks[-1]
        expiring = any(w[0] == name and w[1] <= t <= w[2] and w[2] <= last
                       for w in self.windows)
        return 0.0 if expiring else float(self.config.chip_reserve.get(name, 0.0))

    def _use(self, name: str, t: int):
        """The chip's binary for week t, or 0 when it cannot be played then."""
        return self.chip[name].get(t, 0)

    # -- build ---------------------------------------------------------------
    def build(self) -> pulp.LpProblem:
        ids, weeks = self.ids, self.gameweeks
        cost = self.players["cost"].to_dict()
        position = self.players["position"].to_dict()
        club = self.players["team"].to_dict()
        clubs = sorted({str(club[i]) for i in ids})
        points = lambda i, t: self.points.get((i, t), 0.0)  # noqa: E731

        problem = pulp.LpProblem("FPL_MultiPeriod_Transfers", pulp.LpMaximize)

        self.x = pulp.LpVariable.dicts("squad", (ids, weeks), cat=pulp.LpBinary)
        self.s = pulp.LpVariable.dicts("start", (ids, weeks), cat=pulp.LpBinary)
        self.c = pulp.LpVariable.dicts("captain", (ids, weeks), cat=pulp.LpBinary)
        self.tin = pulp.LpVariable.dicts("in", (ids, weeks), cat=pulp.LpBinary)
        self.tout = pulp.LpVariable.dicts("out", (ids, weeks), cat=pulp.LpBinary)
        self.ft = pulp.LpVariable.dicts("ft", weeks, lowBound=0,
                                        upBound=MAX_FREE_TRANSFERS, cat=pulp.LpInteger)
        self.hits = pulp.LpVariable.dicts("hits", weeks, lowBound=0, cat=pulp.LpInteger)
        # Indicator: 1 when transfers exceed the free-transfer bank this week.
        self.hit_active = pulp.LpVariable.dicts("hit_active", weeks, cat=pulp.LpBinary)
        self.bank = pulp.LpVariable.dicts("bank", weeks, lowBound=0, cat=pulp.LpContinuous)

        for name in CHIP_NAMES:
            for t in self._chip_weeks(name):
                self.chip[name][t] = pulp.LpVariable(f"chip_{name}_{t}", cat=pulp.LpBinary)
        free_hit_weeks = set(self.chip["freehit"])

        # The squad that plays each week. Equal to x except in a Free Hit week,
        # where it is a fresh legal 15 and x carries the real squad through.
        self.z = {i: {} for i in ids}
        for t in weeks:
            for i in ids:
                self.z[i][t] = (pulp.LpVariable(f"play_{i}_{t}", cat=pulp.LpBinary)
                                if t in free_hit_weeks else self.x[i][t])

        # Bench Boost and Triple Captain, one continuous gain per week: the
        # gain is at most the week's bench (or captain) points and at most a
        # tight bound times the chip binary. Both only ever add points, so the
        # upper bounds pin them. Per-player products would be exact too, but
        # cost a variable per player-week and slow the solve tenfold.
        bench_gain, captain_gain = {}, {}
        for t in self.chip["bboost"]:
            top = sorted((points(i, t) for i in ids), reverse=True)
            bound = sum(top[:SQUAD_SIZE - STARTING_XI])
            g = pulp.LpVariable(f"bb_gain_{t}", lowBound=0, upBound=bound)
            problem += g <= pulp.lpSum(points(i, t) * (self.z[i][t] - self.s[i][t])
                                       for i in ids), f"bb_bench_{t}"
            problem += g <= bound * self.chip["bboost"][t], f"bb_on_{t}"
            bench_gain[t] = g
        for t in self.chip["3xc"]:
            bound = max((points(i, t) for i in ids), default=0.0)
            g = pulp.LpVariable(f"tc_gain_{t}", lowBound=0, upBound=bound)
            problem += g <= pulp.lpSum(points(i, t) * self.c[i][t] for i in ids),                 f"tc_captain_{t}"
            problem += g <= bound * self.chip["3xc"][t], f"tc_on_{t}"
            captain_gain[t] = g

        # ---- time-decayed objective ----
        gamma, w_b = self.config.decay, self.config.bench_weight
        weekly = []
        for index, t in enumerate(weeks):
            discount = gamma ** index
            weekly.append(discount * (
                pulp.lpSum(points(i, t) * (self.s[i][t] + self.c[i][t]) for i in ids)
                + w_b * pulp.lpSum(points(i, t) * (self.z[i][t] - self.s[i][t]) for i in ids)
                + (1 - w_b) * bench_gain.get(t, 0)
                + captain_gain.get(t, 0)
                - HIT_COST * self.hits[t]
                - pulp.lpSum(self._reserve(name, t) * self.chip[name][t]
                             for name in CHIP_NAMES if t in self.chip[name])
            ))
        # Tie-breakers. The FT reward is what makes `FT[t] <= ...` bind: without
        # it FT carries no objective weight in any week where no transfer is
        # made, so the solver is free to return any feasible value and the
        # reported bank is meaningless. See the constants for the ordering.
        problem += (pulp.lpSum(weekly)
                    - HIT_TIEBREAK_EPSILON * pulp.lpSum(self.hits[t] for t in weeks)
                    + FT_TIEBREAK_EPSILON * pulp.lpSum(self.ft[t] for t in weeks)
                    ), "discounted_expected_points"

        def legal_fifteen(squad, t: int, tag: str) -> None:
            problem.addConstraint(pulp.lpSum(squad[i][t] for i in ids) == SQUAD_SIZE,
                                  f"{tag}_size_{t}")
            for pos, quota in SQUAD_QUOTA.items():
                problem.addConstraint(
                    pulp.lpSum(squad[i][t] for i in ids if position[i] == pos) == quota,
                    f"{tag}_{pos}_{t}")
            for name in clubs:
                problem.addConstraint(
                    pulp.lpSum(squad[i][t] for i in ids if str(club[i]) == name)
                    <= MAX_PER_CLUB, f"{tag}_club_{_safe(name)}_{t}")

        # ---- per-week squad legality, XI and captaincy ----
        for t in weeks:
            legal_fifteen(self.x, t, "squad")
            if t in free_hit_weeks:
                legal_fifteen(self.z, t, "play")
                fh = self.chip["freehit"][t]
                for i in ids:
                    # z == x unless the Free Hit is on.
                    problem += self.z[i][t] - self.x[i][t] <= fh, f"fh_up_{i}_{t}"
                    problem += self.x[i][t] - self.z[i][t] <= fh, f"fh_down_{i}_{t}"

            problem += pulp.lpSum(self.s[i][t] for i in ids) == STARTING_XI, f"xi_{t}"
            for i in ids:
                problem += self.s[i][t] <= self.z[i][t], f"start_in_squad_{i}_{t}"
                problem += self.c[i][t] <= self.s[i][t], f"captain_starts_{i}_{t}"

            for pos, (low, high) in FORMATION_BOUNDS.items():
                members = [i for i in ids if position[i] == pos]
                problem += pulp.lpSum(self.s[i][t] for i in members) >= low, f"fmin_{pos}_{t}"
                problem += pulp.lpSum(self.s[i][t] for i in members) <= high, f"fmax_{pos}_{t}"

            problem += pulp.lpSum(self.c[i][t] for i in ids) == 1, f"one_captain_{t}"
            problem += pulp.lpSum(self._use(name, t) for name in CHIP_NAMES) <= 1, \
                f"one_chip_{t}"

        # ---- chip copies: each window's copy at most once ----
        for k, (name, start, stop) in enumerate(self.windows):
            in_window = [self.chip[name][t] for t in self.chip[name] if start <= t <= stop]
            if in_window:
                problem += pulp.lpSum(in_window) <= 1, f"chip_window_{name}_{k}"

        # ---- the transfer engine ----
        for index, t in enumerate(weeks):
            previous = weeks[index - 1] if index else None
            for i in ids:
                prior = self.x[i][previous] if previous is not None \
                    else (1 if i in self.initial_squad else 0)

                # x[i,t] = x[i,t-1] + in[i,t] - out[i,t]
                problem += (self.x[i][t] == prior + self.tin[i][t] - self.tout[i][t]), \
                    f"continuity_{i}_{t}"
                # Buying and selling the same player in one week burns a transfer
                # for nothing; forbid it rather than trusting the objective.
                problem += self.tin[i][t] + self.tout[i][t] <= 1, f"no_churn_{i}_{t}"

        # ---- free-transfer bank and hits ----
        big_m = SQUAD_SIZE + MAX_FREE_TRANSFERS + 1
        for index, t in enumerate(weeks):
            made = pulp.lpSum(self.tin[i][t] for i in ids)
            wc, fh = self._use("wildcard", t), self._use("freehit", t)

            if index == 0:
                problem += self.ft[t] == self.ft0, f"ft_initial_{t}"
            else:
                previous = weeks[index - 1]
                problem += self.ft[t] <= MAX_FREE_TRANSFERS, f"ft_cap_{t}"
                if index == 1 and self.fresh_build:
                    # A fresh build is not 15 transfers. Feeding those into the
                    # recursion drives FT to 5 - 15 + 1 = -9, below its lower
                    # bound, and the whole model goes infeasible. After a fresh
                    # build you simply start the next week with one FT.
                    problem += self.ft[t] == 1, f"ft_after_build_{t}"
                else:
                    made_prev = pulp.lpSum(self.tin[i][previous] for i in ids)
                    chip_prev = self._use("wildcard", previous) + self._use("freehit", previous)
                    # FT[t] <= FT[t-1] - n[t-1] + h[t-1] + 1, relaxed in a Wildcard
                    # or Free Hit week, whose transfers spend nothing; FT[t] <=
                    # FT[t-1] + 1 always, so those weeks still bank exactly one.
                    # Upper bounds only: the FT tie-break reward pushes FT up,
                    # so these bind and reproduce min() exactly.
                    problem += (self.ft[t] <= self.ft[previous] - made_prev
                                + self.hits[previous] + 1 + big_m * chip_prev), f"ft_roll_{t}"
                    problem += self.ft[t] <= self.ft[previous] + 1, f"ft_step_{t}"

            if index == 0 and self.fresh_build:
                # Same reason: the 15 opening picks are free, not hits.
                problem += self.hits[t] == 0, f"build_no_hits_{t}"
            else:
                # Pin h[t] = max(0, n[t] - FT[t]) EXACTLY, with an indicator.
                #
                # The -4 penalty alone is not enough. It looks like it should be:
                # inflating a hit costs 4 and can only save 4 later. But the
                # solver does not add a hit, it MOVES one — take a spurious hit
                # now, bank the extra FT, avoid a real hit next week. At decay
                # 1.0 that trade is exactly neutral in the main objective and any
                # tie-break on FT tips it positive. A test at decay 1.0 caught
                # the solver doing precisely this. Two constraints and one binary
                # per week close it off for good. A Wildcard lifts the lower
                # bound and pins hits to zero.
                y = self.hit_active[t]
                problem += self.hits[t] >= made - self.ft[t] - big_m * wc, f"hits_lb_{t}"
                problem += self.hits[t] <= made - self.ft[t] + big_m * (1 - y), \
                    f"hits_ub_active_{t}"
                problem += self.hits[t] <= big_m * y, f"hits_ub_zero_{t}"
                problem += self.hits[t] <= big_m * (1 - wc), f"hits_wildcard_{t}"
                problem += made <= self.config.max_transfers_per_week + SQUAD_SIZE * wc, \
                    f"max_transfers_{t}"
                # A Free Hit leaves the real squad alone that week.
                problem += made <= SQUAD_SIZE * (1 - fh), f"fh_freeze_{t}"

        # ---- cash bank ----
        for index, t in enumerate(weeks):
            previous_bank = self.bank[weeks[index - 1]] if index else self.bank0
            sold = pulp.lpSum(cost[i] * self.tout[i][t] for i in ids)
            bought = pulp.lpSum(cost[i] * self.tin[i][t] for i in ids)
            problem += self.bank[t] == previous_bank + sold - bought, f"bank_{t}"

            if t in free_hit_weeks:
                # The Free Hit squad is bought with the real squad's selling
                # value plus the bank. x[t] == x[t-1] in that week, so this is
                # what the manager could raise at the deadline.
                budget_m = float(sum(sorted(cost.values())[-SQUAD_SIZE:])) + 100.0
                problem += (pulp.lpSum(cost[i] * self.z[i][t] for i in ids)
                            <= self.bank[t] + pulp.lpSum(cost[i] * self.x[i][t] for i in ids)
                            + budget_m * (1 - self.chip["freehit"][t])), f"fh_budget_{t}"

        # Fresh build: no squad at t=0, so week 1 selection is bounded by the
        # budget rather than by transfers. Every "transfer in" is a free pick.
        if self.fresh_build:
            first = weeks[0]
            problem += (pulp.lpSum(cost[i] * self.x[i][first] for i in ids)
                        <= self.config.budget), "build_budget"

        self.problem = problem
        LOGGER.info("Built multi-period MILP: %d variables, %d constraints, %d gameweeks%s.",
                    len(problem.variables()), len(problem.constraints), self.horizon,
                    f", chips {sorted({w[0] for w in self.windows})}" if self.windows else "")
        return problem

    # -- solve ---------------------------------------------------------------
    def solve(self) -> str:
        if self.problem is None:
            self.build()
        solver = pulp.PULP_CBC_CMD(msg=self.config.solver_msg,
                                   timeLimit=self.config.time_limit)
        self.problem.solve(solver)
        status = pulp.LpStatus[self.problem.status]

        if status == "Not Solved":
            raise RuntimeError(
                f"Solver hit the {self.config.time_limit}s limit without a solution. "
                f"Try --candidate-pool 25, --horizon 3, or raise --time-limit."
            )
        if status != "Optimal":
            raise RuntimeError(
                f"Solver status '{status}'. Common causes: the current squad breaks the "
                f"3-per-club rule under the pruned pool, or the bank cannot cover a "
                f"forced transfer. Try --candidate-pool 60."
            )
        LOGGER.info("Solved: %s | objective = %.2f discounted points",
                    status, pulp.value(self.problem.objective))
        return status

    # -- extract -------------------------------------------------------------
    def extract(self) -> Dict[str, Any]:
        """Unpack the solution into a per-gameweek plan and validate it."""
        plan: List[Dict[str, Any]] = []
        on = lambda v: (v.value() if hasattr(v, "value") else v) > 0.5  # noqa: E731
        # Sorted by position so the report pairs like with like: squad quotas
        # are fixed, so every week sells and buys the same position mix.
        by_position = lambda i: (POSITIONS.index(self.players.at[i, "position"]), i)  # noqa: E731

        for index, t in enumerate(self.gameweeks):
            chip = next((name for name in CHIP_NAMES if t in self.chip[name]
                         and on(self.chip[name][t])), None)
            persistent = [i for i in self.ids if on(self.x[i][t])]
            chosen = [i for i in self.ids if on(self.z[i][t])]
            squad = self.players.loc[chosen].copy()
            squad["is_starter"] = [on(self.s[i][t]) for i in chosen]
            squad["is_captain"] = [on(self.c[i][t]) for i in chosen]
            squad["predicted_points"] = [self.points.get((i, t), 0.0) for i in chosen]
            squad["fixtures"] = [self.fixtures.get((i, t), "") for i in chosen]
            squad = squad.reset_index()

            transfers_in = sorted((i for i in self.ids if on(self.tin[i][t])), key=by_position)
            transfers_out = sorted((i for i in self.ids if on(self.tout[i][t])), key=by_position)
            if chip == "freehit":
                # Shown as the week's swaps; the real squad returns next week.
                transfers_in = sorted(set(chosen) - set(persistent), key=by_position)
                transfers_out = sorted(set(persistent) - set(chosen), key=by_position)

            starters = squad[squad["is_starter"]]
            bench = squad[~squad["is_starter"]]
            captain = squad[squad["is_captain"]]
            gw_points = float(starters["predicted_points"].sum())
            multiplier = 3 if chip == "3xc" else 2
            captain_bonus = (multiplier - 1) * float(captain["predicted_points"].sum())
            bench_points = float(bench["predicted_points"].sum()) if chip == "bboost" else 0.0
            hits = int(round(self.hits[t].value() or 0))
            n_transfers = 0 if chip == "freehit" else len(transfers_in)

            plan.append({
                "gameweek": t,
                "discount": self.config.decay ** index,
                "chip": chip,
                "squad": squad,
                "persistent_ids": persistent,
                "transfers_in": self.players.loc[transfers_in, "player_name"].tolist(),
                "transfers_out": self.players.loc[transfers_out, "player_name"].tolist(),
                "n_transfers": n_transfers,
                "free_transfers": int(round(self.ft[t].value() or 0)),
                "hits": hits,
                "hit_cost": HIT_COST * hits,
                "bank": float(self.bank[t].value() or 0.0),
                "squad_value": float(self.players.loc[persistent, "cost"].sum()),
                "xi_points": gw_points,
                "captain_bonus": captain_bonus,
                "bench_boost_points": bench_points,
                "net_points": gw_points + captain_bonus + bench_points - HIT_COST * hits,
            })

        validate_plan(plan, self)
        return {"plan": plan, "objective": pulp.value(self.problem.objective)}


def _safe(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def validate_plan(plan: List[Dict[str, Any]], optimizer: MultiPeriodOptimizer) -> None:
    """Re-check the solver's answer against the FPL rules, week by week."""
    previous_ids: Optional[set] = optimizer.initial_squad or None
    ft = optimizer.ft0
    windows_used: Dict[Tuple[str, int, int], int] = {}

    for index, week in enumerate(plan):
        squad, t, chip = week["squad"], week["gameweek"], week["chip"]
        assert len(squad) == SQUAD_SIZE, f"GW{t}: {len(squad)} players"

        counts = squad["position"].value_counts().to_dict()
        for pos, quota in SQUAD_QUOTA.items():
            assert counts.get(pos, 0) == quota, f"GW{t}: {counts.get(pos, 0)} {pos}"

        starters = squad[squad["is_starter"]]
        assert len(starters) == STARTING_XI, f"GW{t}: {len(starters)} starters"
        for pos, (low, high) in FORMATION_BOUNDS.items():
            n = int((starters["position"] == pos).sum())
            assert low <= n <= high, f"GW{t}: illegal formation, {n} {pos}"

        assert squad["team"].value_counts().max() <= MAX_PER_CLUB, f"GW{t}: club quota"
        assert int(squad["is_captain"].sum()) == 1, f"GW{t}: captain count"
        assert week["bank"] >= -1e-6, f"GW{t}: negative bank {week['bank']:.2f}"
        assert len(week["persistent_ids"]) == SQUAD_SIZE, f"GW{t}: persistent squad size"

        if chip:
            window = next((w for w in optimizer.windows if w[0] == chip and w[1] <= t <= w[2]),
                          None)
            assert window is not None, f"GW{t}: {chip} played outside its window"
            windows_used[window] = windows_used.get(window, 0) + 1
            assert windows_used[window] == 1, f"GW{t}: {chip} copy {window} played twice"

        # The FT bank is checked directly, not just via hits. It is driven to its
        # bound by a tie-break reward rather than an equality constraint, so if
        # that ever fails to bind we want a loud failure, not a wrong number in
        # the report.
        assert week["free_transfers"] == ft, (
            f"GW{t}: FT bank reports {week['free_transfers']}, the rules give {ft}")

        # Transfer accounting must reconcile with the persistent squad delta.
        current_ids = set(week["persistent_ids"])
        free_week = chip in ("wildcard", "freehit")
        if previous_ids is not None:
            actual_in = len(current_ids - previous_ids)
            if chip == "freehit":
                assert actual_in == 0, f"GW{t}: Free Hit week changed the real squad"
            else:
                assert actual_in == week["n_transfers"], (
                    f"GW{t}: {actual_in} new players but {week['n_transfers']} transfers booked")
            expected_hits = 0 if free_week else max(0, week["n_transfers"] - ft)
            assert week["hits"] == expected_hits, (
                f"GW{t}: {week['hits']} hits booked, rules give {expected_hits} "
                f"({week['n_transfers']} transfers vs {ft} FT)")
        previous_ids = current_ids

        if index == 0 and optimizer.fresh_build:
            ft = 1                       # a fresh build does not spend FTs
        elif free_week:
            ft = min(MAX_FREE_TRANSFERS, ft + 1)
        else:
            ft = min(MAX_FREE_TRANSFERS, max(0, ft - week["n_transfers"]) + 1)

    LOGGER.info("Plan validated: squad legality, transfer accounting, FT bank and chips "
                "all consistent.")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def print_plan(result: Dict[str, Any], config: TransferConfig) -> None:
    plan = result["plan"]
    width = 84

    print("\n" + "=" * width)
    print(f"{config.horizon}-GAMEWEEK TRANSFER PLAN".center(width))
    print(f"decay {config.decay} per week · hits at -{HIT_COST:.0f} · "
          f"bench weight {config.bench_weight}".center(width))
    print("=" * width)

    build_week = plan[0]["gameweek"] if config.wildcard else None

    for week in plan:
        squad = week["squad"]
        starters = squad[squad["is_starter"]].sort_values(
            "position", key=lambda c: c.map({p: n for n, p in enumerate(POSITIONS)}))
        captain = squad[squad["is_captain"]].iloc[0]
        is_build = week["gameweek"] == build_week

        print(f"\n{'─' * width}")
        # On a fresh build there is nothing to transfer from, so a free-transfer
        # count is meaningless — showing "FT 5" there just invites confusion.
        ft_label = "  build" if is_build else f"FT {week['free_transfers']}"
        header = (f"GW{week['gameweek']}   xPts {week['net_points']:.2f}"
                  f"   ({starters['position'].value_counts().get('DEF', 0)}-"
                  f"{starters['position'].value_counts().get('MID', 0)}-"
                  f"{starters['position'].value_counts().get('FWD', 0)})"
                  f"   {ft_label}   bank {week['bank']:.1f}"
                  f"   value {week['squad_value']:.1f}")
        print(header)

        chip = week.get("chip")
        if chip:
            note = {
                "wildcard": "unlimited free transfers this week",
                "freehit": "a one-week squad; your real squad returns next week",
                "bboost": f"bench scores too (+{week.get('bench_boost_points', 0):.2f})",
                "3xc": "captain scores triple",
            }[chip]
            print(f"   CHIP  {CHIP_LABELS[chip]}: {note}")

        if is_build:
            print(f"   Initial 15-man squad — {SQUAD_SIZE} free picks, no hits.")
        elif chip == "freehit":
            for out_name, in_name in zip(week["transfers_out"], week["transfers_in"]):
                print(f"   OUT  {out_name:<28} ->  IN  {in_name}   (this week only)")
        elif week["transfers_out"] or week["transfers_in"]:
            for out_name, in_name in zip(week["transfers_out"] + [""] * 5,
                                         week["transfers_in"] + [""] * 5):
                if not out_name and not in_name:
                    break
                print(f"   OUT  {out_name:<28} ->  IN  {in_name}")
            if week["hits"]:
                print(f"   {week['n_transfers']} transfers vs {week['free_transfers']} FT"
                      f"  ->  {week['hits']} hit(s), -{week['hit_cost']:.0f} pts")
            else:
                print(f"   {week['n_transfers']} transfer(s), all free")
        else:
            print("   No transfers — rolling the free transfer.")

        print(f"   XI: " + ", ".join(
            f"{r.player_name}{'(C)' if r.is_captain else ''}"
            for r in starters.itertuples()))
        print(f"   Captain {captain['player_name']} (+{captain['predicted_points']:.2f})")

    print("\n" + "=" * width)
    raw = sum(w["net_points"] for w in plan)
    discounted = sum(w["discount"] * w["net_points"] for w in plan)
    total_hits = sum(w["hits"] for w in plan)
    total_transfers = sum(w["n_transfers"] for w in plan)
    print(f"  Horizon total (undiscounted)   {raw:.2f} pts")
    print(f"  Horizon total (discounted)     {discounted:.2f} pts")
    print(f"  Transfers                      {total_transfers} across "
          f"{len(plan)} gameweeks, {total_hits} hit(s) = -{HIT_COST * total_hits:.0f} pts")
    print(f"  Final bank                     {plan[-1]['bank']:.1f}")
    print("\n  Only GW{} is a commitment. Re-run each week: form, prices and injuries all"
          .format(plan[0]["gameweek"]))
    print("  move, and the later weeks are structure, not instructions.")
    print("=" * width)


def chip_calendar_weeks(config: TransferConfig, client) -> int:
    """Gameweeks to project for the chip calendar: up to the last open copy's expiry."""
    if not (config.chip_windows and config.chip_calendar and client is not None
            and not config.projections_csv):
        return config.horizon
    events = client.fetch_bootstrap().get("events", [])
    upcoming = next((int(e["id"]) for e in events if e.get("is_next")), None)
    if upcoming is None:
        return config.horizon
    horizon_end = upcoming + config.horizon - 1
    last = max([horizon_end] + [w[2] for w in config.chip_windows
                                if w[1] <= horizon_end and w[2] >= upcoming])
    return last - upcoming + 1


def plan_to_frame(plan: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the plan for CSV export."""
    rows = []
    for week in plan:
        for row in week["squad"].itertuples(index=False):
            rows.append({
                "gameweek": week["gameweek"],
                "player_id": row.player_id,
                "player_name": row.player_name,
                "position": row.position,
                "team": row.team,
                "cost": row.cost,
                "predicted_points": row.predicted_points,
                "is_starter": row.is_starter,
                "is_captain": row.is_captain,
                "fixtures": getattr(row, "fixtures", ""),
                "gw_chip": week.get("chip") or "",
                "gw_transfers": week["n_transfers"],
                "gw_hits": week["hits"],
                "gw_free_transfers": week["free_transfers"],
                "gw_bank": week["bank"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(config: TransferConfig) -> Dict[str, Any]:
    client = None
    if config.entry_id and not config.projections_csv:
        from fpl_live_inference import LiveConfig, make_client
        client = make_client(LiveConfig(model_dir=config.model_dir))

    squad, bank, free, purchase, chips_used = load_squad_state(config, client)
    if config.entry_id and client is not None and config.use_chips:
        config.chip_windows = available_chip_windows(chip_rules(client.fetch_bootstrap()),
                                                     chips_used)
        LOGGER.info("Chips still available: %s", ", ".join(
            f"{CHIP_LABELS[n]} (GW{a}-{b})" for n, a, b in config.chip_windows) or "none")

    # Fresh start / wildcard detection. No squad at t=0 means there is nothing
    # to transfer FROM, so week 1 is an unconstrained legal build: x[i,1] is set
    # directly subject to budget and squad rules, no hits are possible, and the
    # FT bank opens at 1 for week 2.
    if not squad:
        if not config.wildcard:
            LOGGER.info("No current squad supplied — treating GW1 as a fresh build "
                        "(wildcard semantics: free selection, no hits, FT=1 from week 2).")
        config.wildcard = True
        bank = config.budget
        free = MAX_FREE_TRANSFERS

    # With chips to time, project past the horizon to each open copy's expiry
    # so the chip calendar can compare this week with every remaining one.
    calendar = None
    calendar_weeks = chip_calendar_weeks(config, client)
    if calendar_weeks > config.horizon:
        extended = load_projections(replace(config, horizon=calendar_weeks), squad)
        calendar = apply_selling_prices(extended, purchase)
        first = int(calendar["gameweek"].min())
        projections = calendar[calendar["gameweek"] < first + config.horizon].copy()
    else:
        projections = apply_selling_prices(load_projections(config, squad), purchase)
    result = plan_transfers(projections, squad, bank, free, config, calendar=calendar)

    if config.output_path:
        path = Path(config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plan_to_frame(result["plan"]).to_csv(path, index=False)
        LOGGER.info("Wrote plan to %s", path)

    return result


def apply_selling_prices(projections: pd.DataFrame,
                         purchase: Dict[int, float]) -> pd.DataFrame:
    """
    Value owned players at what selling them would actually raise.

    The solver uses one price per player for both buying and selling. For an
    owned player the relevant number is his selling price, which keeps only
    half of any rise since purchase; using today's price overstates the budget.
    """
    if not purchase:
        return projections
    projections = projections.copy()
    owned = projections["player_id"].isin(purchase)
    today = projections.loc[owned].drop_duplicates("player_id").set_index("player_id")["cost"]
    sale = {int(pid): selling_value(purchase[int(pid)], cost) for pid, cost in today.items()}
    projections.loc[owned, "cost"] = projections.loc[owned, "player_id"].map(sale)
    LOGGER.info("%d owned player(s) valued at selling price: %.1f to sell vs %.1f at "
                "today's prices.", len(sale), sum(sale.values()), float(today.sum()))
    return projections


# --------------------------------------------------------------------------- #
# Chip calendar: when is each chip's best remaining week?
# --------------------------------------------------------------------------- #

# Chips the calendar times. A Wildcard's worth depends on how many weeks the
# rebuilt squad lasts, which a one-week score cannot see, so it keeps the fixed
# reserve.
CALENDAR_CHIPS = ("freehit", "bboost", "3xc")


def best_lineup(squad_week: pd.DataFrame) -> Tuple[float, float, float]:
    """
    (XI points, bench points, captain points) for a squad's best legal XI.

    Greedy is exact here: fill each position's formation minimum with its best
    players, then the remaining slots with the best of the rest, never passing a
    position's maximum.
    """
    by_position = {pos: sorted(block["predicted_points"], reverse=True)
                   for pos, block in squad_week.groupby("position")}
    xi: List[float] = []
    taken = {pos: 0 for pos in FORMATION_BOUNDS}
    for pos, (low, _high) in FORMATION_BOUNDS.items():
        picks = by_position.get(pos, [])[:low]
        xi += picks
        taken[pos] = len(picks)
    rest = sorted(((pts, pos) for pos, values in by_position.items()
                   for pts in values[taken.get(pos, 0):]), reverse=True)
    for pts, pos in rest:
        if len(xi) == STARTING_XI:
            break
        if taken[pos] < FORMATION_BOUNDS[pos][1]:
            xi.append(pts)
            taken[pos] += 1
    total = float(squad_week["predicted_points"].sum())
    return float(sum(xi)), total - float(sum(xi)), float(max(xi, default=0.0))


def free_hit_score(week: pd.DataFrame, budget: float, pool: int = 12) -> Optional[float]:
    """XI + captain of the best 15 within budget for one week, on a slimmed pool."""
    from fpl_optimizer import OptimizerConfig, SquadOptimizer

    keep = set()
    for _, block in week.groupby("position"):
        keep.update(block.nlargest(pool, "predicted_points")["player_id"])
        keep.update(block.nsmallest(5, "cost")["player_id"])
    for _, block in week.groupby("team"):
        keep.update(block.nsmallest(2, "cost")["player_id"])
    players = week[week["player_id"].isin(keep)].reset_index(drop=True)
    try:
        optimizer = SquadOptimizer(players, OptimizerConfig(budget=budget, bench_weight=0.0))
        optimizer.build()
        optimizer.solve()
        squad = optimizer.extract()
    except (RuntimeError, AssertionError) as exc:
        LOGGER.debug("Free Hit estimate failed: %s", exc)
        return None
    starters = squad[squad["is_starter"]]
    return float(starters["predicted_points"].sum()
                 + squad.loc[squad["is_captain"], "predicted_points"].sum())


def chip_calendar(calendar: pd.DataFrame, squad: Sequence[int], budget: float,
                  windows: Sequence[Tuple[str, int, int]], horizon_end: int,
                  bench_weight: float) -> Dict[Tuple[str, int, int], float]:
    """
    Reserve value for each unused chip copy: what its best remaining week is worth.

    Every remaining week up to the copy's expiry is scored for the current
    squad (Bench Boost: its bench; Triple Captain: its best captain; Free Hit:
    how much the best one-week 15 beats it). A small assignment then gives each
    copy its best week, one chip per gameweek, so four chips cannot all claim
    the same few weeks. A copy whose best week lies beyond the plan's horizon
    is reserved at that week's value, so it is only played now if now is at
    least as good; a copy whose best week is inside the horizon is left for the
    plan to place. Form is frozen at today's, so later weeks differ mainly by
    fixture, blanks and doubles.
    """
    windows = [tuple(w) for w in windows if w[0] in CALENDAR_CHIPS]
    weeks = sorted(int(g) for g in calendar["gameweek"].unique())
    if not windows or not weeks:
        return {}

    meta = calendar.drop_duplicates("player_id").set_index("player_id")
    owned = [int(i) for i in squad if int(i) in meta.index]
    needs_fh = any(w[0] == "freehit" for w in windows)
    values: Dict[Tuple[str, int], float] = {}
    for u in weeks:
        week = calendar[calendar["gameweek"] == u]
        mine = (week[week["player_id"].isin(owned)][["player_id", "position", "predicted_points"]]
                .groupby(["player_id", "position"], as_index=False)["predicted_points"].sum())
        blank = [i for i in owned if i not in set(mine["player_id"])]
        if blank:
            mine = pd.concat([mine, pd.DataFrame({
                "player_id": blank, "position": meta.loc[blank, "position"].to_numpy(),
                "predicted_points": 0.0})], ignore_index=True)
        xi, bench, captain = best_lineup(mine)
        values["bboost", u] = (1 - bench_weight) * bench
        values["3xc", u] = captain
        if needs_fh and any(w[0] == "freehit" and w[1] <= u <= w[2] for w in windows):
            best = free_hit_score(week.groupby(["player_id", "player_name", "position", "team",
                                                "cost"], as_index=False)["predicted_points"].sum(),
                                  budget)
            values["freehit", u] = max(0.0, best - (xi + captain)) if best is not None else 0.0

    problem = pulp.LpProblem("chip_calendar", pulp.LpMaximize)
    pick = {(k, u): pulp.LpVariable(f"pick_{k}_{u}", cat=pulp.LpBinary)
            for k, (name, start, stop) in enumerate(windows) for u in weeks
            if start <= u <= stop and (name, u) in values}
    problem += pulp.lpSum(values[windows[k][0], u] * v for (k, u), v in pick.items())
    for k in range(len(windows)):
        problem += pulp.lpSum(v for (j, _u), v in pick.items() if j == k) <= 1
    for u in weeks:
        problem += pulp.lpSum(v for (_k, w), v in pick.items() if w == u) <= 1
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    reserve: Dict[Tuple[str, int, int], float] = {}
    for k, window in enumerate(windows):
        chosen = [u for (j, u), v in pick.items() if j == k and (v.value() or 0) > 0.5]
        best_week = chosen[0] if chosen else None
        reserve[window] = (values[window[0], best_week]
                           if best_week is not None and best_week > horizon_end else 0.0)
        LOGGER.info("Chip calendar: %s best in GW%s (%.1f)%s.", CHIP_LABELS[window[0]],
                    best_week, values.get((window[0], best_week), 0.0),
                    "" if reserve[window] else ", inside the horizon")
    return reserve


def plan_transfers(projections: pd.DataFrame, squad: Sequence[int], bank: float,
                   free: int, config: TransferConfig,
                   calendar: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Prune, check feasibility, build, solve and validate one H-week plan.

    Split out of `run()` so the season simulator can call it with in-memory
    projections. An empty `squad` with `config.wildcard` set is a fresh build.
    `calendar` holds projections past the horizon, up to the chips' expiry;
    given it, chip reserves come from the chip calendar instead of the fixed
    values.
    """
    if calendar is not None and config.chip_windows and squad:
        horizon_end = int(projections["gameweek"].max())
        owned = calendar[calendar["player_id"].isin(squad)].drop_duplicates("player_id")
        config = replace(config, chip_window_reserve=chip_calendar(
            calendar, squad, bank + float(owned["cost"].sum()), config.chip_windows,
            horizon_end, config.bench_weight))

    # Widen the pool until the problem is provably feasible, rather than letting
    # CBC discover it and report a bare "Infeasible".
    pruned = None
    for attempt, pool in enumerate([config.candidate_pool,
                                    config.candidate_pool * 2,
                                    config.candidate_pool * 4]):
        pruned = prune_candidates(projections, squad, pool)
        problems = diagnose_infeasibility(pruned, squad, bank, config)
        if not problems:
            break
        if attempt < 2:
            LOGGER.warning("Pool of %d per position is infeasible (%s) — widening to %d.",
                           pool, "; ".join(problems), pool * 2)
        else:
            raise RuntimeError(
                "Problem is infeasible even with a widened pool: " + "; ".join(problems)
            )

    optimizer = MultiPeriodOptimizer(pruned, squad, bank, free, config)
    optimizer.build()
    optimizer.solve()
    return optimizer.extract()


def parse_chip_reserve(items: Sequence[str]) -> Dict[str, float]:
    """['bboost=8', '3xc=5'] -> {'bboost': 8.0, '3xc': 5.0}."""
    reserve: Dict[str, float] = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in CHIP_NAMES or not value:
            raise SystemExit(f"--chip-reserve wants CHIP=PTS with CHIP in {CHIP_NAMES}: {item}")
        reserve[name] = float(value)
    return reserve


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL V3 multi-period transfer solver.")
    parser.add_argument("-i", "--input", default="fpl_ml_features.csv")
    parser.add_argument("-d", "--model-dir", default="models")
    parser.add_argument("--medians", default="fpl_position_medians.json")
    parser.add_argument("--projections-csv", default=None,
                        help="Precomputed player x gameweek projections (skips the API)")

    squad_group = parser.add_argument_group("current squad (t=0)")
    squad_group.add_argument("--team-id", type=int, default=None,
                             help="Your FPL entry id; pulls picks and bank from the API")
    squad_group.add_argument("--squad-json", default=None,
                             help='JSON: {"squad": [...], "bank": 1.4, "free_transfers": 2}')
    squad_group.add_argument("--squad", default="",
                             help="Comma-separated element ids of your current 15")
    squad_group.add_argument("--bank", type=float, default=0.0, help="Cash in the bank (£m)")
    squad_group.add_argument("--free-transfers", type=int, default=None,
                             help="Override the free-transfer count (derived from "
                                  "--team-id history, else 1)")
    squad_group.add_argument("--wildcard", action="store_true",
                             help="Ignore the current squad; week 1 is a free selection")

    solver_group = parser.add_argument_group("solver")
    solver_group.add_argument("-H", "--horizon", type=int, default=5)
    solver_group.add_argument("--decay", type=float, default=0.85,
                              help="Per-week discount on future points (default: 0.85)")
    solver_group.add_argument("--bench-weight", type=float, default=0.10)
    solver_group.add_argument("-b", "--budget", type=float, default=100.0,
                              help="Only used with --wildcard")
    solver_group.add_argument("--max-transfers-per-week", type=int, default=3)
    solver_group.add_argument("--candidate-pool", type=int, default=40,
                              help="Top-N players per position kept for the solver")
    solver_group.add_argument("--time-limit", type=int, default=120)
    solver_group.add_argument("--solver-msg", action="store_true")
    solver_group.add_argument("--no-chips", action="store_true",
                              help="Plan without Wildcard/Free Hit/Bench Boost/Triple Captain")
    solver_group.add_argument("--chip-calendar", action="store_true",
                              help="Experimental: hold each chip for its best remaining "
                                   "week instead of the fixed reserves (scored worse in "
                                   "the season replay)")
    solver_group.add_argument("--chip-reserve", nargs="+", default=[], metavar="CHIP=PTS",
                              help="Points a chip must beat to be played, e.g. bboost=8 3xc=5 "
                                   f"(defaults: {DEFAULT_CHIP_RESERVE})")

    parser.add_argument("--allow-status", nargs="+", default=["a"])
    parser.add_argument("--min-chance", type=int, default=0)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ensure_printable_console()
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    config = TransferConfig(
        features_path=Path(args.input),
        model_dir=Path(args.model_dir),
        medians_path=Path(args.medians),
        projections_csv=Path(args.projections_csv) if args.projections_csv else None,
        squad_json=Path(args.squad_json) if args.squad_json else None,
        entry_id=args.team_id,
        current_squad=[int(i) for i in args.squad.split(",") if i.strip()],
        current_bank=args.bank,
        free_transfers=args.free_transfers,
        wildcard=args.wildcard,
        horizon=args.horizon,
        decay=args.decay,
        bench_weight=args.bench_weight,
        budget=args.budget,
        max_transfers_per_week=args.max_transfers_per_week,
        candidate_pool=args.candidate_pool,
        team_rating_window=10,
        allow_status=tuple(args.allow_status),
        min_chance=args.min_chance,
        time_limit=args.time_limit,
        solver_msg=args.solver_msg,
        output_path=Path(args.output) if args.output else None,
        use_chips=not args.no_chips, chip_calendar=args.chip_calendar,
        chip_reserve={**DEFAULT_CHIP_RESERVE, **parse_chip_reserve(args.chip_reserve)},
    )

    try:
        result = run(config)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except AssertionError as exc:
        LOGGER.error("Plan violated an FPL rule: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    print_plan(result, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
