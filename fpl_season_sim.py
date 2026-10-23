"""
FPL ML Pipeline — Season Simulator
==================================

Replays a completed season deadline by deadline and plays it as a real manager
would, then scores the squads it picked against what actually happened.

At every deadline t the simulator only uses what was knowable at that moment:

    form          each player's rolling features as of deadline t, frozen
                  across the planning horizon (GW t+2 never sees GW t+1)
    difficulty    team attack/defence ratings as of deadline t, recombined for
                  every future fixture via the shared derive_fdr_matchups()
    odds          bookmaker prices for GW t's fixtures only; later gameweeks
                  are not priced yet at the deadline, so they go in as NaN
    prices        the price at deadline t, held static across the horizon
    models        trained on prior seasons only (or walk-forward with
                  --retrain-every, on prior seasons + gameweeks < t)
    medians       imputation medians fitted on the training seasons only

Then it executes week 1 of the H-week plan from `fpl_transfers.plan_transfers`
and scores that gameweek with the real FPL rules: automatic substitutions,
vice-captain fallback, -4 per hit, the free-transfer bank capped at 5, and the
selling-price rule (you keep half the profit on a player, rounded down).

Three projectors are run through the same solver, so the comparison isolates
the forecasts:

    model    the two-stage XGBoost model
    roll3    each player's 3-gameweek rolling points average — the baseline
             the ML has to beat to be worth having
    oracle   actual points, with hindsight. Not achievable; it is the ceiling
             the solver and transfer rules allow

Known simplifications
---------------------
* No chips. Real managers have two Wildcards, Free Hits, Bench Boosts and
  Triple Captains; the bot has none, so it plays with a handicap.
* No team news. Availability is "registered and at a club"; an injured player
  is only discounted through his recent minutes. Real managers see press
  conferences.
* The fixture schedule is the one that was played. A fixture that was later
  postponed is therefore known at its final date, a mild form of hindsight.
* Prices are those recorded for each gameweek in the archive.

Usage
-----
    python fpl_season_sim.py                                  # all of 2025-26
    python fpl_season_sim.py --end-gw 10 --projectors model roll3
    python fpl_season_sim.py --retrain-every 4 --horizon 4
    python fpl_season_sim.py --seeds 1 2 3 4 5                # measure run-to-run noise
    python fpl_season_sim.py --benchmark-csv averages.csv     # gameweek,average_points
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fpl_features import (
    FDR_BASE_FEATURES,
    FeatureConfig,
    build_features,
    build_team_match_table,
    compute_team_ratings,
    derive_fdr_matchups,
)
from fpl_odds import ODDS_PREFIX
from fpl_optimizer import FORMATION_BOUNDS, POSITIONS, predict_points
from fpl_transfers import HIT_COST, MAX_FREE_TRANSFERS, TransferConfig, plan_transfers

LOGGER = logging.getLogger("fpl.sim")

PROJECTORS = ("model", "roll3", "oracle")

@dataclass
class SimConfig:
    history_path: Path = Path("fpl_combined_history.csv")
    season: str = "2025-26"
    train_seasons: Sequence[str] = ("2023-24", "2024-25")
    start_gw: int = 1
    end_gw: Optional[int] = None
    projectors: Sequence[str] = PROJECTORS
    retrain_every: int = 0            # 0 = train once on train_seasons
    horizon: int = 4
    decay: float = 0.85
    bench_weight: float = 0.10
    candidate_pool: int = 25
    max_transfers_per_week: int = 2
    time_limit: int = 60
    budget: float = 100.0
    team_rating_window: int = 10
    odds: bool = True
    seeds: Sequence[int] = (42,)
    ensemble_size: int = 1            # seeds averaged inside each model
    random_state: int = 42            # set per run from `seeds`
    benchmark_csv: Optional[Path] = None
    output_dir: Path = Path("sim_results")
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# FPL money and scoring rules
# --------------------------------------------------------------------------- #

def to_tenths(price: float) -> int:
    """Prices are tenths of a million in the game; integer maths avoids drift."""
    return int(round(float(price) * 10))


def selling_price(purchase: int, current: int) -> int:
    """FPL keeps half the profit, rounded down to £0.1m. Losses are taken in full."""
    if current <= purchase:
        return current
    return purchase + (current - purchase) // 2


def next_free_transfers(free: int, made: int) -> int:
    """Unused transfers roll over, one more arrives, the bank is capped at 5."""
    return min(MAX_FREE_TRANSFERS, max(0, free - made) + 1)


def formation_is_legal(positions: Sequence[str]) -> bool:
    counts = pd.Series(list(positions)).value_counts()
    return all(low <= int(counts.get(pos, 0)) <= high
               for pos, (low, high) in FORMATION_BOUNDS.items())


def score_gameweek(squad: pd.DataFrame, actual: Dict[int, Tuple[float, float]]) -> Dict[str, Any]:
    """
    Score one gameweek the way FPL does.

    `squad` carries player_id, position, is_starter, is_captain and
    predicted_points. The bench is ordered by predicted points (keeper
    separately), the vice-captain is the next-best projected starter, and any
    starter with zero minutes is auto-subbed by the first bench player who
    played and keeps the formation legal.

    `actual` maps player_id -> (points, minutes), summed over a double gameweek.
    A player with no fixture scores (0, 0).
    """
    def points(pid: int) -> float:
        return actual.get(pid, (0.0, 0.0))[0]

    def played(pid: int) -> bool:
        return actual.get(pid, (0.0, 0.0))[1] > 0

    starters = squad[squad["is_starter"]].sort_values("predicted_points", ascending=False)
    bench = squad[~squad["is_starter"]]
    bench_keeper = bench[bench["position"] == "GKP"]["player_id"].tolist()
    bench_outfield = (bench[bench["position"] != "GKP"]
                      .sort_values("predicted_points", ascending=False)["player_id"].tolist())

    position = dict(zip(squad["player_id"], squad["position"]))
    xi = starters["player_id"].tolist()
    captain = int(squad.loc[squad["is_captain"], "player_id"].iloc[0])
    vice = next((pid for pid in xi if pid != captain), None)

    subs: List[Tuple[int, int]] = []
    # FPL processes starters in squad order: keeper, then outfield.
    for pid in sorted(xi, key=lambda p: POSITIONS.index(position[p])):
        if played(pid):
            continue
        if position[pid] == "GKP":
            replacement = next((b for b in bench_keeper if played(b)), None)
            if replacement is not None:
                bench_keeper.remove(replacement)
        else:
            replacement = None
            for candidate in bench_outfield:
                trial = [position[p] for p in xi if p != pid] + [position[candidate]]
                if played(candidate) and formation_is_legal(trial):
                    replacement = candidate
                    break
            if replacement is not None:
                bench_outfield.remove(replacement)
        if replacement is not None:
            xi = [replacement if p == pid else p for p in xi]
            subs.append((pid, replacement))

    if played(captain):
        armband = captain
    elif vice is not None and played(vice) and vice in xi:
        armband = vice
    else:
        armband = None

    xi_points = float(sum(points(p) for p in xi))
    captain_points = points(armband) if armband is not None else 0.0
    return {
        "xi_points": xi_points,
        "captain_points": captain_points,
        "gross_points": xi_points + captain_points,
        "final_xi": xi,
        "captain_id": captain,
        "armband_id": armband,
        "auto_subs": subs,
    }


# --------------------------------------------------------------------------- #
# Data available at each deadline
# --------------------------------------------------------------------------- #

class SeasonData:
    """
    Everything the simulator needs about one season, indexed by deadline.

    `features` is the causal Phase 2 frame: a row's rolling columns describe
    form going INTO that fixture. So a player's first row at or after gameweek
    t is his form as of deadline t, whichever gameweek it belongs to.
    """

    def __init__(self, features: pd.DataFrame, season: str, team_rating_window: int) -> None:
        self.season = season
        frame = features[features["season"] == season].copy()
        if frame.empty:
            raise RuntimeError(f"Season {season} not in the feature frame.")
        frame["gameweek"] = frame["gameweek"].astype(int)
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
        self.rows = frame.sort_values(["player_id", "gameweek", "kickoff_time"],
                                      kind="mergesort").reset_index(drop=True)
        self.gameweeks = sorted(self.rows["gameweek"].unique())

        span = self.rows.groupby("player_id")["gameweek"].agg(["min", "max"])
        self.first_gw = span["min"].to_dict()
        self.last_gw = span["max"].to_dict()

        last = self.rows.groupby("player_id").tail(1).set_index("player_id")
        self.static = last[["player_name", "position", "team", "gw_cost"]]

        totals = (self.rows.groupby(["player_id", "gameweek"])
                  .agg(points=("total_points", "sum"), minutes=("minutes", "sum")))
        self.actual: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for (pid, gw), row in totals.iterrows():
            self.actual.setdefault(int(gw), {})[int(pid)] = (float(row.points),
                                                             float(row.minutes))

        team_names = self.rows.dropna(subset=["team_id"]).drop_duplicates("team_id")
        self.team_by_id = dict(zip(team_names["team_id"].astype(int), team_names["team"]))

        # Ratings over EVERY season, so a club's form carries across the summer
        # exactly as it does in Phase 2.
        ratings = compute_team_ratings(build_team_match_table(features), team_rating_window)
        ratings = ratings[ratings["season"] == season].copy()
        ratings["gameweek"] = ratings["gameweek"].astype(int)
        ratings["kickoff_time"] = pd.to_datetime(ratings["kickoff_time"], utc=True,
                                                 errors="coerce")
        self.ratings = ratings.sort_values(["team", "kickoff_time"], kind="mergesort")

    def available(self, t: int) -> List[int]:
        """Registered at a Premier League club at deadline t."""
        return [pid for pid in self.first_gw
                if self.first_gw[pid] <= t <= self.last_gw[pid]]

    def form_at(self, t: int, players: Sequence[int]) -> pd.DataFrame:
        """One row per player: features as of deadline t (his first row >= t)."""
        ahead = self.rows[(self.rows["gameweek"] >= t) & self.rows["player_id"].isin(players)]
        return ahead.groupby("player_id").head(1).set_index("player_id")

    def price_at(self, t: int) -> Dict[int, int]:
        """Latest recorded price at or before t, in tenths; first price for newcomers."""
        rows = self.rows[self.rows["gameweek"] <= t]
        latest = rows.groupby("player_id").tail(1).set_index("player_id")["gw_cost"]
        prices = {int(pid): to_tenths(v) for pid, v in latest.items() if pd.notna(v)}
        first = self.rows.groupby("player_id").head(1).set_index("player_id")["gw_cost"]
        for pid, value in first.items():
            if int(pid) not in prices and pd.notna(value):
                prices[int(pid)] = to_tenths(value)
        return prices

    def team_ratings_at(self, t: int) -> pd.DataFrame:
        """Each club's rating going into its first match at or after deadline t."""
        ahead = self.ratings[self.ratings["gameweek"] >= t]
        return ahead.groupby("team").head(1).set_index("team")

    def fixtures_between(self, start: int, end: int, players: Sequence[int]) -> pd.DataFrame:
        mask = (self.rows["gameweek"].between(start, end)
                & self.rows["player_id"].isin(players))
        return self.rows.loc[mask]


# --------------------------------------------------------------------------- #
# Projections at a deadline
# --------------------------------------------------------------------------- #

def deadline_feature_rows(data: SeasonData, t: int, horizon_end: int,
                          players: Sequence[int]) -> pd.DataFrame:
    """
    One row per player per fixture in [t, horizon_end], as seen at deadline t.

    Form columns come from the player's form-at-t row, so GW t+2 is projected
    from the same information as GW t. Only the fixture-specific columns change:
    venue, opponent, the FDR matchup rebuilt from ratings frozen at t, and the
    odds, which exist for GW t alone.
    """
    form = data.form_at(t, players)
    fixtures = data.fixtures_between(t, horizon_end, form.index)
    ratings = data.team_ratings_at(t)

    odds = [c for c in fixtures.columns if c.startswith(ODDS_PREFIX)]
    fixture_cols = ["player_id", "gameweek", "kickoff_time", "was_home", "team",
                    "opponent_team_id", "total_points", "minutes", *odds]
    out = fixtures[fixture_cols].merge(
        form.drop(columns=[c for c in fixture_cols if c != "player_id" and c in form.columns]
                  + [c for c in FDR_BASE_FEATURES if c in form.columns]),
        left_on="player_id", right_index=True, how="left")

    opponent = out["opponent_team_id"].map(
        lambda v: data.team_by_id.get(int(v)) if pd.notna(v) else None)

    def rating(names: pd.Series, column: str, default: float) -> np.ndarray:
        lookup = ratings[column].to_dict()
        return np.array([lookup.get(n, default) for n in names], dtype=float)

    out["fdr_team_attack"] = rating(out["team"], "attack_rating", np.nan)
    out["fdr_team_defence"] = rating(out["team"], "defence_rating", np.nan)
    out["fdr_opp_attack"] = rating(opponent, "attack_rating", np.nan)
    out["fdr_opp_defence"] = rating(opponent, "defence_rating", np.nan)
    out["fdr_opp_rating_known"] = rating(opponent, "rating_known", 0)
    out.loc[out["gameweek"] > t, odds] = np.nan
    return derive_fdr_matchups(out)


def project(data: SeasonData, t: int, horizon_end: int, players: Sequence[int],
            prices: Dict[int, int], bundles: Optional[Dict[str, Dict[str, Any]]],
            projectors: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """
    Projections for every requested projector, in the shape plan_transfers wants:
    player_id, gameweek, predicted_points, cost, position, team, player_name.
    """
    rows = deadline_feature_rows(data, t, horizon_end, players)
    # Price as of the deadline, never a later gameweek's: price rises are partly
    # a reaction to results that have not happened yet.
    rows["gw_cost"] = rows["player_id"].map(lambda p: prices.get(int(p), np.nan) / 10)

    scored: Dict[str, pd.Series] = {}
    if "model" in projectors:
        if bundles is None:
            raise RuntimeError("The model projector needs trained bundles.")
        predicted = predict_points(rows.reset_index(drop=True), bundles)
        if len(predicted) != len(rows):
            raise RuntimeError("Model scoring dropped rows; every position needs a model.")
        scored["model"] = predicted["predicted_points"]
    if "roll3" in projectors:
        scored["roll3"] = pd.to_numeric(rows["total_points_roll_3"], errors="coerce").fillna(0)
    if "oracle" in projectors:
        scored["oracle"] = pd.to_numeric(rows["total_points"], errors="coerce").fillna(0)

    out: Dict[str, pd.DataFrame] = {}
    for name, values in scored.items():
        frame = rows[["player_id", "gameweek"]].assign(predicted_points=values.to_numpy())
        frame = frame.groupby(["player_id", "gameweek"], as_index=False)["predicted_points"].sum()
        out[name] = frame
    return out


def attach_meta(projection: pd.DataFrame, data: SeasonData, form: pd.DataFrame,
                costs: Dict[int, int], owned: Sequence[int], t: int) -> pd.DataFrame:
    """
    Add cost/position/team/name, and make every owned player representable.

    An owned player with no fixture in the horizon (blank, or sold abroad) still
    occupies a squad slot, so he gets a 0-point row at t and can be sold.
    """
    missing = [pid for pid in owned if pid not in set(projection["player_id"])]
    if missing:
        projection = pd.concat([projection, pd.DataFrame(
            {"player_id": missing, "gameweek": t, "predicted_points": 0.0})],
            ignore_index=True)

    def meta(pid: int, column: str):
        if pid in form.index:
            return form.at[pid, column]
        return data.static.at[pid, column]

    ids = projection["player_id"].astype(int)
    projection["player_id"] = ids
    projection["cost"] = ids.map(lambda p: costs[p] / 10)
    for column in ("position", "team", "player_name"):
        projection[column] = [meta(p, column) for p in ids]
    return projection


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def train_bundles(features: pd.DataFrame, config: SimConfig,
                  before_gw: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """
    Train the two-stage models on data knowable at deadline `before_gw`.

    before_gw=None trains on the training seasons alone. Otherwise the
    simulated season's gameweeks < before_gw are added, walk-forward.
    """
    from fpl_train import (
        TrainConfig,
        add_position_indicators,
        assert_no_post_match_features,
        select_feature_columns,
        train_position_model,
    )

    frame = features[features["total_points"].notna()].copy()
    frame["played"] = (pd.to_numeric(frame["minutes"], errors="coerce").fillna(0) > 0).astype(int)
    frame, position_columns = add_position_indicators(frame)
    feature_columns = select_feature_columns(frame, position_columns)
    assert_no_post_match_features(feature_columns)

    in_train = frame["season"].isin(config.train_seasons)
    season = frame["season"] == config.season
    if before_gw is not None:
        in_train |= season & (frame["gameweek"] < before_gw)
    train = frame[in_train]
    # train_position_model insists on a non-empty test slice for its report;
    # it never fits on it. The deadline's own gameweek is the natural choice.
    probe_gw = before_gw if before_gw is not None else config.start_gw
    test = frame[season & (frame["gameweek"] == probe_gw)]

    train_config = TrainConfig(train_seasons=tuple(config.train_seasons),
                               test_season=config.season, permutation_repeats=0,
                               random_state=config.random_state,
                               ensemble_size=config.ensemble_size)
    bundles: Dict[str, Dict[str, Any]] = {}
    for position in POSITIONS:
        result = train_position_model(position, train, test, feature_columns, train_config)
        if result is None:
            raise RuntimeError(f"Could not train a {position} model.")
        bundles[position] = {"model": result["model"], "features": result["features"]}
    LOGGER.info("Trained models on %d rows%s.", len(train),
                f" (incl. {config.season} GW<{before_gw})" if before_gw else "")
    return bundles


# --------------------------------------------------------------------------- #
# One manager, one season
# --------------------------------------------------------------------------- #

@dataclass
class Manager:
    """The state a real FPL manager carries between deadlines."""
    squad: Dict[int, int] = field(default_factory=dict)   # player_id -> purchase price
    bank: int = 0                                         # tenths of a million
    free_transfers: int = 1
    total_points: float = 0.0

    def sale_values(self, prices: Dict[int, int]) -> Dict[int, int]:
        return {pid: selling_price(paid, prices.get(pid, paid))
                for pid, paid in self.squad.items()}


def play_deadline(manager: Manager, projection: pd.DataFrame, t: int,
                  prices: Dict[int, int], data: SeasonData, config: SimConfig,
                  first_deadline: bool) -> Dict[str, Any]:
    """Plan, execute week 1 of the plan, and score gameweek t."""
    transfer_config = TransferConfig(
        horizon=config.horizon, decay=config.decay, bench_weight=config.bench_weight,
        budget=config.budget, max_transfers_per_week=config.max_transfers_per_week,
        candidate_pool=config.candidate_pool, time_limit=config.time_limit,
        wildcard=first_deadline,
    )
    if first_deadline:
        squad_ids, bank, free = [], config.budget, MAX_FREE_TRANSFERS
    else:
        squad_ids = list(manager.squad)
        bank, free = manager.bank / 10, manager.free_transfers

    result = plan_transfers(projection, squad_ids, bank, free, transfer_config)
    week = result["plan"][0]
    new_squad = week["squad"]
    new_ids = set(int(p) for p in new_squad["player_id"])

    sale = manager.sale_values(prices)
    sold = set(manager.squad) - new_ids
    bought = new_ids - set(manager.squad)
    if first_deadline:
        hits = 0
        manager.bank = to_tenths(config.budget) - sum(prices[p] for p in bought)
        manager.free_transfers = 1
    else:
        hits = max(0, len(bought) - manager.free_transfers)
        manager.bank += sum(sale[p] for p in sold) - sum(prices[p] for p in bought)
        manager.free_transfers = next_free_transfers(manager.free_transfers, len(bought))
    if manager.bank < 0:
        raise RuntimeError(f"GW{t}: bank went negative ({manager.bank / 10:.1f}).")

    for pid in sold:
        del manager.squad[pid]
    for pid in bought:
        manager.squad[pid] = prices[pid]

    scored = score_gameweek(new_squad, data.actual.get(t, {}))
    net = scored["gross_points"] - HIT_COST * hits
    manager.total_points += net

    name = data.static["player_name"].to_dict()
    return {
        "gameweek": t,
        "points": net,
        "gross_points": scored["gross_points"],
        "hits": hits,
        "transfers": len(bought) if not first_deadline else 0,
        "free_transfers_after": manager.free_transfers,
        "bank": manager.bank / 10,
        "squad_value": sum(manager.sale_values(prices).values()) / 10,
        "projected_xi": float(new_squad.loc[new_squad["is_starter"], "predicted_points"].sum()),
        "captain": name.get(scored["captain_id"], scored["captain_id"]),
        "captain_points": scored["captain_points"],
        "auto_subs": len(scored["auto_subs"]),
        "transfers_in": "; ".join(str(name.get(p, p)) for p in sorted(bought))
        if not first_deadline else "",
        "transfers_out": "; ".join(str(name.get(p, p)) for p in sorted(sold)),
        "total_points": manager.total_points,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_sim_features(config: SimConfig) -> pd.DataFrame:
    """Phase 2 in memory, with medians fitted on the training seasons only."""
    feature_config = FeatureConfig(
        input_path=config.history_path, medians_path=None,
        fit_seasons=tuple(config.train_seasons),
        team_rating_window=config.team_rating_window, odds=config.odds,
    )
    return build_features(feature_config)


def simulate(config: SimConfig, features: pd.DataFrame, data: SeasonData,
             projectors: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """One season for each projector, with models seeded by config.random_state."""
    end_gw = config.end_gw or max(data.gameweeks)
    deadlines = [g for g in data.gameweeks if config.start_gw <= g <= end_gw]
    if not deadlines:
        raise RuntimeError(f"No gameweeks in {config.start_gw}-{end_gw}.")

    bundles = None
    trained_for = deadlines[0]
    if "model" in projectors:
        bundles = train_bundles(features, config, None)

    managers = {name: Manager() for name in projectors}
    logs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in projectors}

    for index, t in enumerate(deadlines):
        started = time.perf_counter()
        if (bundles is not None and config.retrain_every > 0
                and t - trained_for >= config.retrain_every):
            bundles = train_bundles(features, config, t)
            trained_for = t

        horizon_end = min(t + config.horizon - 1, max(data.gameweeks))
        available = data.available(t)
        owned = set().union(*(m.squad for m in managers.values()))
        prices = data.price_at(t)
        projections = project(data, t, horizon_end, available, prices, bundles, projectors)
        form = data.form_at(t, list(set(available) | owned))

        for name in projectors:
            manager = managers[name]
            costs = {**prices, **manager.sale_values(prices)}
            projection = attach_meta(projections[name].copy(), data, form, costs,
                                     list(manager.squad), t)
            record = play_deadline(manager, projection, t, prices, data, config,
                                   first_deadline=index == 0)
            logs[name].append(record)

        LOGGER.info("seed %-3d GW%-2d  %s   (%.0fs)", config.random_state, t, "  ".join(
            f"{name} {logs[name][-1]['points']:>3.0f} [{managers[name].total_points:>4.0f}]"
            for name in projectors), time.perf_counter() - started)
    return logs


def run_simulation(config: SimConfig) -> Dict[str, Any]:
    """
    Replay the season once per seed.

    A single replay is noisy: a small change in projections sends the squad
    down a different path and the differences compound over 38 weeks. Varying
    only the model seed measures that noise, so a claim about the model is a
    claim about the spread, not one lucky path. The baselines are
    deterministic and run once.
    """
    features = build_sim_features(config)
    data = SeasonData(features, config.season, config.team_rating_window)

    logs: Dict[str, List[Dict[str, Any]]] = {}
    seed_runs: Dict[int, pd.DataFrame] = {}
    seeds = config.seeds if "model" in config.projectors else config.seeds[:1]
    for index, seed in enumerate(seeds):
        projectors = list(config.projectors) if index == 0 else ["model"]
        result = simulate(replace(config, random_state=seed), features, data, projectors)
        if index == 0:
            logs = {name: rows for name, rows in result.items() if name != "model"}
        if "model" in result:
            seed_runs[seed] = pd.DataFrame(result["model"])
    return write_outputs(logs, seed_runs, config)


def load_benchmark(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    frame = pd.read_csv(path)
    if not {"gameweek", "average_points"}.issubset(frame.columns):
        raise RuntimeError(f"{path} needs columns gameweek,average_points.")
    return frame[["gameweek", "average_points"]]


def _summarise(name: str, frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "projector": name,
        "gameweeks": len(frame),
        "total_points": frame["points"].sum(),
        "mean_points": frame["points"].mean(),
        "hits": int(frame["hits"].sum()),
        "transfers": int(frame["transfers"].sum()),
        "auto_subs": int(frame["auto_subs"].sum()),
        "captain_points": frame["captain_points"].sum(),
        "final_squad_value": frame["squad_value"].iloc[-1] + frame["bank"].iloc[-1],
    }


def write_outputs(logs: Dict[str, List[Dict[str, Any]]], seed_runs: Dict[int, pd.DataFrame],
                  config: SimConfig) -> Dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = {name: pd.DataFrame(rows) for name, rows in logs.items()}
    for name, frame in frames.items():
        frame.to_csv(out / f"sim_{name}_gameweeks.csv", index=False)
    for seed, frame in seed_runs.items():
        frame.to_csv(out / f"sim_model_seed{seed}_gameweeks.csv", index=False)

    rows = [_summarise(name, frame) for name, frame in frames.items()]
    seeds = pd.DataFrame([{"seed": seed, **_summarise("model", frame)}
                          for seed, frame in seed_runs.items()])
    if not seeds.empty:
        seeds = seeds.drop(columns="projector")
        seeds.to_csv(out / "sim_seeds.csv", index=False)
        rows.insert(0, {"projector": "model", **seeds.drop(columns="seed").mean().to_dict(),
                        "seeds": len(seeds),
                        "total_std": seeds["total_points"].std(ddof=1) if len(seeds) > 1
                        else 0.0,
                        "total_min": seeds["total_points"].min(),
                        "total_max": seeds["total_points"].max()})
    summary = pd.DataFrame(rows)

    benchmark = load_benchmark(config.benchmark_csv)
    if benchmark is not None:
        reference = next(iter(seed_runs.values()), None)
        played = (reference if reference is not None else next(iter(frames.values())))["gameweek"]
        average = benchmark[benchmark["gameweek"].isin(played)]["average_points"].sum()
        summary = pd.concat([summary, pd.DataFrame([{
            "projector": "average_manager", "gameweeks": len(played),
            "total_points": average, "mean_points": average / max(len(played), 1)}])],
            ignore_index=True)

    summary.to_csv(out / "sim_summary.csv", index=False)
    settings = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(config).items() if k not in ("metadata", "random_state")}
    (out / "sim_config.json").write_text(json.dumps(settings, indent=2, default=list),
                                         encoding="utf-8")
    plot_cumulative(frames, seed_runs, benchmark, out / "sim_cumulative_points.png", config)
    LOGGER.info("Wrote results to %s/", out)
    return {"summary": summary, "gameweeks": frames, "seeds": seeds}


def plot_cumulative(frames: Dict[str, pd.DataFrame], seed_runs: Dict[int, pd.DataFrame],
                    benchmark: Optional[pd.DataFrame], path: Path, config: SimConfig) -> None:
    """
    Cumulative points, model against the baseline (and the average manager when
    supplied). With several seeds the model is drawn as their mean with the
    min-max band, so the chart shows the noise rather than one path. The
    hindsight ceiling is named in the subtitle rather than drawn: at more than
    twice the scale it would flatten the comparison that matters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    def end_label(x, y, text):
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=muted)

    if "roll3" in frames:
        frame = frames["roll3"]
        total = frame["points"].sum()
        ax.plot(frame["gameweek"], frame["points"].cumsum(), lw=2, color=orange,
                label=f"3-GW rolling average  {total:,.0f}")
        end_label(frame["gameweek"].iloc[-1], total, f"3-GW rolling average  {total:,.0f}")

    reference = None
    if seed_runs:
        paths = pd.DataFrame({seed: frame.set_index("gameweek")["points"].cumsum()
                              for seed, frame in seed_runs.items()})
        reference = paths.index
        mean = paths.mean(axis=1)
        if paths.shape[1] > 1:
            ax.fill_between(paths.index, paths.min(axis=1), paths.max(axis=1), color=blue,
                            alpha=0.16, lw=0, label=f"Model, range over {paths.shape[1]} seeds")
            label = f"Two-stage model, mean of {paths.shape[1]} seeds  {mean.iloc[-1]:,.0f}"
        else:
            label = f"Two-stage model  {mean.iloc[-1]:,.0f}"
        ax.plot(mean.index, mean, lw=2, color=blue, label=label)
        end_label(mean.index[-1], mean.iloc[-1], f"Two-stage model  {mean.iloc[-1]:,.0f}")

    if benchmark is not None:
        gws = reference if reference is not None else next(iter(frames.values()))["gameweek"]
        bench = benchmark[benchmark["gameweek"].isin(gws)].sort_values("gameweek")
        total = bench["average_points"].sum()
        ax.plot(bench["gameweek"], bench["average_points"].cumsum(), color=muted, lw=1.5,
                ls=(0, (4, 3)), label=f"Average manager  {total:,.0f}")

    subtitle = f"Same solver and FPL rules for both, H={config.horizon}, no chips"
    if "oracle" in frames:
        subtitle += f"; hindsight ceiling {frames['oracle']['points'].sum():,.0f}"
    fig.text(0.012, 0.965, f"{config.season} replayed deadline by deadline",
             fontsize=13, color=ink, weight="bold", va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=9.5, color=muted, va="top")

    ax.set_xlabel("Gameweek", color=muted)
    ax.set_ylabel("Cumulative points, after hits", color=muted)
    ax.tick_params(colors=muted, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    # Drawn baseline-first so the model sits on top; listed model-first.
    rank = {"Two-stage": 0, "Model, range": 1}
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: next((v for k, v in rank.items() if labels[i].startswith(k)), 2))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="upper left", labelcolor=ink, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 0.89))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def print_summary(summary: pd.DataFrame, seeds: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("SEASON SIMULATION")
    print("=" * 78)
    columns = ["projector", "total_points", "mean_points", "transfers", "hits",
               "auto_subs", "final_squad_value"]
    print(summary[[c for c in columns if c in summary.columns]]
          .round(1).to_string(index=False))
    if len(seeds) > 1:
        print(f"\nModel over {len(seeds)} seeds: "
              + ", ".join(f"{s}: {t:,.0f}" for s, t in zip(seeds["seed"], seeds["total_points"]))
              + f"  (std {seeds['total_points'].std(ddof=1):.0f})")
    if not seeds.empty and "roll3" in set(summary["projector"]):
        roll3 = summary.set_index("projector").at["roll3", "total_points"]
        wins = int((seeds["total_points"] > roll3).sum())
        print(f"Model vs 3-GW rolling average: {seeds['total_points'].mean() - roll3:+.0f} "
              f"points on average; ahead in {wins} of {len(seeds)} seed(s)")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an FPL season deadline by deadline.")
    parser.add_argument("-i", "--history", default="fpl_combined_history.csv")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--train-seasons", nargs="+", default=["2023-24", "2024-25"])
    parser.add_argument("--start-gw", type=int, default=1)
    parser.add_argument("--end-gw", type=int, default=None)
    parser.add_argument("--projectors", nargs="+", default=list(PROJECTORS),
                        choices=list(PROJECTORS))
    parser.add_argument("--retrain-every", type=int, default=0,
                        help="Retrain walk-forward every N gameweeks (0 = train once)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--bench-weight", type=float, default=0.10)
    parser.add_argument("--candidate-pool", type=int, default=25)
    parser.add_argument("--max-transfers", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--no-odds", action="store_true",
                        help="Leave the bookmaker odds features out (ablation)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Model random seeds, one replay each; several measure the noise")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Seeds averaged inside each model stage")
    parser.add_argument("--benchmark-csv", default=None,
                        help="CSV with gameweek,average_points to plot alongside")
    parser.add_argument("-o", "--output-dir", default="sim_results")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Progress at the requested level; the pipeline modules it drives log a
    # line per solve and per model, so they only speak up for warnings.
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    LOGGER.setLevel(getattr(logging, args.log_level))
    config = SimConfig(
        history_path=Path(args.history), season=args.season,
        train_seasons=tuple(args.train_seasons), start_gw=args.start_gw, end_gw=args.end_gw,
        projectors=tuple(args.projectors), retrain_every=args.retrain_every,
        horizon=args.horizon, decay=args.decay, bench_weight=args.bench_weight,
        candidate_pool=args.candidate_pool, max_transfers_per_week=args.max_transfers,
        time_limit=args.time_limit, odds=not args.no_odds, seeds=tuple(args.seeds),
        ensemble_size=args.ensemble,
        benchmark_csv=Path(args.benchmark_csv) if args.benchmark_csv else None,
        output_dir=Path(args.output_dir),
    )
    try:
        outcome = run_simulation(config)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    print_summary(outcome["summary"], outcome["seeds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
