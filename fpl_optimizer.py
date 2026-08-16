"""
FPL ML Pipeline — Phase 4: Squad Optimization
=============================================

Scores every player with the Phase 3 position models, then solves a mixed-integer
linear program (PuLP/CBC) for the highest-expected-points legal FPL squad.

Decision variables, per player i
    x_i in {0,1}   in the 15-man squad
    s_i in {0,1}   in the starting XI
    c_i in {0,1}   captain (points doubled)

Objective
    maximise  sum_i (s_i + c_i) * predicted_points_i
    (the +c_i term is what doubles the captain: he scores once as a starter and
     once again as captain)

Constraints
    sum x_i = 15                       squad size
    sum x_i by position = 2/5/5/3      GKP/DEF/MID/FWD
    sum x_i * cost_i <= 100.0          budget
    sum x_i by team <= 3               club quota
    sum s_i = 11                       starting XI
    s_i <= x_i                         starters come from the squad
    GKP 1, DEF 3-5, MID 2-5, FWD 1-3   legal formation
    sum c_i = 1, c_i <= s_i            exactly one captain, and he starts

Usage
-----
    python fpl_optimizer.py
    python fpl_optimizer.py --gameweek 38 --budget 100.0
    python fpl_optimizer.py --bench-weight 0.15      # value the bench a little
    python fpl_optimizer.py --exclude "Player 001,Player 042"


Three things to know before you trust the output
------------------------------------------------
1. AVAILABILITY IS NOT MODELLED. The Phase 2 file carries no `status` or
   `chance_of_playing_next_round`, so a player with six months on the treatment
   table but great historic form is a perfectly legal pick here. Before acting on
   a squad, re-pull `elements` from bootstrap-static and drop anyone whose
   `status` is not 'a'. `--exclude` is the manual stopgap.

2. THIS OPTIMISES THE GAMEWEEK YOU SCORE, WHICH HAS ALREADY BEEN PLAYED. Phase 2
   builds row N's features from N-1 and older, so scoring the most recent
   available row predicts *that* gameweek. Forecasting the next one needs a
   feature row for it, which needs the upcoming fixture list (`was_home`,
   opponent, price). That is a Phase 5 job.

3. WITH `--bench-weight 0` THE BENCH IS DELIBERATELY WORTHLESS. A single-gameweek
   objective spends every last 0.1m on the XI and fills the bench with minimum
   price bodies, which is the true optimum for one week and useless for the next
   five. A small positive bench weight buys a usable bench.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("joblib is required: pip install joblib") from exc

try:
    import pulp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pulp is required: pip install pulp") from exc


LOGGER = logging.getLogger("fpl.optimizer")

POSITIONS: List[str] = ["GKP", "DEF", "MID", "FWD"]

# Squad composition mandated by FPL.
SQUAD_QUOTA: Dict[str, int] = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE: int = sum(SQUAD_QUOTA.values())          # 15

# Legal starting XI bounds (min, max) per position.
FORMATION_BOUNDS: Dict[str, Tuple[int, int]] = {
    "GKP": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}
STARTING_XI: int = 11
MAX_PER_CLUB: int = 3


@dataclass
class OptimizerConfig:
    features_path: Path = Path("fpl_ml_features.csv")
    model_dir: Path = Path("models")
    budget: float = 100.0
    season: Optional[str] = None       # default: latest in the file
    gameweek: Optional[int] = None     # default: latest in that season
    bench_weight: float = 0.0          # 0 = pure single-gameweek objective
    exclude: Sequence[str] = ()
    min_minutes_roll_3: float = 0.0    # crude availability filter
    solver_msg: bool = False
    time_limit: Optional[int] = None
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Inference setup
# --------------------------------------------------------------------------- #

def load_features(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fpl_features.py (Phase 2) first.")
    frame = pd.read_csv(path, low_memory=False)
    required = {"player_id", "player_name", "position", "team", "season", "gameweek"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
    LOGGER.info("Loaded %s: %d rows x %d columns", path, len(frame), frame.shape[1])
    return frame


def select_gameweek(
    frame: pd.DataFrame, season: Optional[str], gameweek: Optional[int]
) -> Tuple[pd.DataFrame, str, int]:
    """Slice to one season/gameweek, defaulting to the most recent available."""
    season = season or sorted(frame["season"].dropna().unique())[-1]
    in_season = frame[frame["season"] == season]
    if in_season.empty:
        raise RuntimeError(f"Season '{season}' not found. "
                           f"Available: {sorted(frame['season'].dropna().unique())}")

    gameweeks = pd.to_numeric(in_season["gameweek"], errors="coerce")
    gameweek = int(gameweek if gameweek is not None else gameweeks.max())
    slice_ = in_season[gameweeks == gameweek].copy()
    if slice_.empty:
        raise RuntimeError(
            f"Gameweek {gameweek} not found in {season}. "
            f"Available: {int(gameweeks.min())}-{int(gameweeks.max())}"
        )

    LOGGER.info("Optimising for %s GW%d — %d candidate rows.", season, gameweek, len(slice_))
    return slice_, season, gameweek


def aggregate_double_gameweeks(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per player.

    In a double gameweek a player has two fixture rows sharing one `gameweek`.
    Left as-is, each would become its own decision variable and the solver could
    happily "select" the same player twice. Predicted points are summed across
    his fixtures (he really does score in both); cost and metadata are taken from
    the first row.
    """
    if "predicted_points" not in frame.columns:
        raise RuntimeError("aggregate_double_gameweeks() must run after prediction.")

    key = ["player_id", "player_name"]
    duplicated = frame.duplicated(subset=key, keep=False)
    if not duplicated.any():
        return frame.reset_index(drop=True)

    n_double = frame.loc[duplicated, "player_id"].nunique()
    LOGGER.info("Double gameweek: %d player(s) with multiple fixtures — summing their points.",
                n_double)

    aggregations = {
        "predicted_points": "sum",
        "position": "first",
        "team": "first",
        "cost": "first",
        "n_fixtures": "sum",
    }
    frame = frame.assign(n_fixtures=1)
    keep = [c for c in aggregations if c in frame.columns]
    collapsed = frame.groupby(key, as_index=False).agg({c: aggregations[c] for c in keep})
    return collapsed.reset_index(drop=True)


def resolve_costs(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build the `cost` column the budget constraint uses.

    Prefers `now_cost` as specified. But Phase 1 leaves `now_cost` NaN on archive
    rows by design (today's price on a 2023 row would be lookahead), so if the
    gameweek being optimised came from the archive, every price is NaN and the
    budget constraint would be meaningless. Falls back to `gw_cost` — the price
    at that gameweek — and says so loudly.
    """
    frame = frame.copy()

    now_cost = pd.to_numeric(frame.get("now_cost"), errors="coerce") \
        if "now_cost" in frame.columns else pd.Series(np.nan, index=frame.index)
    gw_cost = pd.to_numeric(frame.get("gw_cost"), errors="coerce") \
        if "gw_cost" in frame.columns else pd.Series(np.nan, index=frame.index)

    frame["cost"] = now_cost
    n_missing = int(frame["cost"].isna().sum())

    if n_missing:
        filled = frame["cost"].isna() & gw_cost.notna()
        frame.loc[filled, "cost"] = gw_cost[filled]
        LOGGER.warning(
            "`now_cost` was NaN for %d/%d players — fell back to `gw_cost` for %d of them. "
            "(Phase 1 leaves now_cost empty on archive rows.)",
            n_missing, len(frame), int(filled.sum()),
        )

    still_missing = frame["cost"].isna()
    if still_missing.any():
        LOGGER.warning("Dropping %d player(s) with no usable price.", int(still_missing.sum()))
        frame = frame[~still_missing]

    if frame.empty:
        raise RuntimeError("No players have a usable price — cannot enforce the budget.")

    LOGGER.info("Prices resolved: %.1f - %.1f (mean %.2f)",
                frame["cost"].min(), frame["cost"].max(), frame["cost"].mean())
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model loading + prediction
# --------------------------------------------------------------------------- #

def load_models(model_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load the four position bundles and the shared feature metadata."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found. Run fpl_train.py (Phase 3) first.")

    bundles: Dict[str, Dict[str, Any]] = {}
    for position in POSITIONS:
        path = model_dir / f"fpl_model_{position}.joblib"
        if not path.exists():
            LOGGER.warning("No model for %s at %s — those players will be skipped.",
                           position, path)
            continue
        try:
            bundles[position] = joblib.load(path)
        except Exception as exc:
            LOGGER.warning("Could not load %s: %s", path, exc)

    if not bundles:
        raise RuntimeError(f"No position models could be loaded from {model_dir}/.")

    metadata_path = model_dir / "feature_metadata.joblib"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = joblib.load(metadata_path)
        except Exception as exc:
            LOGGER.warning("Could not load feature metadata: %s", exc)

    LOGGER.info("Loaded %d position model(s): %s", len(bundles), ", ".join(sorted(bundles)))
    return bundles, metadata


def add_position_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the one-hots Phase 3 trained on (harmless if a model ignores them)."""
    frame = frame.copy()
    for position in POSITIONS:
        frame[f"pos_{position}"] = (frame["position"] == position).astype("int8")
    return frame


def prepare_matrix(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Build the model's exact feature matrix, creating any absent column as NaN."""
    matrix = pd.DataFrame(index=frame.index)
    missing: List[str] = []
    for column in features:
        if column in frame.columns:
            series = frame[column]
            matrix[column] = series.astype("int8") if series.dtype == bool \
                else pd.to_numeric(series, errors="coerce")
        else:
            matrix[column] = np.nan
            missing.append(column)
    if missing:
        LOGGER.warning("Feature(s) absent at inference, passed as NaN: %s", ", ".join(missing))
    return matrix


def predict_points(frame: pd.DataFrame, bundles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Score each position with its own model. Players without a model are dropped."""
    frame = add_position_indicators(frame)
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
        # Never propose a negative expectation to the solver.
        frame.loc[mask, "predicted_points"] = np.clip(predictions, 0.0, None)
        LOGGER.info("  %s: scored %d players (mean %.2f, max %.2f)",
                    position, int(mask.sum()),
                    float(np.mean(predictions)), float(np.max(predictions)))

    unscored = frame["predicted_points"].isna()
    if unscored.any():
        LOGGER.warning("Dropping %d player(s) with no model for their position.",
                       int(unscored.sum()))
        frame = frame[~unscored]

    return frame.reset_index(drop=True)


def apply_filters(frame: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    """Manual availability stopgap — see caveat 1 in the module docstring."""
    if config.exclude:
        excluded = {name.strip().lower() for name in config.exclude if name.strip()}
        mask = frame["player_name"].str.lower().isin(excluded)
        if mask.any():
            LOGGER.info("Excluding %d player(s) by name.", int(mask.sum()))
            frame = frame[~mask]
        unmatched = excluded - set(frame["player_name"].str.lower()) - {
            n for n in excluded if mask.any()}
        for name in sorted(unmatched):
            LOGGER.warning("  --exclude '%s' matched nobody.", name)

    if config.min_minutes_roll_3 > 0 and "minutes_roll_3" in frame.columns:
        minutes = pd.to_numeric(frame["minutes_roll_3"], errors="coerce").fillna(0)
        mask = minutes >= config.min_minutes_roll_3
        LOGGER.info("Minutes filter (minutes_roll_3 >= %.0f): %d of %d players kept.",
                    config.min_minutes_roll_3, int(mask.sum()), len(frame))
        frame = frame[mask]

    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2-3. MILP formulation
# --------------------------------------------------------------------------- #

def check_feasibility(frame: pd.DataFrame, budget: float) -> None:
    """Catch impossible inputs here, where the message can be useful."""
    problems: List[str] = []

    for position, quota in SQUAD_QUOTA.items():
        available = int((frame["position"] == position).sum())
        if available < quota:
            problems.append(f"only {available} {position} available, need {quota}")

    # Cheapest legal squad: the n cheapest players in each position.
    floor = 0.0
    for position, quota in SQUAD_QUOTA.items():
        cheapest = frame.loc[frame["position"] == position, "cost"].nsmallest(quota)
        floor += float(cheapest.sum())
    if floor > budget:
        problems.append(f"cheapest legal squad costs {floor:.1f}, budget is {budget:.1f}")

    clubs = frame["team"].nunique()
    if clubs * MAX_PER_CLUB < SQUAD_SIZE:
        problems.append(f"only {clubs} clubs x max {MAX_PER_CLUB} < {SQUAD_SIZE} players")

    if problems:
        raise RuntimeError("Problem is infeasible before solving: " + "; ".join(problems))

    LOGGER.info("Feasibility pre-check passed (cheapest legal squad: %.1fm).", floor)


class SquadOptimizer:
    """MILP for the best legal 15 + XI + captain under an FPL budget."""

    def __init__(self, players: pd.DataFrame, config: OptimizerConfig) -> None:
        self.players = players.reset_index(drop=True)
        self.config = config
        self.problem: Optional[pulp.LpProblem] = None
        self.x: Dict[int, pulp.LpVariable] = {}
        self.s: Dict[int, pulp.LpVariable] = {}
        self.c: Dict[int, pulp.LpVariable] = {}

    # -- build ---------------------------------------------------------------
    def build(self) -> pulp.LpProblem:
        players = self.players
        index = list(players.index)
        points = players["predicted_points"].to_dict()
        cost = players["cost"].to_dict()

        problem = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)

        self.x = pulp.LpVariable.dicts("squad", index, cat=pulp.LpBinary)
        self.s = pulp.LpVariable.dicts("start", index, cat=pulp.LpBinary)
        self.c = pulp.LpVariable.dicts("captain", index, cat=pulp.LpBinary)

        # ---- objective ----
        # A starter contributes p_i; the captain contributes p_i a second time,
        # which is exactly the doubling rule. Bench players contribute
        # bench_weight * p_i (0 by default — see caveat 3).
        bench_weight = self.config.bench_weight
        problem += (
            pulp.lpSum(points[i] * self.s[i] for i in index)
            + pulp.lpSum(points[i] * self.c[i] for i in index)
            + pulp.lpSum(bench_weight * points[i] * (self.x[i] - self.s[i]) for i in index)
        ), "expected_points"

        # ---- squad ----
        problem += pulp.lpSum(self.x[i] for i in index) == SQUAD_SIZE, "squad_size"

        for position, quota in SQUAD_QUOTA.items():
            members = [i for i in index if players.at[i, "position"] == position]
            problem += pulp.lpSum(self.x[i] for i in members) == quota, f"squad_{position}"

        problem += (
            pulp.lpSum(cost[i] * self.x[i] for i in index) <= self.config.budget
        ), "budget"

        for club in players["team"].dropna().unique():
            members = [i for i in index if players.at[i, "team"] == club]
            problem += (
                pulp.lpSum(self.x[i] for i in members) <= MAX_PER_CLUB
            ), f"club_quota_{_safe_name(club)}"

        # ---- starting XI ----
        problem += pulp.lpSum(self.s[i] for i in index) == STARTING_XI, "starting_xi"

        for i in index:                                    # s_i <= x_i
            problem += self.s[i] <= self.x[i], f"start_in_squad_{i}"

        for position, (low, high) in FORMATION_BOUNDS.items():
            members = [i for i in index if players.at[i, "position"] == position]
            problem += pulp.lpSum(self.s[i] for i in members) >= low, f"formation_min_{position}"
            problem += pulp.lpSum(self.s[i] for i in members) <= high, f"formation_max_{position}"

        # ---- captain ----
        problem += pulp.lpSum(self.c[i] for i in index) == 1, "one_captain"
        for i in index:                                    # c_i <= s_i
            problem += self.c[i] <= self.s[i], f"captain_starts_{i}"

        self.problem = problem
        LOGGER.info("Built MILP: %d variables, %d constraints.",
                    len(problem.variables()), len(problem.constraints))
        return problem

    # -- solve ---------------------------------------------------------------
    def solve(self) -> str:
        if self.problem is None:
            self.build()

        solver = pulp.PULP_CBC_CMD(msg=self.config.solver_msg,
                                   timeLimit=self.config.time_limit)
        self.problem.solve(solver)
        status = pulp.LpStatus[self.problem.status]

        if status != "Optimal":
            raise RuntimeError(
                f"Solver finished with status '{status}'. The constraints may be "
                f"unsatisfiable — try raising --budget or relaxing --min-minutes-roll-3."
            )

        LOGGER.info("Solved: %s | objective = %.2f expected points",
                    status, pulp.value(self.problem.objective))
        return status

    # -- extract -------------------------------------------------------------
    def extract(self) -> pd.DataFrame:
        """Selected squad with role flags, validated against the FPL rules."""
        chosen = [i for i in self.players.index if self.x[i].value() > 0.5]
        squad = self.players.loc[chosen].copy()
        squad["is_starter"] = [self.s[i].value() > 0.5 for i in chosen]
        squad["is_captain"] = [self.c[i].value() > 0.5 for i in chosen]

        # Vice-captain is not a decision variable — FPL only uses him if the
        # captain blanks, so he is simply the best remaining starter.
        squad["is_vice"] = False
        starters = squad[squad["is_starter"] & ~squad["is_captain"]]
        if not starters.empty:
            squad.loc[starters["predicted_points"].idxmax(), "is_vice"] = True

        squad["effective_points"] = squad["predicted_points"] * (
            squad["is_starter"].astype(int) + squad["is_captain"].astype(int)
        )

        validate_squad(squad, self.config.budget)
        return squad.sort_values(
            ["is_starter", "position", "predicted_points"],
            ascending=[False, True, False],
            key=lambda col: col.map(
                {p: n for n, p in enumerate(POSITIONS)}) if col.name == "position" else col,
        ).reset_index(drop=True)


def _safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def validate_squad(squad: pd.DataFrame, budget: float) -> None:
    """Re-check the solution against the rules. Cheap insurance against a bad model."""
    assert len(squad) == SQUAD_SIZE, f"squad has {len(squad)} players, expected {SQUAD_SIZE}"

    counts = squad["position"].value_counts().to_dict()
    for position, quota in SQUAD_QUOTA.items():
        assert counts.get(position, 0) == quota, \
            f"squad has {counts.get(position, 0)} {position}, expected {quota}"

    total_cost = squad["cost"].sum()
    assert total_cost <= budget + 1e-6, f"squad costs {total_cost:.1f}, over budget {budget:.1f}"

    starters = squad[squad["is_starter"]]
    assert len(starters) == STARTING_XI, f"{len(starters)} starters, expected {STARTING_XI}"

    start_counts = starters["position"].value_counts().to_dict()
    for position, (low, high) in FORMATION_BOUNDS.items():
        n = start_counts.get(position, 0)
        assert low <= n <= high, f"illegal formation: {n} {position} (allowed {low}-{high})"

    club_counts = squad["team"].value_counts()
    assert club_counts.max() <= MAX_PER_CLUB, \
        f"{club_counts.idxmax()} has {club_counts.max()} players, max {MAX_PER_CLUB}"

    captains = squad[squad["is_captain"]]
    assert len(captains) == 1, f"{len(captains)} captains, expected 1"
    assert bool(captains.iloc[0]["is_starter"]), "captain is not in the starting XI"

    LOGGER.info("Solution validated against all FPL rules.")


# --------------------------------------------------------------------------- #
# 4. Report
# --------------------------------------------------------------------------- #

def formation_string(starters: pd.DataFrame) -> str:
    counts = starters["position"].value_counts().to_dict()
    return f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"


def _table(rows: pd.DataFrame, width: int = 78) -> str:
    lines = []
    for _, row in rows.iterrows():
        marker = "(C)" if row["is_captain"] else ("(V)" if row.get("is_vice") else "")
        name = f"{row['player_name']} {marker}".strip()
        lines.append(
            f"  {row['position']:<4} {name:<30.30} {str(row['team']):<14.14} "
            f"{row['cost']:>6.1f}  {row['predicted_points']:>7.2f}"
        )
    return "\n".join(lines)


def print_report(squad: pd.DataFrame, season: str, gameweek: int, config: OptimizerConfig) -> None:
    starters = squad[squad["is_starter"]]
    bench = squad[~squad["is_starter"]]

    captain = squad[squad["is_captain"]].iloc[0]
    xi_points = float(starters["predicted_points"].sum())
    captain_bonus = float(captain["predicted_points"])
    total_points = xi_points + captain_bonus

    header = f"{'POS':<4} {'PLAYER':<30} {'TEAM':<14} {'COST':>6}  {'xPTS':>7}"

    print("\n" + "=" * 78)
    print(f"OPTIMAL FPL SQUAD — {season} GW{gameweek}".center(78))
    print("=" * 78)

    print(f"\nSTARTING XI  ({formation_string(starters)})")
    print("  " + header)
    print("  " + "-" * 74)
    print(_table(starters))
    print("  " + "-" * 74)
    print(f"  {'':<4} {'XI subtotal':<30} {'':<14} "
          f"{starters['cost'].sum():>6.1f}  {xi_points:>7.2f}")

    print("\nBENCH")
    print("  " + header)
    print("  " + "-" * 74)
    print(_table(bench))
    print("  " + "-" * 74)
    print(f"  {'':<4} {'Bench subtotal':<30} {'':<14} "
          f"{bench['cost'].sum():>6.1f}  {bench['predicted_points'].sum():>7.2f}")

    print("\n" + "-" * 78)
    print(f"  Captain            {captain['player_name']} "
          f"(+{captain_bonus:.2f} doubled)")
    vice = squad[squad["is_vice"]]
    if not vice.empty:
        print(f"  Vice-captain       {vice.iloc[0]['player_name']}")
    print(f"  Squad cost         {squad['cost'].sum():.1f} / {config.budget:.1f} "
          f"  (remaining {config.budget - squad['cost'].sum():.1f})")
    print(f"  Clubs used         {squad['team'].nunique()} "
          f"(max {squad['team'].value_counts().max()} from one club)")
    print(f"  PROJECTED POINTS   {total_points:.2f}   "
          f"(XI {xi_points:.2f} + captain {captain_bonus:.2f})")
    if config.bench_weight == 0:
        print("\n  Note: --bench-weight is 0, so the bench is optimised to be as cheap as")
        print("        possible. Set it to ~0.1-0.2 for a squad that survives rotation.")
    print("-" * 78)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_optimization(config: OptimizerConfig) -> Dict[str, Any]:
    frame = load_features(config.features_path)
    slice_, season, gameweek = select_gameweek(frame, config.season, config.gameweek)

    bundles, metadata = load_models(config.model_dir)
    if metadata.get("target"):
        LOGGER.info("Models predict `%s`, trained %s.",
                    metadata["target"], metadata.get("trained_at", "at an unknown time"))

    scored = predict_points(slice_, bundles)
    scored = resolve_costs(scored)
    scored = aggregate_double_gameweeks(scored)
    scored = apply_filters(scored, config)

    LOGGER.info("Candidate pool: %d players across %d clubs.",
                len(scored), scored["team"].nunique())
    check_feasibility(scored, config.budget)

    optimizer = SquadOptimizer(scored, config)
    optimizer.build()
    optimizer.solve()
    squad = optimizer.extract()

    if config.output_path:
        path = Path(config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        squad.to_csv(path, index=False)
        LOGGER.info("Wrote squad to %s", path)

    return {"squad": squad, "season": season, "gameweek": gameweek,
            "candidates": scored, "problem": optimizer.problem}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Phase 4 squad optimization (MILP).")
    parser.add_argument("-i", "--input", default="fpl_ml_features.csv")
    parser.add_argument("-d", "--model-dir", default="models")
    parser.add_argument("-b", "--budget", type=float, default=100.0)
    parser.add_argument("--season", default=None, help="Default: latest in the file")
    parser.add_argument("--gameweek", type=int, default=None, help="Default: latest in season")
    parser.add_argument("--bench-weight", type=float, default=0.0,
                        help="Weight on bench players' points (0 = pure single-GW objective)")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated player names to exclude (injuries etc.)")
    parser.add_argument("--min-minutes-roll-3", type=float, default=0.0,
                        help="Drop players whose 3-game minutes average is below this")
    parser.add_argument("-o", "--output", default=None, help="Optional CSV path for the squad")
    parser.add_argument("--time-limit", type=int, default=None, help="Solver time limit (seconds)")
    parser.add_argument("--solver-msg", action="store_true", help="Show CBC solver output")
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

    config = OptimizerConfig(
        features_path=Path(args.input),
        model_dir=Path(args.model_dir),
        budget=args.budget,
        season=args.season,
        gameweek=args.gameweek,
        bench_weight=args.bench_weight,
        exclude=[n for n in args.exclude.split(",") if n.strip()],
        min_minutes_roll_3=args.min_minutes_roll_3,
        solver_msg=args.solver_msg,
        time_limit=args.time_limit,
        output_path=Path(args.output) if args.output else None,
    )

    try:
        outcome = run_optimization(config)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except AssertionError as exc:
        LOGGER.error("Solution violated an FPL rule: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130

    print_report(outcome["squad"], outcome["season"], outcome["gameweek"], config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
