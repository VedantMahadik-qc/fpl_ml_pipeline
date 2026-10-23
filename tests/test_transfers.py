import itertools

import numpy as np
import pandas as pd
import pytest

pulp = pytest.importorskip("pulp")

from fpl_optimizer import SQUAD_QUOTA
from fpl_transfers import TransferConfig, plan_transfers


def synthetic_projections(gameweeks=(1, 2), seed=0):
    """Ten clubs x a full position spread, random but reproducible points."""
    rng = np.random.default_rng(seed)
    rows, pid = [], itertools.count(1)
    for club in range(10):
        for position, count in (("GKP", 2), ("DEF", 4), ("MID", 4), ("FWD", 2)):
            for _ in range(count):
                player = next(pid)
                cost = round(float(rng.uniform(4.0, 9.0)), 1)
                for gw in gameweeks:
                    rows.append({
                        "player_id": player, "gameweek": gw, "position": position,
                        "team": f"Club{club}", "player_name": f"P{player}", "cost": cost,
                        "predicted_points": float(rng.gamma(2.0, 1.5)) * cost / 6,
                    })
    return pd.DataFrame(rows)


def config(**overrides):
    base = dict(horizon=2, candidate_pool=15, max_transfers_per_week=2, time_limit=60)
    base.update(overrides)
    return TransferConfig(**base)


def test_fresh_build_is_legal_and_within_budget():
    projections = synthetic_projections()
    result = plan_transfers(projections, [], 100.0, 5, config(wildcard=True))
    week = result["plan"][0]
    squad = week["squad"]

    assert len(squad) == 15
    assert squad["position"].value_counts().to_dict() == SQUAD_QUOTA
    assert squad["cost"].sum() <= 100.0 + 1e-9
    assert squad["team"].value_counts().max() <= 3
    assert week["hits"] == 0


def test_one_free_transfer_week_never_overspends():
    projections = synthetic_projections()
    first = plan_transfers(projections, [], 100.0, 5, config(wildcard=True))["plan"][0]
    squad = first["squad"]["player_id"].tolist()
    bank = 100.0 - first["squad"]["cost"].sum()

    # Next week one owned player collapses; the plan should replace him.
    week2 = synthetic_projections(gameweeks=(2, 3), seed=1)
    week2.loc[week2["player_id"] == squad[0], "predicted_points"] = 0.0
    result = plan_transfers(week2, squad, bank, 1, config())
    week = result["plan"][0]

    assert squad[0] not in set(week["squad"]["player_id"])
    assert week["bank"] >= -1e-9
    assert week["hits"] == max(0, week["n_transfers"] - 1)
