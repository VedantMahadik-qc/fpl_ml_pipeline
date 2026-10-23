import pytest

pytest.importorskip("pulp")

from fpl_transfers import TransferConfig, plan_transfers
from tests.test_transfers import synthetic_projections


def config(windows, reserve=None, **overrides):
    base = dict(horizon=3, candidate_pool=12, max_transfers_per_week=2, time_limit=60,
                chip_windows=windows, chip_reserve=reserve or {})
    base.update(overrides)
    return TransferConfig(**base)


def owned_squad(gameweeks=(1, 2, 3)):
    """A legal squad from a plain fresh build, to start chip tests from."""
    projections = synthetic_projections(gameweeks=gameweeks)
    week = plan_transfers(projections, [], 100.0, 5,
                          config((), wildcard=True))["plan"][0]
    return projections, week["squad"]["player_id"].tolist(), 100.0 - week["squad"]["cost"].sum()


def chips_played(result):
    return {week["gameweek"]: week["chip"] for week in result["plan"] if week["chip"]}


@pytest.mark.parametrize("chip", ["bboost", "3xc"])
def test_points_chips_go_to_the_big_week(chip):
    projections, squad, bank = owned_squad()
    projections.loc[projections["gameweek"] == 3, "predicted_points"] *= 3
    result = plan_transfers(projections, squad, bank, 1,
                            config([(chip, 1, 38)], {chip: 0.0}))
    assert chips_played(result) == {3: chip}

    week = result["plan"][2]
    if chip == "bboost":
        assert week["bench_boost_points"] > 0
    else:
        captain = week["squad"][week["squad"]["is_captain"]]["predicted_points"].iloc[0]
        assert week["captain_bonus"] == pytest.approx(2 * captain)


def test_a_chip_is_held_unless_it_beats_its_reserve_or_expires():
    projections, squad, bank = owned_squad()
    held = plan_transfers(projections, squad, bank, 1,
                          config([("bboost", 1, 38)], {"bboost": 1000.0}))
    assert chips_played(held) == {}

    # Same reserve, but this copy is gone after GW3: use it or lose it.
    expiring = plan_transfers(projections, squad, bank, 1,
                              config([("bboost", 1, 3)], {"bboost": 1000.0}))
    assert len(chips_played(expiring)) == 1


def test_free_hit_covers_a_blank_and_the_real_squad_returns():
    projections, squad, bank = owned_squad()
    blank = (projections["gameweek"] == 1) & projections["player_id"].isin(squad)
    projections.loc[blank, "predicted_points"] = 0.0
    result = plan_transfers(projections, squad, bank, 1,
                            config([("freehit", 1, 38)], {"freehit": 0.0}))
    assert chips_played(result) == {1: "freehit"}

    week1, week2 = result["plan"][:2]
    assert set(week1["persistent_ids"]) == set(squad)        # the real squad sits out
    assert set(week1["squad"]["player_id"]) != set(squad)
    assert week1["hits"] == 0 and week1["n_transfers"] == 0
    # The Free Hit spends nothing, so the bank still grows by one.
    assert week2["free_transfers"] == min(5, week1["free_transfers"] + 1)
    assert set(week2["squad"]["player_id"]) == set(week2["persistent_ids"])


def test_free_hit_is_only_offered_for_the_coming_week():
    projections, squad, bank = owned_squad()
    blank = (projections["gameweek"] == 2) & projections["player_id"].isin(squad)
    projections.loc[blank, "predicted_points"] = 0.0
    result = plan_transfers(projections, squad, bank, 1,
                            config([("freehit", 1, 38)], {"freehit": 0.0}))
    later = {gw: chip for gw, chip in chips_played(result).items() if gw > 1}
    assert "freehit" not in later.values()


def test_wildcard_rebuilds_without_hits_and_keeps_the_bank():
    projections, squad, bank = owned_squad()
    collapse = (projections["gameweek"] >= 2) & projections["player_id"].isin(squad)
    projections.loc[collapse, "predicted_points"] = 0.0
    result = plan_transfers(projections, squad, bank, 2,
                            config([("wildcard", 2, 38)], {"wildcard": 0.0}))
    assert chips_played(result) == {2: "wildcard"}

    week2 = result["plan"][1]
    assert week2["n_transfers"] > 2           # beyond the usual per-week cap
    assert week2["hits"] == 0
    assert result["plan"][2]["free_transfers"] == min(5, week2["free_transfers"] + 1)


def test_one_chip_per_gameweek():
    projections, squad, bank = owned_squad()
    projections.loc[projections["gameweek"] == 3, "predicted_points"] *= 3
    result = plan_transfers(projections, squad, bank, 1,
                            config([("bboost", 1, 38), ("3xc", 1, 38)],
                                   {"bboost": 0.0, "3xc": 0.0}))
    played = [week["chip"] for week in result["plan"] if week["chip"]]
    weeks = [week["gameweek"] for week in result["plan"] if week["chip"]]
    assert len(played) == 2 and len(set(weeks)) == 2
