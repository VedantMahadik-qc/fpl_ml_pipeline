import pandas as pd
import pytest

from fpl_season_sim import next_free_transfers, score_gameweek, selling_price


@pytest.mark.parametrize("purchase, current, expected", [
    (50, 50, 50),
    (50, 51, 50),   # +0.1 profit, half rounds down to nothing
    (50, 53, 51),
    (50, 54, 52),
    (50, 47, 47),   # losses are taken in full
])
def test_selling_price_keeps_half_the_profit(purchase, current, expected):
    assert selling_price(purchase, current) == expected


@pytest.mark.parametrize("free, made, expected", [
    (1, 0, 2),
    (1, 1, 1),
    (2, 3, 1),      # a hit never leaves a negative bank
    (5, 0, 5),      # capped at 5
    (4, 0, 5),
])
def test_free_transfer_bank(free, made, expected):
    assert next_free_transfers(free, made) == expected


def make_squad():
    """1-3-4-3 XI; bench GK, then MID 5.0, DEF 3.0, DEF 2.0 by projection."""
    rows = [
        (1, "GKP", True, 4.0), (2, "GKP", False, 1.0),
        (10, "DEF", True, 4.5), (11, "DEF", True, 4.4), (12, "DEF", True, 4.3),
        (13, "DEF", False, 3.0), (14, "DEF", False, 2.0),
        (20, "MID", True, 6.0), (21, "MID", True, 5.9), (22, "MID", True, 5.8),
        (23, "MID", True, 5.7), (24, "MID", False, 5.0),
        (30, "FWD", True, 8.0), (31, "FWD", True, 7.0), (32, "FWD", True, 6.5),
    ]
    squad = pd.DataFrame(rows, columns=["player_id", "position", "is_starter",
                                        "predicted_points"])
    squad["is_captain"] = squad["player_id"] == 30
    return squad


def everyone_plays(points=2.0):
    return {pid: (points, 90.0) for pid in make_squad()["player_id"]}


def test_full_xi_scores_with_captain_doubled():
    actual = everyone_plays()
    actual[30] = (10.0, 90.0)
    result = score_gameweek(make_squad(), actual)
    assert result["auto_subs"] == []
    assert result["xi_points"] == 10 * 2.0 + 10.0
    assert result["gross_points"] == 30.0 + 10.0


def test_auto_sub_respects_formation_and_vice_takes_the_armband():
    actual = everyone_plays()
    actual[10] = (0.0, 0.0)     # a starting defender misses out
    actual[30] = (0.0, 0.0)     # so does the captain
    actual[1] = (0.0, 0.0)      # and the keeper
    actual[31] = (9.0, 90.0)    # vice-captain = next-best projected starter
    result = score_gameweek(make_squad(), actual)

    subs = dict(result["auto_subs"])
    assert subs[1] == 2                 # keeper for keeper
    # MID 24 is first on the bench but would leave two defenders; DEF 13 is next.
    assert subs[10] == 13
    # The forward is replaced by the bench MID: 3-5-2 is legal.
    assert subs[30] == 24
    assert result["armband_id"] == 31
    assert result["captain_points"] == 9.0


def test_blank_bench_means_no_sub():
    actual = everyone_plays()
    for pid in (13, 14, 24, 2):
        actual.pop(pid)                 # bench has no fixture
    actual[20] = (0.0, 0.0)
    result = score_gameweek(make_squad(), actual)
    assert result["auto_subs"] == []
    assert len(result["final_xi"]) == 11
    assert result["xi_points"] == 10 * 2.0
