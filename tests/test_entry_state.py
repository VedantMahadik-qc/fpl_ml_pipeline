import pandas as pd
import pytest

from fpl_transfers import (
    add_unavailable_owned,
    apply_selling_prices,
    purchase_prices,
    replay_free_transfers,
    selling_value,
)


def week(event, transfers=0, cost=0):
    return {"event": event, "event_transfers": transfers, "event_transfers_cost": cost}


def test_free_transfers_bank_up_to_five_and_chip_weeks_keep_them():
    # GW3 wildcard and GW5 free hit report zero transfers, as the API does.
    history = {"current": [week(1), week(2), week(3), week(4), week(5), week(6, 1), week(7)]}
    free, problems = replay_free_transfers(history)
    assert free == 5 and problems == []


def test_hits_are_cross_checked_against_the_replay():
    history = {"current": [week(1), week(2, 3, 8), week(3, 1)]}
    free, problems = replay_free_transfers(history)
    assert free == 1 and problems == []           # 1 FT, 3 made -> 2 hits -> 0 left -> +1

    mismatched = {"current": [week(1), week(2, 2, 0)]}   # a top-up we do not model
    _, problems = replay_free_transfers(mismatched)
    assert len(problems) == 1


@pytest.mark.parametrize("paid, now, expected", [
    (5.5, 6.2, 5.8), (5.0, 5.1, 5.0), (7.0, 6.8, 6.8), (4.5, 4.5, 4.5)])
def test_selling_value_keeps_half_the_rise(paid, now, expected):
    assert selling_value(paid, now) == expected


def test_purchase_prices_skip_free_hit_and_fall_back_to_opening_price():
    transfers = [
        {"element_in": 10, "element_in_cost": 62, "event": 3, "time": "a"},
        {"element_in": 11, "element_in_cost": 80, "event": 5, "time": "b"},   # free hit
    ]
    paid = purchase_prices(transfers, [10, 11, 12], free_hit_weeks=[5],
                           opening_prices={11: 7.5, 12: 4.0})
    assert paid == {10: 6.2, 11: 7.5, 12: 4.0}


def test_owned_players_sell_at_selling_price_and_injured_ones_stay():
    projections = pd.DataFrame({
        "player_id": [1, 1, 2], "gameweek": [8, 9, 8], "predicted_points": [5, 4, 3],
        "cost": [6.2, 6.2, 5.0], "position": ["MID", "MID", "DEF"],
        "team": ["A", "A", "B"], "player_name": ["P1", "P1", "P2"]})
    priced = apply_selling_prices(projections, {1: 5.5})
    assert priced.loc[priced["player_id"] == 1, "cost"].tolist() == [5.8, 5.8]
    assert priced.loc[priced["player_id"] == 2, "cost"].tolist() == [5.0]

    everyone = pd.DataFrame({"element_id": [3], "now_cost": [9.0], "position": ["FWD"],
                             "team": ["C"], "player_name": ["Injured"],
                             "web_name": ["Inj"], "status": ["i"]})
    out = add_unavailable_owned(projections, everyone, owned=[1, 2, 3])
    injured = out[out["player_id"] == 3]
    assert len(injured) == 1 and injured["predicted_points"].iloc[0] == 0.0
    assert injured["gameweek"].iloc[0] == 8


def test_chance_of_playing_scales_the_appearance_probability():
    import numpy as np
    from fpl_live_inference import predict_points

    class Stage:
        def __init__(self, value): self.value = value
        def predict_proba(self, X): return np.column_stack([1 - np.full(len(X), self.value),
                                                            np.full(len(X), self.value)])
        def predict(self, X): return np.full(len(X), self.value)

    frame = pd.DataFrame({"position": ["MID", "MID", "MID"], "f": [1.0, 1.0, 1.0],
                          "chance_of_playing": [None, 75, 25]})
    bundles = {"MID": {"classifier": Stage(0.8), "regressor": Stage(5.0), "features": ["f"]}}
    out = predict_points(frame, bundles)
    np.testing.assert_allclose(out["prob_play"], [0.8, 0.6, 0.2])
    np.testing.assert_allclose(out["predicted_points"], [4.0, 3.0, 1.0])
