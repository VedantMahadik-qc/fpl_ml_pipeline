import pandas as pd

from fpl_track import gameweek_table, parse_account
from fpl_track import summarise


def history(points, costs=None, chips=()):
    costs = costs or [0] * len(points)
    return {"current": [{"event": gw, "points": p, "event_transfers_cost": c, "rank": 1000 * gw}
                        for gw, (p, c) in enumerate(zip(points, costs), start=1)],
            "chips": [{"event": gw, "name": name} for gw, name in chips]}


BOOTSTRAP = {"events": [
    {"id": 1, "finished": True, "average_entry_score": 50, "highest_score": 120},
    {"id": 2, "finished": True, "average_entry_score": 60, "highest_score": 130},
    {"id": 3, "finished": False, "average_entry_score": 0, "highest_score": 0},
]}


def test_only_finished_gameweeks_from_the_start_count_and_hits_come_off():
    table = gameweek_table(BOOTSTRAP, {"bot": history([70, 55, 0], costs=[0, 8, 0],
                                                      chips=[(2, "bboost")])}, start_gw=1)
    assert table["gameweek"].tolist() == [1, 2]
    assert table["bot_points"].tolist() == [70, 47]
    assert table["bot_hits"].tolist() == [0, 2]
    assert table["bot_chip"].tolist() == ["", "Bench Boost"]

    later = gameweek_table(BOOTSTRAP, {"bot": history([70, 55, 0])}, start_gw=2)
    assert later["gameweek"].tolist() == [2]


def test_summary_is_measured_against_the_average_manager():
    table = gameweek_table(BOOTSTRAP, {"bot": history([70, 55]), "me": history([40, 61])},
                           start_gw=1)
    summary = summarise(table, ["bot", "me"]).set_index("account")
    assert summary.at["bot", "vs_average"] == 15          # +20, -5
    assert summary.at["me", "vs_average"] == -9           # -10, +1
    assert summary.at["bot", "beat_average"] == 1
    assert summary.at["me", "beat_average"] == 1


def test_accounts_are_label_equals_id():
    assert parse_account("bot=5070003") == ("bot", 5070003)
