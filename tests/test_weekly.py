import pandas as pd

from fpl_weekly import previous_seasons, readable_deadline, render_plan


def test_trains_on_the_seasons_before_the_live_one():
    assert previous_seasons("2026-27", 3) == ["2023-24", "2024-25", "2025-26"]
    assert previous_seasons("2027-28", 2) == ["2025-26", "2026-27"]


def test_deadline_is_human_readable():
    assert "17:30 UTC" in readable_deadline("2026-10-23T17:30:00Z")
    assert readable_deadline(None) == "unknown"


def test_report_escapes_names_and_marks_the_bench():
    squad = pd.DataFrame({
        "player_id": [1, 2], "player_name": ["<Gyökeres>", "Bench Guy"],
        "position": ["FWD", "DEF"], "team": ["Arsenal", "Spurs"], "cost": [9.0, 4.0],
        "is_starter": [True, False], "is_captain": [True, False],
        "predicted_points": [6.1, 1.0], "fixtures": ["CHE (H)", "ARS (A)"]})
    week = {"gameweek": 8, "transfers_out": ["Old"], "transfers_in": ["<Gyökeres>"],
            "n_transfers": 1, "hits": 0, "hit_cost": 0, "free_transfers": 2,
            "bank": 0.5, "net_points": 50.0, "squad": squad}
    page = render_plan({"plan": [week]})
    assert "&lt;Gyökeres&gt;" in page and "<Gyökeres>" not in page
    assert ">Bench<" in page and "make these now" in page
