import numpy as np
import pandas as pd
import pytest

from fpl_odds import _grid_outcomes, LAMBDA_GRID, attach_odds, fit_poisson_rates, margin_free
from fpl_train import with_odds_masked


def outcomes_for(lambda_home, lambda_away):
    """Exact 1X2 / over-2.5 probabilities for a pair of rates on the grid."""
    grid = _grid_outcomes()
    n = len(LAMBDA_GRID)
    i = int(np.argmin(np.abs(LAMBDA_GRID - lambda_home)))
    j = int(np.argmin(np.abs(LAMBDA_GRID - lambda_away)))
    k = i * n + j
    return grid["home"][k], grid["away"][k], grid["over"][k], LAMBDA_GRID[i], LAMBDA_GRID[j]


@pytest.mark.parametrize("home, away", [(1.6, 1.1), (2.4, 0.7), (0.9, 1.8)])
def test_poisson_fit_recovers_known_rates(home, away):
    p_home, p_away, p_over, exact_home, exact_away = outcomes_for(home, away)
    fitted = fit_poisson_rates(np.array([p_home]), np.array([p_away]), np.array([p_over]))
    assert fitted[0] == pytest.approx([exact_home, exact_away], abs=1e-9)


def test_margin_free_probabilities_sum_to_one():
    prices = pd.DataFrame({"H": [1.5, 3.0], "D": [4.2, 3.4], "A": [6.5, 2.4]})
    probs = margin_free(prices)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] > probs[0, 2]


def test_odds_are_oriented_to_the_players_team():
    table = pd.DataFrame({"season": ["2025-26"], "home_team": ["Arsenal"],
                          "away_team": ["Spurs"], "lambda_home": [2.0],
                          "lambda_away": [0.8], "p_home": [0.6], "p_away": [0.15]})
    rows = pd.DataFrame({
        "season": ["2025-26", "2025-26"],
        "team": ["Arsenal", "Spurs"], "team_id": [1, 18],
        "opponent_team_id": [18, 1], "was_home": [True, False],
    })
    out = attach_odds(rows, table)
    assert out["odds_team_goals"].tolist() == [2.0, 0.8]
    assert out["odds_team_conceded"].tolist() == [0.8, 2.0]
    assert out["odds_win_prob"].tolist() == [0.6, 0.15]
    assert out["odds_cs_prob"].iloc[0] == pytest.approx(np.exp(-0.8))


def test_masked_copy_blanks_only_odds():
    frame = pd.DataFrame({"odds_team_goals": [1.5, 2.0], "fdr_team_attack": [1.2, 1.4]})
    out = with_odds_masked(frame, ["odds_team_goals", "fdr_team_attack"])
    assert len(out) == 4
    assert out["odds_team_goals"].isna().tolist() == [False, False, True, True]
    assert out["fdr_team_attack"].notna().all()
