import numpy as np
import pandas as pd
import pytest

from fpl_features import (
    _rolling_over_appearances,
    add_rolling_features,
    apply_benchwarmer_adjustments,
)


def test_clean_sheet_probability_is_nan_without_minutes():
    frame = pd.DataFrame({
        "minutes": [90, 0, 45],
        "expected_goals_conceded": [1.0, 0.0, 0.5],
        "expected_assists": [0.1, 0.0, 0.2],
    })
    out = apply_benchwarmer_adjustments(frame)
    assert out["clean_sheet_probability"].iloc[0] == pytest.approx(np.exp(-1.0))
    assert np.isnan(out["clean_sheet_probability"].iloc[1])
    assert out["clean_sheet_probability"].iloc[2] == pytest.approx(np.exp(-0.5))


def test_rolling_over_appearances_skips_gaps_and_never_sees_the_current_row():
    series = pd.Series([0.5, np.nan, 0.3, 0.7, np.nan, 0.1])
    out = _rolling_over_appearances(series, window=2, min_periods=2)
    expected = [np.nan, np.nan, np.nan, 0.4, 0.5, 0.5]
    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)


def test_rolling_features_per_player_and_causal():
    frame = pd.DataFrame({
        "player_name": ["A"] * 4 + ["B"] * 3,
        "total_points": [2, 6, 1, 9, 3, 3, 12],
        "clean_sheet_probability": [0.6, np.nan, 0.2, 0.4, 0.5, 0.1, np.nan],
    })
    out, created = add_rolling_features(
        frame, "player_name", ["total_points", "clean_sheet_probability"], [2])

    assert set(created) == {"total_points_roll_2", "clean_sheet_probability_roll_2"}
    np.testing.assert_allclose(out["total_points_roll_2"],
                               [np.nan, np.nan, 4.0, 3.5, np.nan, np.nan, 3.0],
                               equal_nan=True)
    # A's appearances before row 3 are 0.6 and 0.2; the missed match is skipped.
    np.testing.assert_allclose(out["clean_sheet_probability_roll_2"],
                               [np.nan, np.nan, np.nan, 0.4, np.nan, np.nan, 0.3],
                               equal_nan=True)
    assert out["games_prior"].tolist() == [0, 1, 2, 3, 0, 1, 2]
