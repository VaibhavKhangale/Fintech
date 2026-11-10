"""Tests for walk-forward portfolio testing."""
import numpy as np
import pandas as pd
import pytest

from pmo import backtest

MONTHS = pd.date_range("2015-01-31", periods=48, freq="ME")


def frame():
    rng = np.random.default_rng(6)
    return pd.DataFrame(rng.normal(0.01, 0.05, (48, 3)), index=MONTHS, columns=list("ABC"))


def equal_weight(train):
    return pd.Series(1 / train.shape[1], index=train.columns)


def test_run_starts_after_the_first_window_and_covers_all_later_months():
    result = backtest.run(frame(), {"EW": equal_weight}, window=24, step=12)
    assert result.returns.index.min() == MONTHS[24]
    assert result.returns.index.max() == MONTHS[-1]
    assert len(result.returns) == 24


def test_strategy_sees_only_earlier_months():
    seen = []

    def spy(train):
        seen.append(train.index.max())
        return equal_weight(train)

    result = backtest.run(frame(), {"spy": spy}, window=24, step=12)
    assert seen == [MONTHS[23], MONTHS[35]]
    assert list(result.weights["spy"].index) == [MONTHS[24], MONTHS[36]]


def test_returns_are_weighted_averages():
    data = frame()
    result = backtest.run(data, {"EW": equal_weight}, window=24, step=12)
    assert result.returns["EW"].iloc[0] == pytest.approx(data.iloc[24].mean())


def test_turnover_counts_one_way_trading():
    weights = pd.DataFrame({"A": [1.0, 0.5, 0.5], "B": [0.0, 0.5, 0.5]})
    assert backtest.turnover(weights) == pytest.approx(0.25)


def test_turnover_counts_positions_entered_and_left():
    weights = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"C": 0.5, "B": 0.5}])
    assert backtest.turnover(weights) == pytest.approx(0.5)


def two_assets():
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame({"A": [0.10, 0.10, 0.10], "B": [0.0, 0.0, 0.0]}, index=index)


def test_simulate_lets_holdings_drift_between_rebalances():
    frame = two_assets()
    targets = {frame.index[0]: pd.Series({"A": 0.5, "B": 0.5})}
    out = backtest.simulate(frame, targets, pd.Series(dtype=float))
    assert out.returns.iloc[0] == pytest.approx(0.05)
    expected = (0.55 * 1.1 + 0.5) / (0.55 + 0.5) - 1
    assert out.returns.iloc[1] == pytest.approx(expected)


def test_simulate_charges_costs_on_the_amount_traded():
    frame = two_assets()
    targets = {frame.index[0]: pd.Series({"A": 1.0})}
    out = backtest.simulate(frame, targets, pd.Series({"A": 0.01}))
    assert out.costs == pytest.approx(0.01)
    assert out.returns.iloc[0] == pytest.approx(0.99 * 1.1 - 1)


def test_simulate_taxes_only_the_gain_on_shares_sold():
    frame = two_assets()
    targets = {frame.index[0]: pd.Series({"A": 1.0}), frame.index[2]: pd.Series({"B": 1.0})}
    out = backtest.simulate(frame, targets, pd.Series(dtype=float), pd.Series({"A": 0.2}))
    assert out.taxes == pytest.approx(0.21 * 0.2)


def test_simulate_charges_no_tax_on_losses():
    frame = two_assets() * -1
    targets = {frame.index[0]: pd.Series({"A": 1.0}), frame.index[2]: pd.Series({"B": 1.0})}
    out = backtest.simulate(frame, targets, pd.Series(dtype=float), pd.Series({"A": 0.2}))
    assert out.taxes == 0


def test_simulate_with_no_costs_and_constant_targets_matches_weighted_return():
    frame = two_assets()
    weights = pd.Series({"A": 0.5, "B": 0.5})
    targets = {month: weights for month in frame.index}
    out = backtest.simulate(frame, targets, pd.Series(dtype=float))
    assert out.returns.to_numpy() == pytest.approx(np.full(3, 0.05))


def test_scorecard_columns_and_values():
    result = backtest.run(frame(), {"EW": equal_weight}, window=24, step=12)
    card = backtest.scorecard(result).loc["EW"]
    series = result.returns["EW"]
    assert card["sharpe"] == pytest.approx(series.mean() / series.std() * np.sqrt(12))
    assert card["turnover"] == pytest.approx(0)
    assert card["short_exposure"] == pytest.approx(0)
