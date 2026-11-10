"""Tests for performance evaluation."""
import numpy as np
import pandas as pd
import pytest

from pmo import performance as perf

MONTHS = pd.date_range("2010-01-31", periods=120, freq="ME")


def world():
    rng = np.random.default_rng(14)
    market = pd.Series(rng.normal(0.008, 0.05, 120), index=MONTHS)
    noise = pd.Series(rng.normal(0, 0.01, 120), index=MONTHS)
    portfolio = 0.002 + 0.8 * market + noise
    return pd.DataFrame({"P": portfolio}), market


def test_scorecard_recovers_alpha_and_beta():
    frame, market = world()
    row = perf.scorecard(frame, market).loc["P"]
    assert row["beta"] == pytest.approx(0.8, abs=0.03)
    assert row["alpha"] == pytest.approx(0.024, abs=0.012)
    assert row["alpha_t"] > 2


def test_scorecard_definitions():
    frame, market = world()
    row = perf.scorecard(frame, market).loc["P"]
    assert row["sharpe"] == pytest.approx(row["excess_return"] / row["volatility"])
    assert row["treynor"] == pytest.approx(row["excess_return"] / row["beta"])
    assert row["information_ratio"] == pytest.approx(row["alpha"] / row["residual_vol"])


def test_benchmark_has_zero_alpha_and_zero_m_squared():
    _, market = world()
    row = perf.scorecard(market.to_frame("M"), market).loc["M"]
    assert row["alpha"] == pytest.approx(0, abs=1e-10)
    assert row["beta"] == pytest.approx(1)
    assert row["m_squared"] == pytest.approx(0, abs=1e-10)


def test_m_squared_rewards_a_better_sharpe():
    _, market = world()
    better = 0.5 * market + 0.004
    row = perf.scorecard(better.to_frame("B"), market).loc["B"]
    assert row["m_squared"] > 0


def test_alpha_interval_covers_the_estimate_and_repeats():
    frame, market = world()
    portfolio = frame["P"]
    low, high = perf.alpha_interval(portfolio, market, draws=300)
    alpha = perf.scorecard(frame, market).loc["P", "alpha"]
    assert low < alpha < high
    assert perf.alpha_interval(portfolio, market, draws=300) == (low, high)


def test_rolling_alpha_of_exact_line():
    _, market = world()
    portfolio = 0.002 + 0.8 * market
    out = perf.rolling_alpha(portfolio, market, 24)
    assert out.to_numpy() == pytest.approx(np.full(len(out), 0.024))
    assert len(out) == 120 - 23


def test_random_portfolios_shape_and_full_holdings_equal_average():
    rng = np.random.default_rng(1)
    excess = pd.DataFrame(rng.normal(0, 0.05, (120, 6)), index=MONTHS, columns=list("ABCDEF"))
    out = perf.random_portfolios(excess, holdings=6, start=24, draws=5)
    assert out.shape == (96, 5)
    assert out[0].to_numpy() == pytest.approx(excess.mean(axis=1).iloc[24:].to_numpy())


def test_random_portfolios_hold_the_stated_number_of_stocks():
    excess = pd.DataFrame(np.eye(4), index=MONTHS[:4], columns=list("ABCD"))
    out = perf.random_portfolios(excess, holdings=2, start=0, step=4, draws=10)
    assert out.to_numpy().max() == pytest.approx(0.5)
    assert out.sum(axis=0).to_numpy() == pytest.approx(np.ones(10))


def test_percentile_rank():
    assert perf.percentile_rank(3, [1, 2, 3, 4]) == pytest.approx(0.5)
