"""Tests for return and risk statistics, using hand-checked values."""
import numpy as np
import pandas as pd
import pytest

from pmo import returns, stats


def monthly(values, start="2020-01-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"))


def test_geometric_return_annualizes_compound_growth():
    assert stats.geometric_return(monthly([0.01] * 24)) == pytest.approx(1.01 ** 12 - 1)


def test_geometric_return_is_below_arithmetic_for_volatile_series():
    series = monthly([0.2, -0.2] * 12)
    assert stats.geometric_return(series) < series.mean() * 12


def test_max_drawdown_measures_peak_to_trough():
    assert stats.max_drawdown(monthly([0.5, -0.5, 0.1])) == pytest.approx(-0.5)


def test_max_drawdown_counts_loss_from_the_start():
    assert stats.max_drawdown(monthly([-0.1, -0.1])) == pytest.approx(-0.19)


def test_downside_deviation_ignores_gains():
    gains_only = monthly([0.01, 0.02, 0.03])
    assert stats.downside_deviation(gains_only) == 0
    mixed = monthly([0.02, -0.02])
    assert stats.downside_deviation(mixed) == pytest.approx(np.sqrt(0.0002 * 12))


def test_historical_cvar_averages_the_worst_months():
    series = monthly(list(np.arange(-10, 10) / 100))
    assert stats.historical_cvar(series, tail=0.10) == pytest.approx(-0.095)


def test_asset_summary_values():
    rng = np.random.default_rng(0)
    series = monthly(rng.normal(0.01, 0.05, 240))
    rf = pd.Series(0.002, index=series.index)
    row = stats.asset_summary(series.to_frame("A"), rf).loc["A"]
    assert row["mean"] == pytest.approx(series.mean() * 12)
    assert row["volatility"] == pytest.approx(series.std() * np.sqrt(12))
    assert row["mean_se"] == pytest.approx(series.std() * 12 / np.sqrt(240))
    assert row["sharpe"] == pytest.approx((series.mean() - 0.002) / series.std() * np.sqrt(12))


def test_equal_weight_volatility_of_uncorrelated_assets():
    cov = pd.DataFrame(np.eye(4) * 0.01)
    assert stats.equal_weight_volatility(cov) == pytest.approx(np.sqrt(0.01 / 4 * 12))


def test_diversification_curve_falls_and_ends_at_full_portfolio():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(0, 0.05, (120, 8)), columns=list("ABCDEFGH"))
    curve = stats.diversification_curve(frame, draws=50)
    assert curve["mean_volatility"].iloc[-1] < curve["mean_volatility"].iloc[0]
    assert curve["mean_volatility"].iloc[-1] == pytest.approx(
        stats.equal_weight_volatility(frame.cov()))


def test_diversification_floor_with_constant_covariance():
    # Four assets sharing one factor: the average covariance is the factor variance.
    rng = np.random.default_rng(2)
    factor = rng.normal(0, 0.04, 100_000)
    frame = pd.DataFrame({c: factor + rng.normal(0, 0.05, 100_000) for c in "ABCD"})
    assert stats.diversification_floor(frame) == pytest.approx(0.04 * np.sqrt(12), rel=0.02)


def test_capital_allocation_end_points():
    out = stats.capital_allocation(0.14, 0.2, 0.06, [0, 1])
    assert out["expected_return"].tolist() == pytest.approx([0.06, 0.14])
    assert out["volatility"].tolist() == pytest.approx([0, 0.2])


def test_capital_allocation_slope_is_the_sharpe_ratio():
    out = stats.capital_allocation(0.14, 0.2, 0.06, [0.25, 0.75])
    slope = out["expected_return"].diff().iloc[1] / out["volatility"].diff().iloc[1]
    assert slope == pytest.approx(0.4)


def test_capital_allocation_charges_borrow_rate_only_when_leveraged():
    out = stats.capital_allocation(0.14, 0.2, 0.06, [1.0, 1.5], borrow_rate=0.09)
    assert out["expected_return"].tolist() == pytest.approx([0.14, 0.09 + 1.5 * 0.05])


def test_select_period_filters_by_label():
    frame = pd.DataFrame({"A": range(4)}, index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    periods = returns.split_periods(frame.index, 1)
    assert len(returns.select_period(frame, periods, "estimation")) == 3
    assert len(returns.select_period(frame, periods, "holdout")) == 1
