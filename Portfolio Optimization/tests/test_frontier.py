"""Tests for the frontier, checked against a hand-solved case and the numerical optimizer."""
import numpy as np
import pandas as pd
import pytest

from pmo import frontier as fr

NAMES = ["A", "B", "C", "D"]


@pytest.fixture
def inputs():
    mean = pd.Series([0.08, 0.12, 0.16, 0.10], index=NAMES)
    vols = np.array([0.15, 0.2, 0.3, 0.18])
    corr = np.array([[1, .2, .1, .3], [.2, 1, .4, .1], [.1, .4, 1, .2], [.3, .1, .2, 1]])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=NAMES, columns=NAMES)
    return mean, cov


def test_two_uncorrelated_assets_match_hand_solution():
    mean = pd.Series([0.1, 0.2], index=["A", "B"])
    cov = pd.DataFrame(np.diag([0.04, 0.09]), index=mean.index, columns=mean.index)
    frontier = fr.Frontier(mean, cov)
    assert frontier.gmv_weights().tolist() == pytest.approx([9 / 13, 4 / 13])
    assert frontier.gmv_volatility == pytest.approx(np.sqrt(0.04 * 0.09 / 0.13))
    assert frontier.gmv_return == pytest.approx((0.9 + 0.8) / 13)


def test_weights_meet_target_and_are_fully_invested(inputs):
    mean, cov = inputs
    frontier = fr.Frontier(mean, cov)
    weights = frontier.weights(0.15)
    assert weights.sum() == pytest.approx(1)
    assert weights @ mean == pytest.approx(0.15)
    assert fr.portfolio_stats(weights, mean, cov)[1] == pytest.approx(frontier.volatility(0.15))


def test_gmv_is_lowest_point_on_the_curve(inputs):
    mean, cov = inputs
    frontier = fr.Frontier(mean, cov)
    curve = frontier.curve(0.0, 0.3, 601)
    assert curve["volatility"].min() == pytest.approx(frontier.gmv_volatility, abs=1e-4)
    assert frontier.weights(frontier.gmv_return).to_numpy() == pytest.approx(
        frontier.gmv_weights().to_numpy())


@pytest.mark.parametrize("target", [0.09, 0.12, 0.2])
def test_numerical_optimizer_agrees_with_closed_form(inputs, target):
    mean, cov = inputs
    numerical = fr.solve_min_variance(mean, cov, target)
    assert numerical.to_numpy() == pytest.approx(fr.Frontier(mean, cov).weights(target).to_numpy(),
                                                 abs=1e-5)


def test_two_fund_theorem(inputs):
    mean, cov = inputs
    frontier = fr.Frontier(mean, cov)
    low, high = frontier.weights(0.1), frontier.weights(0.2)
    mixed = 0.5 * low + 0.5 * high
    assert mixed.to_numpy() == pytest.approx(frontier.weights(0.15).to_numpy())


def test_assets_never_beat_the_frontier(inputs):
    mean, cov = inputs
    frontier = fr.Frontier(mean, cov)
    for name in NAMES:
        assert np.sqrt(cov.loc[name, name]) >= frontier.volatility(mean[name]) - 1e-9


def test_equal_means_are_rejected():
    mean = pd.Series([0.1, 0.1], index=["A", "B"])
    cov = pd.DataFrame(np.eye(2), index=mean.index, columns=mean.index)
    with pytest.raises(ValueError):
        fr.Frontier(mean, cov)


def test_annualize_scales_and_adds_risk_free(inputs):
    mean, cov = inputs
    annual_mean, annual_cov = fr.annualize(mean / 12, cov / 12, 0.06)
    assert annual_mean.to_numpy() == pytest.approx((mean + 0.06).to_numpy())
    assert annual_cov.to_numpy() == pytest.approx(cov.to_numpy())


def test_exposure_measures_short_positions():
    out = fr.exposure(pd.Series({"A": 1.5, "B": -0.5}))
    assert out == {"short": 0.5, "gross": 2.0, "largest_long": 1.5, "largest_short": -0.5}


def simulated_returns():
    rng = np.random.default_rng(8)
    return pd.DataFrame(rng.normal(0.01, 0.05, (120, 4)) + rng.normal(0, 0.03, (120, 1)),
                        index=pd.date_range("2010-01-31", periods=120, freq="ME"), columns=NAMES)


def test_bootstrap_shape_and_repeatability():
    data = simulated_returns()
    sample_inputs = lambda frame: (frame.mean(), frame.cov())
    first = fr.bootstrap_weights(data, sample_inputs, 0.06, draws=20)
    second = fr.bootstrap_weights(data, sample_inputs, 0.06, draws=20)
    assert first.shape == (20, 4)
    assert first.equals(second)
    assert first.sum(axis=1).to_numpy() == pytest.approx(np.ones(20))


def test_promised_vs_realized_uses_test_months_only():
    data = simulated_returns()
    sample_inputs = lambda frame: (frame.mean(), frame.cov())
    train, test = data.iloc[:80], data.iloc[80:]
    first = fr.promised_vs_realized(train, test, sample_inputs, 0.06, [0.15])
    changed = fr.promised_vs_realized(train, test * 2, sample_inputs, 0.06, [0.15])
    assert first["promised_volatility"].iloc[0] == pytest.approx(
        changed["promised_volatility"].iloc[0])
    assert changed["realized_volatility"].iloc[0] == pytest.approx(
        first["realized_volatility"].iloc[0] * 2)


SECTORS = pd.Series({"A": "x", "B": "x", "C": "y", "D": "z"})


def test_long_only_has_no_short_positions(inputs):
    mean, cov = inputs
    weights = fr.solve_min_variance(mean, cov, 0.14, fr.Constraints())
    assert weights.min() >= -1e-7
    assert weights.sum() == pytest.approx(1)
    assert weights @ mean == pytest.approx(0.14, abs=1e-6)


def test_caps_are_respected(inputs):
    mean, cov = inputs
    rules = fr.Constraints(upper=0.4, sector_cap=0.6, sectors=SECTORS)
    weights = fr.solve_min_variance(mean, cov, None, rules)
    assert weights.max() <= 0.4 + 1e-7
    assert weights[["A", "B"]].sum() <= 0.6 + 1e-7


def test_constraints_can_only_raise_risk(inputs):
    mean, cov = inputs
    free = fr.solve_min_variance(mean, cov, 0.12)
    capped = fr.solve_min_variance(mean, cov, 0.12, fr.Constraints(upper=0.35))
    assert fr.portfolio_stats(capped, mean, cov)[1] >= fr.portfolio_stats(free, mean, cov)[1] - 1e-7


def test_unconstrained_gmv_matches_closed_form(inputs):
    mean, cov = inputs
    solved = fr.solve_min_variance(mean, cov)
    assert solved.to_numpy() == pytest.approx(fr.Frontier(mean, cov).gmv_weights().to_numpy(),
                                              abs=1e-5)


def test_highest_return_long_only_is_the_best_asset(inputs):
    mean, _ = inputs
    assert fr.highest_return(mean, fr.Constraints()) == pytest.approx(0.16)


def test_highest_return_with_a_cap_splits_across_top_assets(inputs):
    mean, _ = inputs
    assert fr.highest_return(mean, fr.Constraints(upper=0.5)) == pytest.approx(0.14)


def test_infeasible_constraints_raise(inputs):
    mean, cov = inputs
    with pytest.raises(RuntimeError):
        fr.solve_min_variance(mean, cov, None, fr.Constraints(upper=0.2))


def test_efficient_frontier_rises_in_return_and_risk(inputs):
    mean, cov = inputs
    summary, weights = fr.efficient_frontier(mean, cov, fr.Constraints(), points=8)
    assert summary["expected_return"].is_monotonic_increasing
    assert summary["volatility"].diff().dropna().min() > -1e-6
    assert weights.shape == (8, 4)


def test_frontier_point_position_zero_is_lowest_risk(inputs):
    mean, cov = inputs
    rules = fr.Constraints()
    assert fr.frontier_point(mean, cov, rules, 0).to_numpy() == pytest.approx(
        fr.solve_min_variance(mean, cov, None, rules).to_numpy())


def test_effective_stocks():
    assert fr.effective_stocks(pd.Series([0.25] * 4)) == pytest.approx(4)
    assert fr.effective_stocks(pd.Series([1.0, 0.0])) == pytest.approx(1)


def test_bootstrap_accepts_a_constrained_builder():
    data = simulated_returns()
    rules = fr.Constraints()
    out = fr.bootstrap_weights(data, lambda f: (f.mean(), f.cov()), 0.06, draws=5,
                               builder=lambda m, c: fr.solve_min_variance(m, c, None, rules))
    assert out.min().min() >= -1e-7


def test_tangency_has_the_highest_sharpe_on_the_frontier(inputs):
    mean, cov = inputs
    frontier = fr.Frontier(mean, cov)
    best = fr.sharpe_ratio(frontier.tangency_weights(0.04), mean, cov, 0.04)
    curve = frontier.curve(frontier.gmv_return, 0.4, 400)
    sharpes = (curve["expected_return"] - 0.04) / curve["volatility"]
    assert best == pytest.approx(sharpes.max(), abs=1e-5)


def test_tangency_two_uncorrelated_assets_match_hand_solution():
    mean = pd.Series([0.1, 0.2], index=["A", "B"])
    cov = pd.DataFrame(np.diag([0.04, 0.09]), index=mean.index, columns=mean.index)
    weights = fr.Frontier(mean, cov).tangency_weights(0.05)
    assert weights["A"] / weights["B"] == pytest.approx((0.05 / 0.04) / (0.15 / 0.09))


def test_tangency_is_refused_when_risk_free_is_above_gmv(inputs):
    mean, cov = inputs
    frontier = fr.Frontier(mean, cov)
    with pytest.raises(ValueError, match="No tangency"):
        frontier.tangency_weights(frontier.gmv_return + 0.01)


def test_max_sharpe_solver_matches_closed_form(inputs):
    mean, cov = inputs
    solved = fr.solve_max_sharpe(mean, cov, 0.04)
    assert solved.to_numpy() == pytest.approx(
        fr.Frontier(mean, cov).tangency_weights(0.04).to_numpy(), abs=1e-5)


def test_max_sharpe_respects_limits(inputs):
    mean, cov = inputs
    weights = fr.solve_max_sharpe(mean, cov, 0.04, fr.Constraints(upper=0.4))
    assert weights.sum() == pytest.approx(1)
    assert weights.min() >= -1e-7
    assert weights.max() <= 0.4 + 1e-6


def test_constrained_max_sharpe_beats_frontier_points(inputs):
    mean, cov = inputs
    rule = fr.Constraints(upper=0.4)
    best = fr.sharpe_ratio(fr.solve_max_sharpe(mean, cov, 0.04, rule), mean, cov, 0.04)
    for position in [0, 0.25, 0.5, 0.75]:
        point = fr.frontier_point(mean, cov, rule, position)
        assert best >= fr.sharpe_ratio(point, mean, cov, 0.04) - 1e-6


def test_capital_market_line_passes_through_risk_free_and_tangency():
    assert fr.capital_market_line(0.06, 0.5, 0.0) == pytest.approx(0.06)
    assert fr.capital_market_line(0.06, 0.5, 0.2) == pytest.approx(0.16)


def test_tracking_portfolio_recovers_known_weights():
    data = simulated_returns()
    target = data[["A", "B"]] @ pd.Series({"A": 0.3, "B": 0.7})
    weights = fr.tracking_portfolio(target, data)
    assert weights[["A", "B"]].tolist() == pytest.approx([0.3, 0.7], abs=1e-4)


def test_max_sharpe_refuses_shorts_when_risk_free_is_above_gmv(inputs):
    mean, cov = inputs
    rate = fr.Frontier(mean, cov).gmv_return + 0.01
    with pytest.raises(ValueError, match="No tangency"):
        fr.solve_max_sharpe(mean, cov, rate)


def test_long_only_max_sharpe_still_works_above_gmv(inputs):
    mean, cov = inputs
    rate = fr.Frontier(mean, cov).gmv_return + 0.01
    weights = fr.solve_max_sharpe(mean, cov, rate, fr.Constraints())
    assert weights.sum() == pytest.approx(1)
    assert weights @ mean > rate
