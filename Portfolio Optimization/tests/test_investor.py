"""Tests for utility and the choice of risky share."""
import numpy as np
import pandas as pd
import pytest

from pmo import frontier as fr
from pmo import investor as inv


def test_utility_penalizes_variance():
    assert inv.utility(0.15, 0.2, 4) == pytest.approx(0.15 - 0.5 * 4 * 0.04)


def test_indifference_curve_has_constant_utility():
    vols = np.linspace(0, 0.4, 9)
    curve = inv.indifference_curve(0.07, vols, 4)
    assert inv.utility(curve, vols, 4) == pytest.approx(np.full(9, 0.07))


def test_optimal_share_formula():
    assert inv.optimal_risky_share(0.14, 0.2, 0.06, 4) == pytest.approx(0.5)


def test_optimal_share_maximizes_utility():
    best = inv.optimal_risky_share(0.14, 0.2, 0.06, 4)
    shares = np.linspace(0, 1, 201)
    values = inv.utility(0.06 + shares * 0.08, shares * 0.2, 4)
    assert shares[values.argmax()] == pytest.approx(best)


def test_share_stops_at_one_without_borrowing():
    assert inv.optimal_risky_share(0.3, 0.2, 0.06, 2) == 1.0


def test_share_is_never_negative():
    assert inv.optimal_risky_share(0.03, 0.2, 0.06, 4) == 0.0


def test_borrowing_rate_creates_a_flat_region():
    # Worth levering at the T-bill rate, but not at the borrowing rate.
    share = inv.optimal_risky_share(0.14, 0.2, 0.06, 1.5, borrow_rate=0.13, max_share=2)
    assert share == 1.0


def test_borrowing_share_uses_the_borrowing_rate():
    share = inv.optimal_risky_share(0.20, 0.2, 0.06, 1.0, borrow_rate=0.10, max_share=3)
    assert share == pytest.approx(0.10 / 0.04)


def test_higher_aversion_lowers_the_share():
    shares = [inv.optimal_risky_share(0.14, 0.2, 0.06, a) for a in [3, 4, 8]]
    assert shares == sorted(shares, reverse=True)


def test_optimal_on_frontier_prefers_safer_assets_with_higher_aversion():
    names = ["safe", "risky"]
    mean = pd.Series([0.08, 0.2], index=names)
    cov = pd.DataFrame(np.diag([0.01, 0.09]), index=names, columns=names)
    timid = inv.optimal_on_frontier(mean, cov, 10, fr.Constraints())
    bold = inv.optimal_on_frontier(mean, cov, 1, fr.Constraints())
    assert timid["safe"] > bold["safe"]
    assert timid.sum() == pytest.approx(1)


def test_optimal_on_frontier_matches_two_asset_solution():
    names = ["a", "b"]
    mean = pd.Series([0.1, 0.2], index=names)
    cov = pd.DataFrame(np.diag([0.04, 0.04]), index=names, columns=names)
    # w_b = 0.5 + (mu_b - mu_a) / (2 * A * (sigma_a^2 + sigma_b^2) / 2) with sigma equal
    weights = inv.optimal_on_frontier(mean, cov, 5, fr.Constraints())
    assert weights["b"] == pytest.approx(0.5 + 0.1 / (5 * 0.08), abs=1e-5)


@pytest.fixture
def market():
    names = list("ABC")
    mean = pd.Series([0.09, 0.13, 0.18], index=names)
    cov = pd.DataFrame(np.diag([0.02, 0.04, 0.09]), index=names, columns=names)
    return mean, cov


def test_cautious_investor_holds_tangency_and_cash(market):
    mean, cov = market
    fund = fr.solve_max_sharpe(mean, cov, 0.06, fr.Constraints())
    weights = inv.investor_portfolio(mean, cov, 0.06, 20, fr.Constraints())
    assert weights.sum() < 1
    assert (weights / weights.sum()).to_numpy() == pytest.approx(fund.to_numpy(), abs=1e-5)


def test_bold_investor_moves_up_the_frontier_without_cash(market):
    mean, cov = market
    fund = fr.solve_max_sharpe(mean, cov, 0.06, fr.Constraints())
    weights = inv.investor_portfolio(mean, cov, 0.06, 1, fr.Constraints())
    assert weights.sum() == pytest.approx(1)
    assert weights @ mean > fund @ mean


def test_aversion_from_score_end_points_and_order():
    assert inv.aversion_from_score(0) == pytest.approx(inv.HIGHEST_AVERSION)
    assert inv.aversion_from_score(1) == pytest.approx(inv.LOWEST_AVERSION)
    assert inv.aversion_from_score(0.3) > inv.aversion_from_score(0.7)


def test_aversion_from_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        inv.aversion_from_score(1.2)


def test_loss_limited_share():
    assert inv.loss_limited_share(0.15, 0.30) == pytest.approx(0.5)
    assert inv.loss_limited_share(0.5, 0.30) == 1.0
