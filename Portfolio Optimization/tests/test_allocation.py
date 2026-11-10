"""Tests for asset allocation methods."""
import numpy as np
import pandas as pd
import pytest

from pmo import allocation as al
from pmo import sources

NAMES = ["Equity", "Gold", "Bonds"]


@pytest.fixture
def cov():
    vols = np.array([0.18, 0.15, 0.05])
    corr = np.array([[1, 0.05, 0.1], [0.05, 1, 0.1], [0.1, 0.1, 1]])
    return pd.DataFrame(np.outer(vols, vols) * corr, index=NAMES, columns=NAMES)


def test_risk_contributions_sum_to_one(cov):
    shares = al.risk_contributions(pd.Series([0.6, 0.2, 0.2], index=NAMES), cov)
    assert shares.sum() == pytest.approx(1)
    assert shares["Equity"] > 0.6


def test_inverse_volatility_for_uncorrelated_assets():
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
    assert al.inverse_volatility(cov).tolist() == pytest.approx([1 / 3, 2 / 3])


def test_risk_parity_equalizes_contributions(cov):
    weights = al.risk_parity(cov)
    assert weights.sum() == pytest.approx(1)
    assert al.risk_contributions(weights, cov).to_numpy() == pytest.approx(
        np.full(3, 1 / 3), abs=1e-4)


def test_risk_parity_matches_inverse_vol_when_uncorrelated():
    cov = pd.DataFrame(np.diag([0.04, 0.01, 0.0225]), index=NAMES, columns=NAMES)
    assert al.risk_parity(cov).to_numpy() == pytest.approx(al.inverse_volatility(cov).to_numpy(),
                                                           abs=1e-4)


def test_implied_returns_recover_the_prior_as_optimal(cov):
    prior = pd.Series([0.6, 0.2, 0.2], index=NAMES)
    pi = al.implied_returns(prior, cov, 3)
    optimal = np.linalg.solve(3 * cov.to_numpy(), pi.to_numpy())
    assert optimal == pytest.approx(prior.to_numpy())


def test_black_litterman_without_views_is_the_implied_return(cov):
    prior = pd.Series([0.6, 0.2, 0.2], index=NAMES)
    assert al.black_litterman(prior, cov, 3).equals(al.implied_returns(prior, cov, 3))


def test_black_litterman_moves_toward_a_view(cov):
    prior = pd.Series([0.6, 0.2, 0.2], index=NAMES)
    pi = al.implied_returns(prior, cov, 3)
    views = pd.DataFrame([[0.0, 1.0, 0.0]], columns=NAMES)
    target = pi["Gold"] + 0.05
    out = al.black_litterman(prior, cov, 3, views, pd.Series([target]))
    assert pi["Gold"] < out["Gold"] < target


def test_black_litterman_certain_view_is_met_exactly(cov):
    prior = pd.Series([0.6, 0.2, 0.2], index=NAMES)
    views = pd.DataFrame([[0.0, 1.0, 0.0]], columns=NAMES)
    out = al.black_litterman(prior, cov, 3, views, pd.Series([0.09]),
                             view_variance=pd.Series([1e-12]))
    assert out["Gold"] == pytest.approx(0.09, abs=1e-6)


def test_best_mix_stays_in_cash_when_nothing_pays(cov):
    weights = al.best_mix(pd.Series([-0.01, -0.01, -0.01], index=NAMES), cov, 4)
    assert weights.sum() == pytest.approx(0, abs=1e-6)


def test_best_mix_respects_limits(cov):
    upper = pd.Series([1.0, 0.1, 1.0], index=NAMES)
    weights = al.best_mix(pd.Series([0.08, 0.3, 0.02], index=NAMES), cov, 2, upper)
    assert weights["Gold"] <= 0.1 + 1e-6
    assert weights.sum() <= 1 + 1e-6


def test_parse_mf_nav_sorts_and_converts():
    payload = {"data": [{"date": "03-02-2020", "nav": "11.5"}, {"date": "31-01-2020", "nav": "11.0"}]}
    nav = sources.parse_mf_nav(payload)
    assert nav.index[0] == pd.Timestamp("2020-01-31")
    assert nav.iloc[-1] == pytest.approx(11.5)
