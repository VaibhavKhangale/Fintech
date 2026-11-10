"""Tests for security pricing."""
import numpy as np
import pandas as pd
import pytest

from pmo import pricing

MONTHS = pd.date_range("2005-01-31", periods=240, freq="ME")
BETAS = pd.Series({"A": 0.5, "B": 1.0, "C": 1.5})


def capm_world(premium=0.01):
    """Returns that follow CAPM exactly, up to noise."""
    rng = np.random.default_rng(12)
    market = pd.Series(rng.normal(premium, 0.05, 240), index=MONTHS)
    noise = rng.normal(0, 0.02, (240, 3))
    excess = pd.DataFrame(np.outer(market, BETAS) + noise, index=MONTHS, columns=BETAS.index)
    return excess, market


def test_required_return_is_the_sml():
    assert pricing.required_return(1.5, 0.06, 0.08) == pytest.approx(0.18)


def test_required_return_at_beta_zero_and_one():
    assert pricing.required_return(0.0, 0.06, 0.08) == pytest.approx(0.06)
    assert pricing.required_return(1.0, 0.06, 0.08) == pytest.approx(0.14)


def test_blume_beta_leaves_one_unchanged_and_pulls_extremes_in():
    assert pricing.blume_beta(1.0) == pytest.approx(1.0)
    assert pricing.blume_beta(1.6) == pytest.approx(1.4)


def test_vasicek_shrinks_noisy_betas_more():
    beta = pd.Series({"A": 0.5, "B": 1.5})
    se = pd.Series({"A": 0.01, "B": 0.5})
    out = pricing.vasicek_beta(beta, se)
    assert abs(out["A"] - 0.5) < abs(out["B"] - 1.5)


def test_beta_standard_errors_match_statsmodels():
    import statsmodels.api as sm
    excess, market = capm_world()
    fit = sm.OLS(excess["A"], sm.add_constant(market)).fit()
    assert pricing.beta_standard_errors(excess, market)["A"] == pytest.approx(fit.bse.iloc[1],
                                                                              rel=1e-6)


def test_security_table_relative_alpha_sums_to_zero():
    excess, market = capm_world()
    table = pricing.security_table(excess + 0.003, market, 0.06, 0.08)
    assert table["relative_alpha"].sum() == pytest.approx(0)
    assert table["beta"].to_numpy() == pytest.approx(BETAS.to_numpy(), abs=0.05)


def test_beta_forecast_errors_reports_each_method():
    excess, market = capm_world()
    errors = pricing.beta_forecast_errors(excess, market, window=60, horizon=36, step=24)
    assert set(errors.index) == {"Raw", "Blume", "Vasicek", "All equal to 1"}
    assert errors["Raw"] < errors["All equal to 1"]


def test_fama_macbeth_recovers_capm_in_a_capm_world():
    excess, market = capm_world()
    out = pricing.fama_macbeth(excess, market, window=60)
    assert len(out) == 180
    gap = out["slope"] - out["market"]
    assert abs(gap.mean()) < 2.5 * gap.std() / np.sqrt(len(gap))
    assert abs(out["intercept"].mean()) < 2.5 * out["intercept"].std() / np.sqrt(len(out))


def test_fama_macbeth_uses_only_prior_months():
    excess, market = capm_world()
    changed = excess.copy()
    changed.iloc[100:] = changed.iloc[100:] * 3
    first = pricing.fama_macbeth(excess, market, window=60)
    second = pricing.fama_macbeth(changed, market, window=60)
    month = MONTHS[99]
    assert first.loc[month].to_numpy() == pytest.approx(second.loc[month].to_numpy())


def test_monthly_mean_test():
    out = pricing.monthly_mean_test(pd.Series([0.01, 0.03]))
    assert out["annual_mean"] == pytest.approx(0.24)
    assert out["t_stat"] == pytest.approx(0.02 / (np.std([0.01, 0.03], ddof=1) / np.sqrt(2)))


def test_rank_portfolio_picks_extremes():
    score = pd.Series({"A": 1.0, "B": 3.0, "C": 2.0})
    assert list(pricing.rank_portfolio(score, 2, True).index) == ["B", "C"]
    assert list(pricing.rank_portfolio(score, 1, False).index) == ["A"]
    assert pricing.rank_portfolio(score, 2, True).sum() == pytest.approx(1)


def test_gordon_value():
    assert pricing.gordon_value(10, 0.12, 0.07) == pytest.approx(214)


def test_gordon_value_needs_growth_below_required_return():
    with pytest.raises(ValueError):
        pricing.gordon_value(10, 0.1, 0.1)


def test_two_stage_equals_gordon_when_growth_rates_match():
    assert pricing.two_stage_value(10, 0.12, 0.07, 10, 0.07) == pytest.approx(
        pricing.gordon_value(10, 0.12, 0.07))


def test_two_stage_implied_growth_inverts_value():
    price = pricing.two_stage_value(10, 0.12, 0.15, 10, 0.06)
    assert pricing.implied_two_stage_growth(price, 10, 0.12, 10, 0.06) == pytest.approx(0.15)


def test_implied_growth_inverts_gordon():
    price = pd.Series({"A": pricing.gordon_value(10, 0.12, 0.07)})
    dividend = pd.Series({"A": 10.0})
    required = pd.Series({"A": 0.12})
    assert pricing.implied_growth(price, dividend, required)["A"] == pytest.approx(0.07)
