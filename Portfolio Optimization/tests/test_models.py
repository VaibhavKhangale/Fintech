"""Tests for return generating models, using simulated data with known parameters."""
import numpy as np
import pandas as pd
import pytest

from pmo import models

MONTHS = pd.date_range("2000-01-31", periods=600, freq="ME")
SECTORS = pd.Series({"A": "x", "B": "x", "C": "y"})


@pytest.fixture
def world():
    rng = np.random.default_rng(3)
    factors = pd.DataFrame(rng.normal(0.005, 0.04, (600, 2)), index=MONTHS, columns=["F1", "F2"])
    betas = pd.DataFrame({"F1": [0.5, 1.0, 1.5], "F2": [0.2, -0.3, 0.0]}, index=["A", "B", "C"])
    noise = rng.normal(0, 0.03, (600, 3))
    excess = pd.DataFrame(factors.to_numpy() @ betas.to_numpy().T + noise, index=MONTHS,
                          columns=betas.index)
    return excess, factors, betas


def test_fit_recovers_betas_and_zero_alpha(world):
    excess, factors, betas = world
    model = models.fit_factor_model(excess, factors)
    assert model.betas.to_numpy() == pytest.approx(betas.to_numpy(), abs=0.1)
    assert model.alpha.abs().max() < 0.01
    assert model.r_squared.between(0, 1).all()


def test_fit_uses_only_common_months(world):
    excess, factors, _ = world
    model = models.fit_factor_model(excess.iloc[:300], factors)
    assert len(model.residuals) == 300


def test_model_covariance_matches_sample_when_model_is_true(world):
    excess, factors, _ = world
    model_cov = models.fit_factor_model(excess, factors).covariance()
    assert model_cov.to_numpy() == pytest.approx(excess.cov().to_numpy(), abs=0.0004)


def test_model_covariance_is_positive_definite_and_symmetric(world):
    excess, factors, _ = world
    cov = models.fit_factor_model(excess, factors).covariance().to_numpy()
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_expected_excess_ignores_alpha():
    index = MONTHS[:120]
    market = pd.Series(np.random.default_rng(4).normal(0.005, 0.04, 120), index=index, name="MKT")
    excess = (2 * market + 0.01).to_frame("A")
    model = models.fit_factor_model(excess, market.to_frame())
    assert model.alpha["A"] == pytest.approx(0.01)
    premia = pd.Series({"MKT": 0.005})
    assert model.expected_excess(premia)["A"] == pytest.approx(0.01)


def test_sector_block_covariance_keeps_only_within_sector_residuals(world):
    excess, factors, _ = world
    model = models.fit_factor_model(excess, factors)
    base = model.covariance()
    blocks = models.sector_block_covariance(model, SECTORS)
    within = model.residuals.cov()
    assert blocks.loc["A", "B"] == pytest.approx(base.loc["A", "B"] + within.loc["A", "B"])
    assert blocks.loc["A", "C"] == pytest.approx(base.loc["A", "C"])
    assert np.linalg.eigvalsh(blocks.to_numpy()).min() > 0


def test_ledoit_wolf_keeps_labels_and_pulls_correlation_toward_zero(world):
    excess, _, _ = world
    shrunk = models.ledoit_wolf_covariance(excess.iloc[:30])
    assert list(shrunk.columns) == list(excess.columns)
    sample = excess.iloc[:30].cov()
    correlation = lambda c: c.loc["B", "C"] / np.sqrt(c.loc["B", "B"] * c.loc["C", "C"])
    assert abs(correlation(shrunk)) <= abs(correlation(sample))


def test_average_correlation_by_group_separates_pairs():
    rng = np.random.default_rng(5)
    shared = rng.normal(size=500)
    residuals = pd.DataFrame({
        "A": shared + rng.normal(0, 0.1, 500), "B": shared + rng.normal(0, 0.1, 500),
        "C": rng.normal(size=500)})
    groups = pd.Series({"A": "x", "B": "x", "C": "y"})
    out = models.average_correlation_by_group(residuals, groups)
    assert out["same_group"] > 0.9
    assert abs(out["different_group"]) < 0.1


def test_min_variance_weights_for_uncorrelated_assets():
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])
    weights = models.min_variance_weights(cov)
    assert weights.tolist() == pytest.approx([0.8, 0.2])


def test_walk_forward_covariance_only_uses_later_months(world):
    excess, factors, _ = world
    market = factors["F1"]
    paths = models.walk_forward_covariance(excess, market, factors, SECTORS, window=120, step=60)
    assert paths.index.min() == MONTHS[120]
    assert set(paths.columns) == {"Sample", "Single index", "Single index + sectors",
                                  "Four factor", "Ledoit-Wolf"}
    assert paths.notna().all().all()


def test_walk_forward_has_no_look_ahead(world):
    excess, factors, _ = world
    market = factors["F1"]
    changed = excess.copy()
    changed.iloc[300:] = changed.iloc[300:] * 5
    first = models.walk_forward_covariance(excess, market, factors, SECTORS, 240, 60)
    second = models.walk_forward_covariance(changed, market, factors, SECTORS, 240, 60)
    test_month = MONTHS[250]
    assert first.loc[test_month].to_numpy() == pytest.approx(second.loc[test_month].to_numpy())


def test_walk_forward_returns_scores_every_estimator(world):
    excess, factors, _ = world
    named = factors.rename(columns={"F1": "MKT_RF"})
    out = models.walk_forward_returns(excess, named["MKT_RF"], named, named, window=120, step=60)
    assert set(out["estimator"]) == {"Historical mean", "Grand mean", "Shrunk mean", "CAPM",
                                     "Blend", "Four factor"}
    assert (out["rmse"] >= 0).all()


def test_textbook_inputs_are_sample_statistics(world):
    excess, _, _ = world
    mean, cov = models.textbook_inputs(excess)
    assert mean.equals(excess.mean())
    assert cov.equals(excess.cov())


def test_practical_inputs_average_history_and_capm(world):
    excess, factors, _ = world
    market = factors["F1"]
    mean, cov = models.practical_inputs(excess, market, 0.006)
    model = models.fit_factor_model(excess, market.to_frame("MKT"))
    assert mean["A"] == pytest.approx(0.5 * excess["A"].mean() + 0.5 * model.betas.loc["A", "MKT"] * 0.006)
    assert cov.equals(model.covariance())
