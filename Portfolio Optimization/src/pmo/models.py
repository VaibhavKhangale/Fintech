"""Return generating models and the estimators built on them.

All functions work in monthly excess returns unless a name says otherwise.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

HAC_LAGS = 3


@dataclass(frozen=True)
class FactorModel:
    """Fitted r_i - r_f = alpha_i + B_i F + e_i for each asset."""
    alpha: pd.Series
    alpha_t: pd.Series          # Newey-West t-statistics
    betas: pd.DataFrame         # assets by factors
    residuals: pd.DataFrame
    r_squared: pd.Series
    factor_cov: pd.DataFrame
    factor_mean: pd.Series

    @property
    def residual_variance(self) -> pd.Series:
        return self.residuals.var(ddof=self.betas.shape[1] + 1)

    def covariance(self) -> pd.DataFrame:
        """Model covariance, B Sigma_F B' + D, which assumes residuals are uncorrelated."""
        common = self.betas.to_numpy() @ self.factor_cov.to_numpy() @ self.betas.to_numpy().T
        total = common + np.diag(self.residual_variance.to_numpy())
        return pd.DataFrame(total, index=self.betas.index, columns=self.betas.index)

    def expected_excess(self, premia: pd.Series) -> pd.Series:
        """Expected excess return with alpha set to zero, as no-arbitrage pricing implies."""
        return self.betas[premia.index] @ premia


def fit_factor_model(excess: pd.DataFrame, factors: pd.DataFrame) -> FactorModel:
    """Regress each asset on the factors over their common months."""
    common = excess.index.intersection(factors.index)
    y, x = excess.loc[common], factors.loc[common]
    design = sm.add_constant(x)
    alpha, alpha_t, betas, r2, residuals = {}, {}, {}, {}, {}
    for name in y:
        fit = sm.OLS(y[name], design).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        alpha[name] = fit.params["const"]
        alpha_t[name] = fit.tvalues["const"]
        betas[name] = fit.params[x.columns]
        r2[name] = fit.rsquared
        residuals[name] = fit.resid
    return FactorModel(
        alpha=pd.Series(alpha), alpha_t=pd.Series(alpha_t), betas=pd.DataFrame(betas).T,
        residuals=pd.DataFrame(residuals), r_squared=pd.Series(r2),
        factor_cov=x.cov(), factor_mean=x.mean(),
    )


def ledoit_wolf_covariance(excess: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance shrunk toward a scaled identity matrix."""
    fitted = LedoitWolf().fit(excess.to_numpy())
    return pd.DataFrame(fitted.covariance_, index=excess.columns, columns=excess.columns)


def sector_block_covariance(model: FactorModel, sectors: pd.Series) -> pd.DataFrame:
    """Model covariance that keeps residual covariance within sectors and drops it across them."""
    within = model.residuals.cov().to_numpy()
    labels = sectors[model.betas.index].to_numpy()
    same_sector = labels[:, None] == labels[None, :]
    diagonal_removed = model.covariance().to_numpy() - np.diag(model.residual_variance.to_numpy())
    total = diagonal_removed + within * same_sector
    return pd.DataFrame(total, index=model.betas.index, columns=model.betas.index)


def average_correlation_by_group(residuals: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """Average residual correlation for pairs in the same group and in different groups."""
    corr = residuals.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    same = pd.Series([groups[a] == groups[b] for a, b in upper.index], index=upper.index)
    return pd.Series({"same_group": upper[same].mean(), "different_group": upper[~same].mean()})


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Fully invested weights with the lowest variance, short sales allowed."""
    raw = np.linalg.solve(cov.to_numpy(), np.ones(len(cov)))
    return pd.Series(raw / raw.sum(), index=cov.index)


def covariance_estimators(excess: pd.DataFrame, market: pd.Series, factors: pd.DataFrame,
                          sectors: pd.Series) -> dict[str, pd.DataFrame]:
    """Each candidate covariance matrix, fitted on the same window."""
    single = fit_factor_model(excess, market.to_frame("MKT"))
    return {
        "Sample": excess.cov(),
        "Single index": single.covariance(),
        "Single index + sectors": sector_block_covariance(single, sectors),
        "Four factor": fit_factor_model(excess, factors).covariance(),
        "Ledoit-Wolf": ledoit_wolf_covariance(excess),
    }


def return_estimators(excess: pd.DataFrame, market: pd.Series, factors: pd.DataFrame,
                      premia: pd.Series) -> dict[str, pd.Series]:
    """Each candidate monthly expected excess return.

    `premia` holds long-run factor means known at the estimation date. The historical means use
    the fitting window only; the grand mean gives every stock the cross-sectional average.
    """
    history = excess.mean()
    single = fit_factor_model(excess, market.to_frame("MKT"))
    four = fit_factor_model(excess, factors)
    return {
        "Historical mean": history,
        "Grand mean": pd.Series(history.mean(), index=history.index),
        "Shrunk mean": 0.5 * history + 0.5 * history.mean(),
        "CAPM": single.betas["MKT"] * premia["MKT_RF"],
        "Blend": 0.5 * history + 0.5 * single.betas["MKT"] * premia["MKT_RF"],
        "Four factor": four.expected_excess(premia[factors.columns]),
    }


def textbook_inputs(excess: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly expected excess returns and covariance from raw sample statistics."""
    return excess.mean(), excess.cov()


def practical_inputs(excess: pd.DataFrame, market: pd.Series,
                     market_premium: float) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly inputs chosen in Stage 2: blended returns and single-index covariance."""
    single = fit_factor_model(excess, market.to_frame("MKT"))
    blended = 0.5 * excess.mean() + 0.5 * single.betas["MKT"] * market_premium
    return blended, single.covariance()


def _origins(index: pd.DatetimeIndex, window: int, step: int) -> list[int]:
    return list(range(window, len(index) - step + 1, step))


def walk_forward_covariance(excess: pd.DataFrame, market: pd.Series, factors: pd.DataFrame,
                            sectors: pd.Series, window: int = 60,
                            step: int = 12) -> pd.DataFrame:
    """Out-of-sample monthly returns of the minimum variance portfolio for each estimator.

    Refit every `step` months on the previous `window` months, then hold the weights.
    """
    months = excess.index
    paths: dict[str, list[pd.Series]] = {}
    for start in _origins(months, window, step):
        train = months[start - window:start]
        test = months[start:start + step]
        estimates = covariance_estimators(excess.loc[train], market.loc[train],
                                          factors.loc[train], sectors)
        for name, cov in estimates.items():
            held = excess.loc[test] @ min_variance_weights(cov)
            paths.setdefault(name, []).append(held)
    return pd.DataFrame({name: pd.concat(parts) for name, parts in paths.items()})


def walk_forward_returns(excess: pd.DataFrame, market: pd.Series, factors: pd.DataFrame,
                         long_factors: pd.DataFrame, window: int = 60,
                         step: int = 12) -> pd.DataFrame:
    """Score return forecasts against the next `step` months' realized mean, per origin.

    `long_factors` is the full factor history. Premia use only months before each origin.
    """
    months = excess.index
    rows = []
    for start in _origins(months, window, step):
        train = months[start - window:start]
        test = months[start:start + step]
        premia = long_factors.loc[:train[-1]].mean()
        realized = excess.loc[test].mean()
        estimates = return_estimators(excess.loc[train], market.loc[train], factors.loc[train],
                                      premia)
        for name, forecast in estimates.items():
            rows.append({
                "origin": train[-1], "estimator": name,
                "rmse": float(np.sqrt(((forecast - realized) ** 2).mean())) * 12,
                "bias": float((forecast - realized).mean()) * 12,
                "rank_corr": (forecast.corr(realized, method="spearman")
                              if forecast.nunique() > 1 else np.nan),
            })
    return pd.DataFrame(rows)
