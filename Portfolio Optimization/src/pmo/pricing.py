"""Security pricing: CAPM required returns, beta estimates, CAPM tests and dividend valuation.

Annual decimals unless a name says monthly.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pmo.models import fit_factor_model

BLUME_WEIGHT = 2 / 3


def required_return(beta: pd.Series | float, risk_free: float,
                    premium: float) -> pd.Series | float:
    """The security market line: r_f + beta * market premium."""
    return risk_free + beta * premium


def blume_beta(beta: pd.Series | float) -> pd.Series | float:
    """Pull a beta toward 1, because extreme betas tend to drift back."""
    return BLUME_WEIGHT * beta + (1 - BLUME_WEIGHT)


def vasicek_beta(beta: pd.Series, standard_error: pd.Series) -> pd.Series:
    """Shrink each beta toward the cross-sectional mean, more so when it is measured poorly."""
    spread = beta.var()
    return (spread * beta + standard_error ** 2 * beta.mean()) / (spread + standard_error ** 2)


def beta_standard_errors(excess: pd.DataFrame, market: pd.Series) -> pd.Series:
    """Ordinary standard errors of single-index betas."""
    fit = fit_factor_model(excess, market.to_frame("MKT"))
    residual_sd = np.sqrt(fit.residual_variance)
    return residual_sd / (market.loc[excess.index].std() * np.sqrt(len(excess) - 1))


def security_table(excess: pd.DataFrame, market: pd.Series, risk_free: float,
                   premium: float) -> pd.DataFrame:
    """Beta, required and realized return, and alpha relative to the average stock.

    Subtracting the average alpha removes what all stocks share, such as survivorship, so a
    stock's relative alpha compares it only with the others.
    """
    fit = fit_factor_model(excess, market.to_frame("MKT"))
    table = pd.DataFrame({"beta": fit.betas["MKT"]})
    table["required_return"] = required_return(table["beta"], risk_free, premium)
    table["realized_return"] = excess.mean() * 12 + risk_free
    table["alpha"] = fit.alpha * 12
    table["relative_alpha"] = table["alpha"] - table["alpha"].mean()
    standard_error = table["alpha"] / fit.alpha_t
    table["relative_t"] = table["relative_alpha"] / standard_error
    return table


def beta_forecast_errors(excess: pd.DataFrame, market: pd.Series, window: int = 60,
                         horizon: int = 36, step: int = 12) -> pd.Series:
    """How well each beta estimate predicts the beta over the next `horizon` months."""
    months = excess.index
    errors: dict[str, list[float]] = {"Raw": [], "Blume": [], "Vasicek": [], "All equal to 1": []}
    for start in range(window, len(months) - horizon + 1, step):
        train = months[start - window:start]
        test = months[start:start + horizon]
        past = fit_factor_model(excess.loc[train], market.loc[train].to_frame("MKT")).betas["MKT"]
        later = fit_factor_model(excess.loc[test], market.loc[test].to_frame("MKT")).betas["MKT"]
        se = beta_standard_errors(excess.loc[train], market.loc[train])
        forecasts = {"Raw": past, "Blume": blume_beta(past), "Vasicek": vasicek_beta(past, se),
                     "All equal to 1": past * 0 + 1}
        for name, forecast in forecasts.items():
            errors[name].append(float(np.sqrt(((forecast - later) ** 2).mean())))
    return pd.Series({name: float(np.mean(values)) for name, values in errors.items()})


def fama_macbeth(excess: pd.DataFrame, market: pd.Series, window: int = 60) -> pd.DataFrame:
    """Monthly cross-sectional regressions of returns on betas from the prior `window` months.

    Returns the intercept and slope for each month. CAPM predicts an intercept of zero and a slope
    equal to the market's excess return.
    """
    rows = {}
    months = excess.index
    for start in range(window, len(months)):
        train = months[start - window:start]
        beta = fit_factor_model(excess.loc[train], market.loc[train].to_frame("MKT")).betas["MKT"]
        realized = excess.loc[months[start]]
        slope, intercept = np.polyfit(beta.to_numpy(), realized[beta.index].to_numpy(), 1)
        rows[months[start]] = {"intercept": intercept, "slope": slope,
                               "market": market.loc[months[start]]}
    return pd.DataFrame(rows).T


def monthly_mean_test(series: pd.Series) -> dict[str, float]:
    """Annualized mean, its standard error and t-statistic for a monthly series."""
    standard_error = series.std() / np.sqrt(len(series))
    return {"annual_mean": series.mean() * 12, "standard_error": standard_error * 12,
            "t_stat": series.mean() / standard_error}


def rank_portfolio(score: pd.Series, count: int, highest: bool) -> pd.Series:
    """Equal weights on the `count` stocks with the highest or lowest score."""
    picks = score.sort_values(ascending=not highest).index[:count]
    return pd.Series(1 / count, index=picks)


def gordon_value(dividend: float, required: float, growth: float) -> float:
    """Value of a dividend growing forever: D0 (1 + g) / (k - g)."""
    if growth >= required:
        raise ValueError("Growth must be below the required return")
    return dividend * (1 + growth) / (required - growth)


def two_stage_value(dividend: float, required: float, early_growth: float, years: int,
                    late_growth: float) -> float:
    """Value with `early_growth` for `years`, then `late_growth` forever."""
    times = np.arange(1, years + 1)
    early = (dividend * (1 + early_growth) ** times / (1 + required) ** times).sum()
    last = dividend * (1 + early_growth) ** years
    return float(early + gordon_value(last, required, late_growth) / (1 + required) ** years)


def implied_two_stage_growth(price: float, dividend: float, required: float, years: int,
                             late_growth: float) -> float:
    """Early growth rate that makes the two-stage value equal the market price."""
    return float(brentq(lambda g: two_stage_value(dividend, required, g, years, late_growth)
                        - price, -0.5, 2.0))


def implied_growth(price: pd.Series, dividend: pd.Series,
                   required: pd.Series) -> pd.Series:
    """Perpetual dividend growth that makes the Gordon value equal the market price."""
    return (required * price - dividend) / (price + dividend)
