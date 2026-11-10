"""Portfolio performance evaluation. Inputs are monthly returns in excess of T-bills."""
import numpy as np
import pandas as pd

from pmo import stats
from pmo.models import fit_factor_model

MONTHS = stats.MONTHS_PER_YEAR


def scorecard(excess: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Jensen's alpha and the standard risk-adjusted measures for each portfolio."""
    fit = fit_factor_model(excess, benchmark.to_frame("MKT"))
    benchmark = benchmark.loc[excess.index]
    table = pd.DataFrame({
        "excess_return": excess.mean() * MONTHS,
        "volatility": excess.std() * np.sqrt(MONTHS),
        "beta": fit.betas["MKT"],
        "alpha": fit.alpha * MONTHS,
        "alpha_t": fit.alpha_t,
    })
    table["sharpe"] = table["excess_return"] / table["volatility"]
    table["treynor"] = table["excess_return"] / table["beta"]
    table["residual_vol"] = fit.residuals.std() * np.sqrt(MONTHS)
    table["information_ratio"] = table["alpha"] / table["residual_vol"]
    # M-squared: excess return after levering or diluting to the benchmark's volatility.
    table["m_squared"] = (table["sharpe"] * benchmark.std() * np.sqrt(MONTHS)
                          - benchmark.mean() * MONTHS)
    table["sortino"] = table["excess_return"] / [stats.downside_deviation(excess[c])
                                                 for c in excess]
    table["max_drawdown"] = [stats.max_drawdown(excess[c]) for c in excess]
    return table


def alpha_interval(portfolio: pd.Series, benchmark: pd.Series, draws: int = 2000, block: int = 6,
                   level: float = 0.95, seed: int = 21) -> tuple[float, float]:
    """Bootstrap range for annual alpha, resampling blocks so market runs stay intact."""
    pairs = pd.concat([portfolio, benchmark], axis=1, join="inner").to_numpy()
    rng = np.random.default_rng(seed)
    length = len(pairs)
    blocks = int(np.ceil(length / block))
    alphas = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, length - block + 1, blocks)
        index = (starts[:, None] + np.arange(block)).ravel()[:length]
        y, x = pairs[index, 0], pairs[index, 1]
        beta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
        alphas[i] = (y.mean() - beta * x.mean()) * MONTHS
    tail = (1 - level) / 2
    low, high = np.quantile(alphas, [tail, 1 - tail])
    return float(low), float(high)


def rolling_alpha(portfolio: pd.Series, benchmark: pd.Series, window: int = 36) -> pd.Series:
    """Annual alpha over each trailing window."""
    cov = portfolio.rolling(window).cov(benchmark)
    beta = cov / benchmark.rolling(window).var()
    return ((portfolio.rolling(window).mean() - beta * benchmark.rolling(window).mean())
            * MONTHS).dropna()


def random_portfolios(excess: pd.DataFrame, holdings: int, start: int, step: int = 12,
                      draws: int = 2000, seed: int = 31) -> pd.DataFrame:
    """Monthly returns of random equal-weight portfolios, redrawn every `step` months.

    They hold the same stocks over the same months as a real strategy and use no information, so
    they show what chance alone produces in this universe, survivorship included.
    """
    rng = np.random.default_rng(seed)
    values = excess.to_numpy()
    assets = values.shape[1]
    months = values.shape[0]
    out = np.empty((months - start, draws))
    for begin in range(start, months, step):
        stop = min(begin + step, months)
        order = rng.random((draws, assets)).argsort(axis=1)[:, :holdings]
        mask = np.zeros((draws, assets))
        np.put_along_axis(mask, order, 1 / holdings, axis=1)
        out[begin - start:stop - start] = values[begin:stop] @ mask.T
    return pd.DataFrame(out, index=excess.index[start:])


def percentile_rank(value: float, distribution: pd.Series | np.ndarray) -> float:
    """Share of the distribution below `value`."""
    return float(np.mean(np.asarray(distribution) < value))
