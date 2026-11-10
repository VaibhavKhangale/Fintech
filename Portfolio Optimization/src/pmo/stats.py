"""Return and risk statistics for monthly return series."""
import numpy as np
import pandas as pd
from scipy import stats as sps

MONTHS_PER_YEAR = 12


def geometric_return(returns: pd.Series) -> float:
    """Annualized compound growth rate."""
    return float((1 + returns).prod() ** (MONTHS_PER_YEAR / len(returns)) - 1)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss, as a negative number."""
    wealth = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()])
    return float((wealth / wealth.cummax() - 1).min())


def downside_deviation(excess: pd.Series) -> float:
    """Annualized semi-deviation of returns below the risk-free rate."""
    return float(np.sqrt((excess.clip(upper=0) ** 2).mean() * MONTHS_PER_YEAR))


def historical_cvar(returns: pd.Series, tail: float = 0.05) -> float:
    """Average monthly return in the worst `tail` share of months."""
    return float(returns[returns <= returns.quantile(tail)].mean())


def asset_summary(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """One row of statistics per asset. Means and volatilities are annualized."""
    rows = {}
    years = np.sqrt(len(returns) / MONTHS_PER_YEAR)
    for name, series in returns.items():
        excess = series - rf.reindex(series.index)
        vol = series.std() * np.sqrt(MONTHS_PER_YEAR)
        mean_excess = excess.mean() * MONTHS_PER_YEAR
        downside = downside_deviation(excess)
        rows[name] = {
            "mean": series.mean() * MONTHS_PER_YEAR,
            "cagr": geometric_return(series),
            "volatility": vol,
            "mean_se": vol / years,
            "sharpe": mean_excess / (excess.std() * np.sqrt(MONTHS_PER_YEAR)),
            "sortino": mean_excess / downside,
            "max_drawdown": max_drawdown(series),
            "cvar_5": historical_cvar(series),
            "skew": series.skew(),
            "excess_kurtosis": series.kurt(),
            "jarque_bera_p": sps.jarque_bera(series).pvalue,
        }
    return pd.DataFrame(rows).T


def equal_weight_volatility(cov: pd.DataFrame) -> float:
    """Annualized volatility of an equal-weight portfolio, given a monthly covariance matrix."""
    weights = np.full(len(cov), 1 / len(cov))
    return float(np.sqrt(weights @ cov.to_numpy() @ weights * MONTHS_PER_YEAR))


def diversification_curve(returns: pd.DataFrame, draws: int = 2000,
                          seed: int = 7) -> pd.DataFrame:
    """Average equal-weight volatility for random portfolios of each size."""
    rng = np.random.default_rng(seed)
    cov = returns.cov()
    names = np.array(returns.columns)
    rows = []
    for size in range(1, len(names) + 1):
        vols = [equal_weight_volatility(cov.loc[pick, pick])
                for pick in (rng.choice(names, size, replace=False) for _ in range(draws))]
        rows.append({"size": size, "mean_volatility": np.mean(vols)})
    return pd.DataFrame(rows).set_index("size")


def diversification_floor(returns: pd.DataFrame) -> float:
    """Annualized volatility left with many similar stocks: the root of the average covariance."""
    cov = returns.cov().to_numpy()
    average_covariance = cov[~np.eye(len(cov), dtype=bool)].mean()
    return float(np.sqrt(average_covariance * MONTHS_PER_YEAR))


def capital_allocation(risky_mean: float, risky_vol: float, rf: float,
                       weights: list[float] | np.ndarray,
                       borrow_rate: float | None = None) -> pd.DataFrame:
    """Expected return and risk of mixing one risky asset with the risk-free asset.

    A weight above 1 means borrowing to hold more of the risky asset, at `borrow_rate` if given.
    """
    w = np.asarray(weights, dtype=float)
    financing = np.where(w > 1, rf if borrow_rate is None else borrow_rate, rf)
    return pd.DataFrame({
        "risky_weight": w,
        "expected_return": financing + w * (risky_mean - financing),
        "volatility": np.abs(w) * risky_vol,
    })
