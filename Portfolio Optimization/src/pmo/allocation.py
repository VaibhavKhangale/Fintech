"""Asset allocation across broad asset classes.

Covariances here are in the same units as the returns they are used with.
"""
import cvxpy as cp
import numpy as np
import pandas as pd


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance coming from each asset. Shares sum to 1."""
    w = weights[cov.index].to_numpy()
    marginal = cov.to_numpy() @ w
    return pd.Series(w * marginal / (w @ marginal), index=cov.index)


def inverse_volatility(cov: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1 / volatility, ignoring correlations."""
    raw = 1 / np.sqrt(np.diag(cov.to_numpy()))
    return pd.Series(raw / raw.sum(), index=cov.index)


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Weights where each asset contributes equally to portfolio variance."""
    x = cp.Variable(len(cov), pos=True)
    objective = 0.5 * cp.quad_form(x, cp.psd_wrap(cov.to_numpy())) - cp.sum(cp.log(x)) / len(cov)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=cp.CLARABEL)
    if x.value is None:
        raise RuntimeError(f"Risk parity failed with status {problem.status}")
    return pd.Series(x.value / x.value.sum(), index=cov.index)


def implied_returns(prior: pd.Series, cov: pd.DataFrame, risk_aversion: float) -> pd.Series:
    """Excess returns at which the prior weights would be optimal: pi = delta * Sigma * w."""
    return risk_aversion * cov @ prior[cov.index]


def black_litterman(prior: pd.Series, cov: pd.DataFrame, risk_aversion: float,
                    views: pd.DataFrame | None = None, view_returns: pd.Series | None = None,
                    tau: float = 0.05, view_variance: pd.Series | None = None) -> pd.Series:
    """Posterior expected excess returns.

    `views` has one row per view, with asset exposures in its columns; `view_returns` gives the
    expected return of each row. With no views the result is the implied return. By default each
    view is as uncertain as the prior says that portfolio is.
    """
    pi = implied_returns(prior, cov, risk_aversion)
    if views is None:
        return pi
    p = views[cov.index].to_numpy()
    sigma = cov.to_numpy()
    uncertainty = tau * p @ sigma @ p.T
    omega = (np.diag(np.diag(uncertainty)) if view_variance is None
             else np.diag(view_variance.to_numpy()))
    middle = p @ (tau * sigma) @ p.T + omega
    gap = view_returns.to_numpy() - p @ pi.to_numpy()
    shift = (tau * sigma) @ p.T @ np.linalg.solve(middle, gap)
    return pi + pd.Series(shift, index=cov.index)


def best_mix(expected_excess: pd.Series, cov: pd.DataFrame, risk_aversion: float,
             upper: pd.Series | None = None) -> pd.Series:
    """Weights maximizing utility, long-only with no leverage. Any remainder stays in cash."""
    w = cp.Variable(len(cov), nonneg=True)
    rules = [cp.sum(w) <= 1]
    if upper is not None:
        rules.append(w <= upper[cov.index].to_numpy())
    utility = (expected_excess[cov.index].to_numpy() @ w
               - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov.to_numpy())))
    problem = cp.Problem(cp.Maximize(utility), rules)
    problem.solve(solver=cp.CLARABEL)
    if w.value is None:
        raise RuntimeError(f"Allocation failed with status {problem.status}")
    return pd.Series(np.clip(w.value, 0, None), index=cov.index)
