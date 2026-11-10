"""Investor preferences: utility, indifference curves and the choice of risky share.

Returns and volatilities are annual decimals. Risk aversion A is the coefficient in
U = E(r) - A * sigma^2 / 2.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from pmo.frontier import Constraints, portfolio_stats, solve_max_sharpe


@dataclass(frozen=True)
class Profile:
    """An illustrative investor type, not a recommendation for any real person."""
    name: str
    risk_aversion: float


PROFILES = (Profile("Cautious", 8.0), Profile("Balanced", 4.0), Profile("Growth", 2.0))

LOWEST_AVERSION, HIGHEST_AVERSION = 1.5, 10.0


def utility(expected: float | np.ndarray, volatility: float | np.ndarray,
            aversion: float) -> float | np.ndarray:
    """Mean-variance utility, also the certainty-equivalent return."""
    return np.asarray(expected) - 0.5 * aversion * np.square(volatility)


def indifference_curve(level: float, volatility: np.ndarray, aversion: float) -> np.ndarray:
    """Expected return that gives the same utility at each volatility."""
    return level + 0.5 * aversion * np.square(volatility)


def optimal_risky_share(expected: float, volatility: float, risk_free: float, aversion: float,
                        borrow_rate: float | None = None, max_share: float = 1.0) -> float:
    """Best share of a risky fund, the rest in T-bills.

    Without leverage the share stops at `max_share`. Pass `borrow_rate` with a `max_share` above 1
    to allow borrowing: the share stays at 1 until the fund's return justifies the higher rate.
    """
    lending = (expected - risk_free) / (aversion * volatility ** 2)
    if lending <= 1 or borrow_rate is None:
        share = lending
    else:
        share = max(1.0, (expected - borrow_rate) / (aversion * volatility ** 2))
    return float(np.clip(share, 0.0, max_share))


def optimal_on_frontier(mean: pd.Series, cov: pd.DataFrame, aversion: float,
                        constraints: Constraints) -> pd.Series:
    """Best fully invested portfolio when T-bills are not available."""
    weights = cp.Variable(len(mean))
    risk = cp.quad_form(weights, cp.psd_wrap(cov.to_numpy()))
    objective = mean.to_numpy() @ weights - 0.5 * aversion * risk
    problem = cp.Problem(cp.Maximize(objective), constraints.build(weights, mean.index))
    problem.solve(solver=cp.CLARABEL)
    if weights.value is None:
        raise RuntimeError(f"Optimizer failed with status {problem.status}")
    return pd.Series(weights.value, index=mean.index)


def investor_portfolio(mean: pd.Series, cov: pd.DataFrame, risk_free: float, aversion: float,
                       constraints: Constraints) -> pd.Series:
    """Best stock weights for an investor. Any shortfall from 1 is held in T-bills.

    Below a share of 1 the investor mixes T-bills with the tangency portfolio. With no borrowing,
    an investor wanting more risk than that moves up the frontier instead.
    """
    fund = solve_max_sharpe(mean, cov, risk_free, constraints)
    expected, volatility = portfolio_stats(fund, mean, cov)
    share = optimal_risky_share(expected, volatility, risk_free, aversion)
    return share * fund if share < 1 else optimal_on_frontier(mean, cov, aversion, constraints)


def aversion_from_score(score: float) -> float:
    """Map a questionnaire score to risk aversion. 0 is the most cautious, 1 the least.

    Spacing is geometric, because each step should change the risky share by a similar ratio.
    """
    if not 0 <= score <= 1:
        raise ValueError("score must be between 0 and 1")
    return float(HIGHEST_AVERSION * (LOWEST_AVERSION / HIGHEST_AVERSION) ** score)


def loss_limited_share(tolerated_loss: float, stress_loss: float) -> float:
    """Largest risky share whose loss in a stress event stays within what the investor accepts."""
    if stress_loss <= 0:
        raise ValueError("stress_loss must be positive")
    return float(min(1.0, tolerated_loss / stress_loss))
