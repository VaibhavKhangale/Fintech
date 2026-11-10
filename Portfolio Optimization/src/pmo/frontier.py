"""The minimum variance frontier, in closed form and by numerical optimization.

Inputs here are annual: `mean` holds expected total returns and `cov` the annual covariance.
"""
from collections.abc import Callable
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def annualize(monthly_excess_mean: pd.Series, monthly_cov: pd.DataFrame,
              risk_free: float) -> tuple[pd.Series, pd.DataFrame]:
    """Convert monthly excess inputs to annual total-return inputs."""
    return monthly_excess_mean * MONTHS_PER_YEAR + risk_free, monthly_cov * MONTHS_PER_YEAR


def portfolio_stats(weights: pd.Series, mean: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    """Expected return and volatility of a portfolio."""
    w = weights[mean.index].to_numpy()
    return float(w @ mean.to_numpy()), float(np.sqrt(w @ cov.to_numpy() @ w))


def exposure(weights: pd.Series) -> dict[str, float]:
    """How much of a portfolio is short, and its total long plus short positions."""
    return {"short": float(-weights.clip(upper=0).sum()), "gross": float(weights.abs().sum()),
            "largest_long": float(weights.max()), "largest_short": float(weights.min())}


@dataclass(frozen=True)
class Frontier:
    """Closed-form frontier with short sales allowed and no other limits."""
    mean: pd.Series
    cov: pd.DataFrame

    def __post_init__(self) -> None:
        one = np.ones(len(self.mean))
        mu = self.mean.to_numpy()
        inv_one = np.linalg.solve(self.cov.to_numpy(), one)
        inv_mu = np.linalg.solve(self.cov.to_numpy(), mu)
        a, b, c = one @ inv_one, one @ inv_mu, mu @ inv_mu
        d = a * c - b ** 2
        if d <= 0:
            raise ValueError("Frontier needs at least two assets with different means")
        solved = {"a": a, "b": b, "c": c, "d": d, "inv_one": inv_one, "inv_mu": inv_mu}
        for name, value in solved.items():
            object.__setattr__(self, f"_{name}", value)

    @property
    def gmv_return(self) -> float:
        return self._b / self._a

    @property
    def gmv_volatility(self) -> float:
        return float(np.sqrt(1 / self._a))

    def gmv_weights(self) -> pd.Series:
        return pd.Series(self._inv_one / self._a, index=self.mean.index)

    def weights(self, target: float) -> pd.Series:
        """Lowest-variance fully invested portfolio with the target expected return."""
        lam = (self._c - self._b * target) / self._d
        gam = (self._a * target - self._b) / self._d
        return pd.Series(lam * self._inv_one + gam * self._inv_mu, index=self.mean.index)

    def volatility(self, target: float | np.ndarray) -> float | np.ndarray:
        """Frontier volatility: the hyperbola sigma^2 = (A mu^2 - 2 B mu + C) / D."""
        return np.sqrt((self._a * np.square(target) - 2 * self._b * np.asarray(target) + self._c)
                       / self._d)

    def curve(self, low: float, high: float, points: int = 121) -> pd.DataFrame:
        targets = np.linspace(low, high, points)
        return pd.DataFrame({"expected_return": targets, "volatility": self.volatility(targets)})

    def tangency_weights(self, risk_free: float) -> pd.Series:
        """Highest-Sharpe portfolio. It exists only if the GMV return exceeds the risk-free rate."""
        if risk_free >= self.gmv_return:
            raise ValueError("No tangency portfolio: risk-free rate is at or above the GMV return")
        raw = self._inv_mu - risk_free * self._inv_one
        return pd.Series(raw / raw.sum(), index=self.mean.index)


@dataclass(frozen=True)
class Constraints:
    """Limits on portfolio weights. `lower=None` allows short sales."""
    lower: float | None = 0.0
    upper: float | None = None
    sector_cap: float | None = None
    sectors: pd.Series | None = None

    def build(self, weights: cp.Variable, names: pd.Index, scale=1) -> list:
        """Linear rules on weights. `scale` is the sum of `weights`, 1 unless rescaled."""
        rules = [cp.sum(weights) == scale]
        if self.lower is not None:
            rules.append(weights >= self.lower * scale)
        if self.upper is not None:
            rules.append(weights <= self.upper * scale)
        if self.sector_cap is not None:
            labels = self.sectors[names].to_numpy()
            for sector in np.unique(labels):
                members = np.flatnonzero(labels == sector)
                rules.append(cp.sum(weights[members]) <= self.sector_cap * scale)
        return rules


SHORTS_ALLOWED = Constraints(lower=None)


def _solve(problem: cp.Problem, weights: cp.Variable, names: pd.Index) -> pd.Series:
    problem.solve(solver=cp.CLARABEL)
    if weights.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Optimizer failed with status {problem.status}")
    return pd.Series(weights.value, index=names)


def solve_min_variance(mean: pd.Series, cov: pd.DataFrame, target: float | None = None,
                       constraints: Constraints = SHORTS_ALLOWED) -> pd.Series:
    """Minimize variance, at a target return if given, otherwise the lowest-risk portfolio."""
    weights = cp.Variable(len(mean))
    rules = constraints.build(weights, mean.index)
    if target is not None:
        rules.append(mean.to_numpy() @ weights == target)
    risk = cp.quad_form(weights, cp.psd_wrap(cov.to_numpy()))
    return _solve(cp.Problem(cp.Minimize(risk), rules), weights, mean.index)


def sharpe_ratio(weights: pd.Series, mean: pd.Series, cov: pd.DataFrame,
                 risk_free: float) -> float:
    expected, volatility = portfolio_stats(weights, mean, cov)
    return (expected - risk_free) / volatility


def solve_max_sharpe(mean: pd.Series, cov: pd.DataFrame, risk_free: float,
                     constraints: Constraints = SHORTS_ALLOWED) -> pd.Series:
    """Highest-Sharpe portfolio under limits.

    Sharpe is a ratio, so it is rescaled into a convex problem: find y with unit excess return and
    the lowest variance, then w = y / sum(y). Limits are multiplied by sum(y) to keep them linear.
    With shorts allowed and the risk-free rate at or above the GMV return, the Sharpe ratio has no
    maximum, so this raises ValueError instead of returning exploding weights.
    """
    if constraints.lower is None:
        lowest = solve_min_variance(mean, cov, None, constraints)
        if risk_free >= float(lowest @ mean):
            raise ValueError("No tangency portfolio: risk-free rate is at or above the GMV return")
    scaled = cp.Variable(len(mean))
    total = cp.Variable(nonneg=True)
    rules = constraints.build(scaled, mean.index, scale=total)
    rules.append((mean.to_numpy() - risk_free) @ scaled == 1)
    risk = cp.quad_form(scaled, cp.psd_wrap(cov.to_numpy()))
    solved = _solve(cp.Problem(cp.Minimize(risk), rules), scaled, mean.index)
    return solved / solved.sum()


def capital_market_line(risk_free: float, sharpe: float,
                        volatility: float | np.ndarray) -> float | np.ndarray:
    """Expected return on a line from the risk-free rate with slope equal to the Sharpe ratio."""
    return risk_free + sharpe * np.asarray(volatility)


def tracking_portfolio(target: pd.Series, assets: pd.DataFrame,
                       constraints: Constraints | None = None) -> pd.Series:
    """Long-only weights in `assets` that follow `target` most closely, unless told otherwise."""
    weights = cp.Variable(assets.shape[1])
    gap = assets.to_numpy() @ weights - target.to_numpy()
    rules = (constraints or Constraints()).build(weights, assets.columns)
    return _solve(cp.Problem(cp.Minimize(cp.sum_squares(gap)), rules), weights, assets.columns)


def highest_return(mean: pd.Series, constraints: Constraints) -> float:
    """Top of the frontier. With shorts allowed it has no limit, so use the best single asset."""
    if constraints.lower is None:
        return float(mean.max())
    weights = cp.Variable(len(mean))
    problem = cp.Problem(cp.Maximize(mean.to_numpy() @ weights),
                         constraints.build(weights, mean.index))
    return float(_solve(problem, weights, mean.index) @ mean)


def frontier_point(mean: pd.Series, cov: pd.DataFrame, constraints: Constraints,
                   position: float) -> pd.Series:
    """Efficient portfolio at a position along the frontier: 0 is lowest risk, 1 the highest."""
    lowest = solve_min_variance(mean, cov, None, constraints)
    if position == 0:
        return lowest
    start = float(lowest @ mean)
    top = highest_return(mean, constraints)
    return solve_min_variance(mean, cov, start + position * (top - start), constraints)


def efficient_frontier(mean: pd.Series, cov: pd.DataFrame, constraints: Constraints,
                       points: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace the efficient part of the frontier. Returns a summary and the weights per point."""
    positions = np.linspace(0, 0.999, points)
    weights = pd.DataFrame([frontier_point(mean, cov, constraints, p) for p in positions],
                           index=positions)
    stats = [portfolio_stats(row, mean, cov) for _, row in weights.iterrows()]
    summary = pd.DataFrame(stats, columns=["expected_return", "volatility"], index=positions)
    return summary, weights


def effective_stocks(weights: pd.Series) -> float:
    """Number of equal positions with the same concentration: the inverse of summed squares."""
    return float(1 / (weights ** 2).sum())


InputFn = Callable[[pd.DataFrame], tuple[pd.Series, pd.DataFrame]]


def bootstrap_weights(excess: pd.DataFrame, inputs: InputFn, risk_free: float,
                      target: float | None = None, draws: int = 500, seed: int = 11,
                      builder: Callable[[pd.Series, pd.DataFrame], pd.Series] | None = None,
                      ) -> pd.DataFrame:
    """Portfolio weights re-estimated on resampled months, one row per draw.

    With `target` the portfolio is that frontier point; without it, the global minimum variance
    portfolio. A `builder` overrides both, for constrained portfolios. The spread across draws
    shows how much sampling error moves the weights.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        picks = rng.integers(0, len(excess), len(excess))
        sample = excess.iloc[picks].set_axis(excess.index)
        mean, cov = annualize(*inputs(sample), risk_free)
        if builder is not None:
            rows.append(builder(mean, cov))
            continue
        frontier = Frontier(mean, cov)
        rows.append(frontier.gmv_weights() if target is None else frontier.weights(target))
    return pd.DataFrame(rows).reset_index(drop=True)


def promised_vs_realized(train: pd.DataFrame, test: pd.DataFrame, inputs: InputFn,
                         risk_free: float, targets: list[float]) -> pd.DataFrame:
    """Compare what frontier portfolios promised on `train` with their results on `test`.

    Returns stay in excess of the monthly risk-free rate on `test`, so `risk_free` is added back
    to put realized and promised on the same footing.
    """
    mean, cov = annualize(*inputs(train), risk_free)
    frontier = Frontier(mean, cov)
    rows = []
    for target in targets:
        weights = frontier.weights(target)
        held = test[weights.index] @ weights
        rows.append({
            "target_return": target,
            "promised_volatility": float(frontier.volatility(target)),
            "realized_return": float(held.mean() * MONTHS_PER_YEAR + risk_free),
            "realized_volatility": float(held.std() * np.sqrt(MONTHS_PER_YEAR)),
        })
    return pd.DataFrame(rows)
