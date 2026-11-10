"""Walk-forward portfolio testing: fit on the past, hold over the next period."""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pmo import stats

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class WalkForward:
    returns: pd.DataFrame            # monthly excess returns, one column per strategy
    weights: dict[str, pd.DataFrame]  # target weights, one row per rebalance date


def run(excess: pd.DataFrame, strategies: dict[str, Strategy], window: int = 60,
        step: int = 12) -> WalkForward:
    """Refit each strategy every `step` months on the prior `window` months.

    Holdings are reset to target weights monthly, so drift between rebalances is ignored.
    """
    months = excess.index
    paths: dict[str, list[pd.Series]] = {name: [] for name in strategies}
    held: dict[str, dict] = {name: {} for name in strategies}
    for start in range(window, len(months), step):
        train = excess.loc[months[start - window:start]]
        test = excess.loc[months[start:start + step]]
        for name, strategy in strategies.items():
            weights = strategy(train)
            paths[name].append(test[weights.index] @ weights)
            held[name][test.index[0]] = weights
    returns = pd.DataFrame({name: pd.concat(parts) for name, parts in paths.items()})
    weights = {name: pd.DataFrame.from_dict(rows, orient="index") for name, rows in held.items()}
    return WalkForward(returns, weights)


def turnover(weights: pd.DataFrame) -> float:
    """Average one-way trading per rebalance, as a share of the portfolio."""
    return float(weights.fillna(0.0).diff().abs().sum(axis=1).iloc[1:].mean() / 2)


@dataclass(frozen=True)
class Simulation:
    returns: pd.Series      # monthly return after costs and any tax
    costs: float            # total trading cost, in units of starting wealth
    taxes: float            # total tax paid, in the same units
    turnover: float         # average one-way trading per rebalance


def simulate(returns: pd.DataFrame, targets: dict[pd.Timestamp, pd.Series], costs: pd.Series,
             tax_rates: pd.Series | None = None) -> Simulation:
    """Follow one portfolio month by month, with holdings drifting between rebalances.

    `returns` has one column per asset, including any cash. Each date in `targets` is the first
    month of a holding period: weights are set just before that month's return, and trades pay
    `costs` per rupee traded. Without `tax_rates` no tax is charged. With them, gains realized at a
    rebalance are taxed at each asset's rate, on an average-cost basis, with no annual exemption.
    """
    assets = list(returns.columns)
    rates = (tax_rates if tax_rates is not None else pd.Series(0.0, index=assets))
    rate = rates.reindex(assets).fillna(0.0).to_numpy()
    cost_rate = costs.reindex(assets).fillna(0.0).to_numpy()
    values = returns.to_numpy()
    holdings, basis = np.zeros(len(assets)), np.zeros(len(assets))
    wealth = 1.0
    path, paid_costs, paid_tax, traded = [], 0.0, 0.0, []
    for index, month in enumerate(returns.index):
        start = wealth
        if month in targets:
            target = targets[month].reindex(assets).fillna(0.0).to_numpy()
            before = holdings.copy()
            sold = np.clip(before - target * wealth, 0, None)
            share_sold = np.divide(sold, before, out=np.zeros_like(sold), where=before > 0)
            gains = sold - basis * share_sold
            tax = float(max((gains * rate).sum(), 0.0))
            cost = float((np.abs(target * wealth - before) * cost_rate).sum())
            wealth -= tax + cost
            paid_tax += tax
            paid_costs += cost
            holdings = target * wealth
            bought = np.clip(holdings - before, 0, None)
            basis = basis * (1 - share_sold) + bought
            traded.append(float(np.abs(holdings - before).sum() / 2 / wealth))
        holdings = holdings * (1 + values[index])
        wealth = float(holdings.sum())
        path.append(wealth / start - 1)
    return Simulation(pd.Series(path, index=returns.index), paid_costs, paid_tax,
                      float(np.mean(traded)))


def scorecard(result: WalkForward) -> pd.DataFrame:
    """Annualized excess return, risk and trading for each strategy."""
    rows = {}
    for name in result.returns:
        series = result.returns[name]
        weights = result.weights.get(name)
        rows[name] = {
            "excess_return": series.mean() * stats.MONTHS_PER_YEAR,
            "volatility": series.std() * np.sqrt(stats.MONTHS_PER_YEAR),
            "sharpe": series.mean() / series.std() * np.sqrt(stats.MONTHS_PER_YEAR),
            "max_drawdown": stats.max_drawdown(series),
            "turnover": turnover(weights) if weights is not None else np.nan,
            "largest_position": weights.abs().max(axis=1).mean() if weights is not None else np.nan,
            "short_exposure": (-weights.clip(upper=0).sum(axis=1)).mean()
            if weights is not None else np.nan,
        }
    return pd.DataFrame(rows).T
