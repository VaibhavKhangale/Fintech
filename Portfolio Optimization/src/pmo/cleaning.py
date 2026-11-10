"""Corrections for known defects in vendor prices."""
import pandas as pd


def drop_bad_ticks(series: pd.Series, reference: pd.Series, tolerance: float = 0.25) -> pd.Series:
    """Blank prices that break from a closely related series, then revert.

    The check compares the price/reference ratio to its local median, so a lasting change in
    level is kept and only a brief misprint is removed.
    """
    ratio = series / reference.reindex(series.index)
    local = ratio.rolling(21, center=True, min_periods=5).median()
    return series.mask((ratio / local - 1).abs() > tolerance)


def neutralize_event(prices: pd.Series, market: pd.Series, ex_date: str) -> pd.Series:
    """Back-adjust a stock so its ex-date return equals the market's.

    Used for demergers the vendor did not adjust. The value of the spun-off business is not
    separately traded on the ex-date, so the market return is the neutral assumption.
    """
    ex = pd.Timestamp(ex_date)
    if ex not in prices.index:
        raise ValueError(f"{ex_date} is not a trading day in the price series")
    earlier = prices.index[prices.index < ex]
    previous = earlier[-1]
    stock_move = prices[ex] / prices[previous]
    market_move = market[ex] / market.loc[:previous].dropna().iloc[-1]
    adjusted = prices.copy()
    adjusted.loc[earlier] = prices.loc[earlier] * (stock_move / market_move)
    return adjusted


def remove_unit_change(nav: pd.Series, limit: float = 0.5) -> pd.Series:
    """Rescale earlier values across a one-day jump too large to be a return.

    Fund NAVs jump when a scheme changes its unit size, and holders' wealth doesn't change.
    """
    adjusted = nav.copy()
    moves = adjusted.pct_change()
    for date in moves.index[moves.abs() > limit]:
        adjusted.loc[adjusted.index < date] *= 1 + moves[date]
    return adjusted
