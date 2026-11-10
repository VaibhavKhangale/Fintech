"""Data quality checks. Each returns a table of problems; empty means the check passed."""
import pandas as pd


def missing_share(prices: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Stocks missing too many trading days after their first valid price."""
    share = {c: prices[c].loc[prices[c].first_valid_index():].isna().mean() for c in prices}
    out = pd.Series(share, name="missing_share")
    return out[out > limit].to_frame()


def late_starts(prices: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Stocks whose history begins more than a month after the study start."""
    first = prices.apply(pd.Series.first_valid_index).rename("first_date")
    return first[first > start + pd.Timedelta(days=31)].to_frame()


def market_adjusted_jumps(prices: pd.DataFrame, market: pd.Series,
                          threshold: float) -> pd.DataFrame:
    """Daily moves far from the market's, the usual sign of an unadjusted corporate action."""
    stock_ret = prices.pct_change(fill_method=None)
    market_ret = market.reindex(prices.index).pct_change(fill_method=None)
    rows = stock_ret.sub(market_ret, axis=0).stack().rename("excess_return").reset_index()
    rows.columns = ["date", "symbol", "excess_return"]
    return rows[rows["excess_return"].abs() > threshold].sort_values("excess_return")


def monthly_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing months in a month-end indexed table."""
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="ME")
    return pd.DataFrame({"missing_month": expected.difference(frame.index)})


def rf_out_of_range(rf: pd.Series, low: float = 0.0, high: float = 0.02) -> pd.DataFrame:
    """Monthly risk-free returns outside a plausible range for Indian T-bills."""
    return rf[(rf < low) | (rf > high)].to_frame("rf")
