"""Return calculations and the study's estimation/holdout split."""
import pandas as pd


def complete_month_prices(daily: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Last price of each month, dropping the month containing `as_of` unless it has ended."""
    monthly = daily.resample("ME").last()
    if as_of < monthly.index[-1]:
        monthly = monthly.iloc[:-1]
    monthly.index.name = "date"
    return monthly


def monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_prices.pct_change(fill_method=None).iloc[1:]


def yield_to_monthly_return(yields: pd.Series) -> pd.Series:
    """Convert annualized yields, averaged within each calendar month, to monthly returns."""
    monthly = yields.resample("ME").mean().dropna()
    return (1 + monthly) ** (monthly.index.days_in_month / 365) - 1


def splice_risk_free(iima_rf: pd.Series, rbi_rf: pd.Series) -> pd.DataFrame:
    """IIMA's series where it exists, RBI-derived returns after it ends."""
    extension = rbi_rf[rbi_rf.index > iima_rf.index.max()]
    out = pd.concat([
        pd.DataFrame({"rf": iima_rf, "source": "IIMA"}),
        pd.DataFrame({"rf": extension, "source": "RBI WSS"}),
    ])
    out.index.name = "date"
    return out


def select_period(frame: pd.DataFrame, periods: pd.Series, label: str) -> pd.DataFrame:
    """Rows of `frame` in the given period. Stages 1 to 9 use 'estimation' only."""
    return frame.loc[frame.index.intersection(periods.index[periods == label])]


def split_periods(index: pd.DatetimeIndex, holdout_months: int) -> pd.Series:
    """Label each month 'estimation' or 'holdout'; holdout is the most recent months."""
    if not 0 < holdout_months < len(index):
        raise ValueError("holdout_months must be positive and shorter than the sample")
    labels = pd.Series("estimation", index=index, name="period")
    labels.iloc[-holdout_months:] = "holdout"
    return labels
