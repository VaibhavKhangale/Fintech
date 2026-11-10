"""Offline tests for parsing, cleaning, return and quality logic."""
import numpy as np
import pandas as pd
import pytest

from pmo import cleaning, quality, returns, sources

WSS_PAGE = """
<html><script>var x = 1;</script><body>
Date : Jan 02, 2026 Ratios and Rates
<td>2024</td><td>2025</td> Item/Week Ended Dec. 27 Nov. 28 Dec. 05 Dec. 12 Dec. 19 Dec. 26
1 2 3 4 5 6
91-Day Treasury Bill (Primary) Yield 6.55 5.36 .. 5.27 5.28 5.26
182-Day Treasury Bill (Primary) Yield 6.70 5.50 5.50 5.50 5.50 5.50
</body></html>
"""
WSS_YEAR_FIRST = WSS_PAGE.replace(
    "<td>2024</td><td>2025</td> Item/Week Ended", "Item/Week Ended 2024 2025")


def month_ends(n, start="2020-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


@pytest.mark.parametrize("page", [WSS_PAGE, WSS_YEAR_FIRST])
def test_wss_parser_handles_year_position_and_missing_weeks(page):
    series = sources.parse_wss_tbill(page)
    assert list(series.index) == [pd.Timestamp("2025-11-28"), pd.Timestamp("2025-12-12"),
                                  pd.Timestamp("2025-12-19"), pd.Timestamp("2025-12-26")]
    assert series.iloc[0] == pytest.approx(0.0536)


def test_wss_parser_rejects_changed_layout():
    with pytest.raises(ValueError, match="layout changed"):
        sources.parse_wss_tbill("<html>nothing here</html>")


def test_iima_parser_scales_and_drops_incomplete_rows():
    csv = "Date,SMB,HML,WML,MF,RF\n1993-10,3,2,1,NA,NA\n1993-11,1,2,3,4,0.5\n"
    frame = sources.parse_iima_factors(csv)
    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("1993-11-30")
    assert frame.loc[frame.index[0], "MKT_RF"] == pytest.approx(0.04)
    assert frame.loc[frame.index[0], "RF"] == pytest.approx(0.005)


def test_latest_iima_file_picks_newest_monthly_file():
    stem = "_FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted.csv"
    page = f'<a href="DATA/2024-12{stem}"></a><a href="DATA/2025-12{stem}"></a>'
    assert sources.latest_iima_file(page) == f"DATA/2025-12{stem}"


def test_complete_month_prices_drops_partial_month():
    index = pd.date_range("2024-01-01", "2024-03-15", freq="B")
    daily = pd.DataFrame({"A": range(len(index))}, index=index, dtype=float)
    monthly = returns.complete_month_prices(daily, index[-1])
    assert list(monthly.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]


def test_complete_month_prices_keeps_month_ended_on_as_of():
    index = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    daily = pd.DataFrame({"A": 1.0}, index=index)
    assert returns.complete_month_prices(daily, index[-1]).index[-1] == pd.Timestamp("2024-03-31")


def test_yield_conversion_compounds_over_calendar_days():
    yields = pd.Series([0.06, 0.06], index=pd.to_datetime(["2023-02-03", "2023-02-17"]))
    result = returns.yield_to_monthly_return(yields)
    assert result.iloc[0] == pytest.approx(1.06 ** (28 / 365) - 1)


def test_splice_prefers_iima_and_labels_source():
    iima = pd.Series([0.004, 0.005], index=month_ends(2))
    rbi = pd.Series([0.9, 0.9, 0.006], index=month_ends(3))
    out = returns.splice_risk_free(iima, rbi)
    assert out["rf"].tolist() == [0.004, 0.005, 0.006]
    assert out["source"].tolist() == ["IIMA", "IIMA", "RBI WSS"]


def test_split_periods_marks_latest_months_as_holdout():
    labels = returns.split_periods(month_ends(10), 3)
    assert labels.tolist() == ["estimation"] * 7 + ["holdout"] * 3
    with pytest.raises(ValueError):
        returns.split_periods(month_ends(3), 3)


def test_drop_bad_ticks_removes_misprint_but_keeps_real_moves():
    index = pd.bdate_range("2024-01-01", periods=40)
    reference = pd.Series(100 * 1.001 ** np.arange(40), index=index)
    series = reference * 0.01
    series.iloc[20:22] = series.iloc[20:22] / 10
    cleaned = cleaning.drop_bad_ticks(series, reference)
    assert cleaned.isna().sum() == 2
    assert cleaned.iloc[20:22].isna().all()


def test_drop_bad_ticks_keeps_lasting_level_change():
    index = pd.bdate_range("2024-01-01", periods=40)
    reference = pd.Series(100.0, index=index)
    series = pd.Series(1.0, index=index)
    series.iloc[20:] = 2.0
    assert cleaning.drop_bad_ticks(series, reference).iloc[:18].notna().all()


def test_remove_unit_change_keeps_returns_on_other_days():
    index = pd.bdate_range("2024-01-01", periods=4)
    nav = pd.Series([10.0, 10.1, 1010.0, 1020.1], index=index)
    adjusted = cleaning.remove_unit_change(nav)
    assert adjusted.iloc[2] / adjusted.iloc[1] == pytest.approx(1.0)
    assert adjusted.iloc[1] / adjusted.iloc[0] == pytest.approx(1.01)
    assert adjusted.iloc[3] == pytest.approx(1020.1)


def test_neutralize_event_sets_ex_date_return_to_market():
    index = pd.bdate_range("2024-01-01", periods=5)
    stock = pd.Series([100.0, 101.0, 95.0, 96.0, 97.0], index=index)
    market = pd.Series([200.0, 202.0, 204.02, 205.0, 206.0], index=index)
    adjusted = cleaning.neutralize_event(stock, market, str(index[2].date()))
    assert adjusted.iloc[2] / adjusted.iloc[1] == pytest.approx(1.01)
    assert adjusted.iloc[2:].equals(stock.iloc[2:])
    assert adjusted.iloc[1] / adjusted.iloc[0] == pytest.approx(1.01)


def test_neutralize_event_rejects_non_trading_day():
    index = pd.bdate_range("2024-01-01", periods=3)
    series = pd.Series(1.0, index=index)
    with pytest.raises(ValueError):
        cleaning.neutralize_event(series, series, "2024-06-01")


def test_market_adjusted_jumps_ignores_market_wide_moves():
    index = pd.bdate_range("2024-01-01", periods=4)
    market = pd.Series([100.0, 80.0, 80.0, 80.0], index=index)
    prices = pd.DataFrame({"A": [100.0, 80.0, 80.0, 120.0]}, index=index)
    jumps = quality.market_adjusted_jumps(prices, market, 0.15)
    assert len(jumps) == 1
    assert jumps.iloc[0]["date"] == index[3]


def test_missing_share_ignores_days_before_listing():
    index = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": [np.nan] * 5 + [1.0] * 5}, index=index)
    assert quality.missing_share(prices, 0.0).empty


def test_monthly_gaps_finds_missing_month():
    frame = pd.DataFrame({"A": 1.0}, index=month_ends(4).delete(2))
    assert quality.monthly_gaps(frame)["missing_month"].iloc[0] == month_ends(4)[2]


def test_rf_out_of_range_flags_percent_scale_mistake():
    rf = pd.Series([0.005, 0.5], index=month_ends(2))
    assert len(quality.rf_out_of_range(rf)) == 1
