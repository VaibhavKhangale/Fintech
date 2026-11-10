"""Tests for the survivorship-free stock history."""
import numpy as np
import pandas as pd
import pytest

from pmo import universe as un

DAYS = pd.bdate_range("2020-01-01", periods=4)


def panel_of(symbol, closes, series="EQ", value=100.0, isin="INE1"):
    return pd.DataFrame({"date": DAYS[:len(closes)], "symbol": symbol, "series": series,
                         "close": closes, "value": value, "isin": isin})


def no_actions():
    return pd.DataFrame(columns=["symbol", "date", "factor", "dividend", "spin_off"]).astype(
        {"date": "datetime64[ns]", "symbol": "str", "factor": float, "dividend": float,
         "spin_off": bool})


def test_symbol_map_follows_chains():
    changes = pd.DataFrame({"old": ["A", "B"], "new": ["B", "C"],
                            "effective": pd.to_datetime(["2015-01-01", "2020-01-01"])})
    assert un.symbol_map(changes) == {"A": "C", "B": "C"}


def test_apply_symbols_joins_history_and_keeps_busiest_line():
    panel = pd.concat([panel_of("OLD", [10.0, 11.0]),
                       panel_of("NEW", [10.5], series="BE", value=1.0)])
    out = un.apply_symbols(panel, {"OLD": "NEW"})
    assert out["symbol"].unique().tolist() == ["NEW"]
    assert out.loc[out["date"] == DAYS[0], "series"].item() == "EQ"


def test_action_table_parses_and_combines_same_day_events():
    raw = pd.DataFrame({
        "symbol": ["A", "A", "B"], "exDate": ["02-Jan-2020", "02-Jan-2020", "-"],
        "subject": ["Bonus 1:1", "Dividend - Rs 2 Per Share", "Dividend - Rs 9 Per Share"]})
    out = un.action_table(raw, {})
    assert len(out) == 1
    assert out.loc[0, "factor"] == pytest.approx(0.5)
    assert out.loc[0, "dividend"] == pytest.approx(2)


def test_daily_returns_adjust_for_bonus_and_dividend():
    panel = panel_of("A", [100.0, 50.0, 49.0])
    actions = pd.DataFrame({"symbol": ["A", "A"], "date": [DAYS[1], DAYS[2]], "factor": [0.5, 1.0],
                            "dividend": [0.0, 1.0], "spin_off": [False, False]})
    out = un.daily_returns(panel, actions)["A"]
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0)
    assert out.iloc[2] == pytest.approx(0)


def test_spin_off_day_takes_the_median_return():
    panel = pd.concat([panel_of("A", [100.0, 80.0]), panel_of("B", [100.0, 101.0]),
                       panel_of("C", [100.0, 103.0])])
    actions = pd.DataFrame({"symbol": ["A"], "date": [DAYS[1]], "factor": [1.0], "dividend": [0.0],
                            "spin_off": [True]})
    out = un.daily_returns(panel, actions)
    assert out.loc[DAYS[1], "A"] == pytest.approx(0.01)


def test_returns_span_gaps_in_trading():
    panel = pd.DataFrame({"date": [DAYS[0], DAYS[3]], "symbol": "A", "series": "EQ",
                          "close": [100.0, 90.0], "value": 1.0})
    filler = panel_of("B", [10.0, 10.0, 10.0, 10.0])
    out = un.daily_returns(pd.concat([panel, filler]), no_actions())
    assert out.loc[DAYS[3], "A"] == pytest.approx(-0.1)


def test_unexplained_jumps_flags_large_moves():
    returns = pd.DataFrame({"A": [0.01, -0.5], "B": [0.6, 0.0]}, index=DAYS[:2])
    out = un.unexplained_jumps(returns)
    assert out["symbol"].tolist() == ["A", "B"]


def test_action_corrections_add_a_missed_split():
    raw = pd.DataFrame({"symbol": ["A"], "exDate": ["02-Jan-2020"], "subject": ["Dividend - Rs 2"]})
    fixes = [{"symbol": "B", "date": "2020-01-03", "factor": 0.2}]
    out = un.action_table(raw, {}, fixes).set_index("symbol")
    assert out.loc["B", "factor"] == pytest.approx(0.2)
    assert out.loc["A", "dividend"] == pytest.approx(2)


def test_rights_issue_counts_as_a_neutral_day():
    raw = pd.DataFrame({"symbol": ["A"], "exDate": ["02-Jan-2020"], "subject": ["Rights 7:6 @ Premium Rs 18"]})
    assert bool(un.action_table(raw, {}).loc[0, "spin_off"])


def test_liquidity_leaves_out_etfs_and_penny_stocks():
    panel = pd.concat([panel_of("ETF", [100.0] * 4, isin="INF1"), panel_of("PENNY", [5.0] * 4),
                       panel_of("REAL", [100.0] * 4)])
    assert list(un.liquidity(panel, window=2).columns) == ["REAL"]


def test_liquidity_counts_quiet_days_as_zero():
    panel = pd.concat([panel_of("A", [50.0, 50.0, 50.0, 50.0], value=10.0),
                       panel_of("B", [50.0], value=10.0)])
    out = un.liquidity(panel, window=3)
    assert out.loc[DAYS[3], "A"] == 10
    assert out.loc[DAYS[3], "B"] == 0


def test_liquidity_ignores_other_series():
    panel = panel_of("A", [50.0, 50.0, 50.0, 50.0], series="BE")
    panel = pd.concat([panel, panel_of("Z", [50.0, 50.0, 50.0, 50.0])])
    assert "A" not in un.liquidity(panel, window=2).columns


def test_universe_uses_only_data_up_to_each_month_end():
    index = pd.bdate_range("2020-01-01", "2020-03-31")
    value = pd.DataFrame({"A": 1.0, "B": 2.0}, index=index)
    value.loc["2020-03-01":, "A"] = 5.0
    out = un.month_end_universe(value, size=1)
    first = out[out["month"] == pd.Timestamp("2020-01-31")]
    assert first["symbol"].tolist() == ["B"]
    assert out[out["month"] == pd.Timestamp("2020-03-31")]["symbol"].tolist() == ["A"]


def test_monthly_returns_compound_and_mark_absence():
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"])
    daily = pd.DataFrame({"A": [0.1, 0.1, 0.0], "B": [0.0, np.nan, np.nan]}, index=index)
    out = un.monthly_total_returns(daily)
    assert out.loc["2020-01-31", "A"] == pytest.approx(0.21)
    assert np.isnan(out.loc["2020-02-29", "B"])


def test_delisting_loss_hits_only_the_last_traded_month():
    index = pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"])
    daily = pd.DataFrame({"A": [0.1, 0.0, np.nan], "B": [0.0, 0.0, 0.0]}, index=index)
    out = un.monthly_total_returns(daily, final_loss=0.3)
    assert out.loc["2020-02-29", "A"] == pytest.approx(-0.3)
    assert out.loc["2020-03-31", "B"] == 0


def test_overlap():
    assert un.overlap({"A", "B"}, {"B", "C"}) == pytest.approx(0.5)
    assert np.isnan(un.overlap({"A"}, set()))


def filings_table(rows):
    table = pd.DataFrame(rows, columns=["symbol", "published", "shares"])
    return table.assign(quarter_end=table["published"] - pd.Timedelta(days=40))


def test_market_cap_waits_for_publication():
    closes = pd.DataFrame({"date": pd.to_datetime(["2020-01-31", "2020-03-31"]), "symbol": "A",
                           "close": [100.0, 100.0]})
    filings = filings_table([("A", pd.Timestamp("2020-02-10 18:00"), 50.0)])
    out = un.market_caps(closes, filings, no_actions()).set_index("date")
    assert np.isnan(out.loc["2020-01-31", "market_cap"])
    assert out.loc["2020-03-31", "market_cap"] == pytest.approx(5000)


def test_market_cap_scales_shares_after_a_split():
    closes = pd.DataFrame({"date": pd.to_datetime(["2020-03-31"]), "symbol": "A", "close": [50.0]})
    filings = filings_table([("A", pd.Timestamp("2020-02-10"), 50.0)])
    actions = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2020-03-02")], "factor": [0.5],
                            "dividend": [0.0], "spin_off": [False]})
    out = un.market_caps(closes, filings, actions)
    assert out["market_cap"].iloc[0] == pytest.approx(5000)


def test_market_cap_ignores_a_split_before_the_filing():
    closes = pd.DataFrame({"date": pd.to_datetime(["2020-03-31"]), "symbol": "A", "close": [50.0]})
    filings = filings_table([("A", pd.Timestamp("2020-02-10"), 100.0)])
    actions = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2019-12-02")], "factor": [0.5],
                            "dividend": [0.0], "spin_off": [False]})
    out = un.market_caps(closes, filings, actions)
    assert out["market_cap"].iloc[0] == pytest.approx(5000)


def test_largest_stocks_needs_liquidity():
    day = pd.Timestamp("2020-01-31")
    caps = pd.DataFrame({"date": day, "symbol": ["BIG", "MID", "SMALL"], "market_cap": [9.0, 5.0, 1.0]})
    liquid = pd.DataFrame({"date": day, "symbol": ["MID", "SMALL"], "rank": [1, 2]})
    out = un.largest_stocks(caps, liquid, size=1)
    assert out["symbol"].tolist() == ["MID"]


def test_market_cap_keeps_each_split_with_its_own_stock():
    days = pd.to_datetime(["2020-03-31"] * 3)
    closes = pd.DataFrame({"date": days, "symbol": ["C", "A", "B"], "close": [10.0, 10.0, 10.0]})
    filings = filings_table([(s, pd.Timestamp("2020-01-10"), 100.0) for s in "ABC"])
    actions = pd.DataFrame({"symbol": ["B"], "date": [pd.Timestamp("2020-02-03")], "factor": [0.05],
                            "dividend": [0.0], "spin_off": [False]})
    out = un.market_caps(closes, filings, actions).set_index("symbol")["market_cap"]
    assert out["A"] == pytest.approx(1000)
    assert out["B"] == pytest.approx(20000)
    assert out["C"] == pytest.approx(1000)


def split_on(day, factor=0.1, spin_off=False):
    return pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp(day)], "factor": [factor],
                         "dividend": [0.0], "spin_off": [spin_off]})


def month_close(day, close=1.0):
    return pd.DataFrame({"date": [pd.Timestamp(day)], "symbol": ["A"], "close": [close]})


def test_filing_published_after_a_split_still_counts_shares_at_quarter_end():
    filings = pd.DataFrame({"symbol": ["A"], "published": [pd.Timestamp("2020-05-10")],
                            "quarter_end": [pd.Timestamp("2020-03-31")], "shares": [100.0]})
    out = un.market_caps(month_close("2020-05-29"), filings, split_on("2020-04-20"))
    assert out["market_cap"].iloc[0] == pytest.approx(1000)


def test_a_typing_error_in_shares_is_held_back():
    filings = filings_table([("A", pd.Timestamp("2020-02-10"), 100.0),
                             ("A", pd.Timestamp("2020-05-10"), 100000.0),
                             ("A", pd.Timestamp("2020-08-10"), 102.0)])
    days = pd.concat([month_close("2020-06-30"), month_close("2020-09-30")])
    out = un.market_caps(days, filings, no_actions()).set_index("date")["market_cap"]
    assert out.loc["2020-06-30"] == pytest.approx(100)
    assert out.loc["2020-09-30"] == pytest.approx(102)


def test_a_real_jump_is_accepted_once_a_second_filing_agrees():
    filings = filings_table([("A", pd.Timestamp("2020-02-10"), 100.0),
                             ("A", pd.Timestamp("2020-05-10"), 500.0),
                             ("A", pd.Timestamp("2020-08-10"), 505.0)])
    days = pd.concat([month_close("2020-06-30"), month_close("2020-09-30")])
    out = un.market_caps(days, filings, no_actions()).set_index("date")["market_cap"]
    assert out.loc["2020-06-30"] == pytest.approx(100)
    assert out.loc["2020-09-30"] == pytest.approx(505)


def test_spin_off_days_do_not_change_the_share_count():
    filings = filings_table([("A", pd.Timestamp("2020-02-10"), 100.0)])
    out = un.market_caps(month_close("2020-05-29"), filings, split_on("2020-04-20", 0.2, True))
    assert out["market_cap"].iloc[0] == pytest.approx(100)
