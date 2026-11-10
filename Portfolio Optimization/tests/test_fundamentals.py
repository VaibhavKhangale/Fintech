"""Offline tests for parsing filings."""
import pandas as pd
import pytest

from pmo import fundamentals as fu

HTML = """<html><script>var x=1;</script><table>
<tr><td>Description</td><td>Amount(Rs. in lakhs)</td></tr>
<tr><td>(a) Net sales/income from operations (Net of excise duty)</td><td>1219200.00</td></tr>
<tr><td>Net Profit(+)/ Loss(-) for the period</td><td>(350.50)</td></tr>
<tr><td>Face Value (in Rs.)</td><td>5.00</td></tr>
<tr><td>Paid-up Equity Share Capital</td><td>57400.00</td></tr>
<tr><td>Basic EPS before Extraordinary items (in Rs.)</td><td>26.35</td></tr>
</table></html>"""
BANK = """<td>Amount(Rs. in lakhs)</td><td>Interest Earned</td><td>870762.00</td>
<td>Net Profit (+) / Loss (-) for the period</td><td>185907.00</td>"""
XBRL = """<xbrl>
<in-bse-fin:RevenueFromOperations contextRef="OneD" unitRef="INR">206090000000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="TwoD" unitRef="INR">100.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:ProfitLossForPeriod contextRef="OneD" unitRef="INR">41100000000.00</in-bse-fin:ProfitLossForPeriod>
<in-bse-fin:FaceValueOfEquityShareCapital contextRef="OneI" unitRef="INR">5</in-bse-fin:FaceValueOfEquityShareCapital>
<in-bse-fin:PaidUpValueOfEquityShareCapital contextRef="OneI" unitRef="INR">21760000000.00</in-bse-fin:PaidUpValueOfEquityShareCapital>
</xbrl>"""


def test_html_figures_are_converted_to_rupees():
    out = fu.parse_html(HTML)
    assert out["revenue"] == pytest.approx(1219200 * 1e5)
    assert out["net_profit"] == pytest.approx(-350.5 * 1e5)
    assert out["paid_up"] == pytest.approx(57400 * 1e5)
    assert out["face_value"] == 5
    assert out["eps"] == 26.35


def test_html_bank_layout():
    out = fu.parse_html(BANK)
    assert out["revenue"] == pytest.approx(870762 * 1e5)
    assert out["net_profit"] == pytest.approx(185907 * 1e5)


def test_xbrl_uses_the_current_period_context():
    out = fu.parse_xbrl(XBRL)
    assert out["revenue"] == pytest.approx(206090000000)
    assert out["face_value"] == 5
    assert out["paid_up"] == pytest.approx(21760000000)


def filing(symbol, broadcast, to_date, cumulative="Non-cumulative", period="Quarterly"):
    return {"symbol": symbol, "broadCastDate": broadcast, "toDate": to_date, "period": period,
            "cumulative": cumulative, "consolidated": "Consolidated"}


def test_first_filings_keep_the_earliest_publication():
    table = pd.DataFrame([
        filing("A", "20-Jan-2015 10:00:00", "31-Dec-2014"),
        filing("A", "05-Mar-2015 10:00:00", "31-Dec-2014"),
        filing("A", "20-Jan-2015 10:00:00", "31-Dec-2014", cumulative="Cumulative"),
    ])
    out = fu.first_filings(table)
    assert len(out) == 1
    assert out.iloc[0]["published"] == pd.Timestamp("2015-01-20 10:00:00")
    assert out.iloc[0]["days_to_publish"] == 20


def test_filing_url_prefers_xbrl_and_skips_placeholders():
    assert fu.filing_url(pd.Series({"xbrl": "https://x/a.xml", "resultDetailedDataLink": None})).endswith("a.xml")
    assert fu.filing_url(pd.Series({"xbrl": "https://x/-", "resultDetailedDataLink": "https://x/b.html"})).endswith("b.html")
    assert fu.filing_url(pd.Series({"xbrl": "https://x/-", "resultDetailedDataLink": None})) is None


def test_share_count_uses_paid_up_capital_when_consistent():
    table = pd.DataFrame({"paid_up": [1e9], "face_value": [10.0], "net_profit": [2e8], "eps": [2.1]})
    assert fu.share_count(table).iloc[0] == pytest.approx(1e8)


def test_share_count_recovers_from_zero_face_value():
    table = pd.DataFrame({"paid_up": [1e9], "face_value": [0.0], "net_profit": [2e8], "eps": [2.0]})
    assert fu.share_count(table).iloc[0] == pytest.approx(1e8)


def test_share_count_rejects_capital_typed_100_times_too_large():
    table = pd.DataFrame({"paid_up": [1e11], "face_value": [10.0], "net_profit": [2e8], "eps": [2.0]})
    assert fu.share_count(table).iloc[0] == pytest.approx(1e8)


def test_share_count_ignores_eps_that_is_mostly_rounding():
    table = pd.DataFrame({"paid_up": [1e9], "face_value": [10.0], "net_profit": [1e6], "eps": [0.01]})
    assert fu.share_count(table).iloc[0] == pytest.approx(1e8)


def test_parse_xbrl_reads_integrated_filing_tags():
    document = ('<in-capmkt:RevenueFromOperations contextRef="OneD" unitRef="INR">5000</in-capmkt:'
                'RevenueFromOperations><in-capmkt:ProfitLossForPeriod contextRef="OneD">700</in-capmkt:'
                'ProfitLossForPeriod>')
    figures = fu.parse_xbrl(document)
    assert figures["revenue"] == 5000
    assert figures["net_profit"] == 700
