"""Offline tests for NSE file handling."""
import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from pmo import nse

OLD = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
ABC,EQ,10,11,9,10.5,10.4,10,100,1000,01-JAN-2015,5,INE1,
ABC,N1,10,11,9,10.5,10.4,10,100,1000,01-JAN-2015,5,INE1,
"""
NEW = """BizDt,TckrSymb,SctySrs,ISIN,OpnPric,HghPric,LwPric,ClsPric,PrvsClsgPric,TtlTradgVol,TtlTrfVal
2024-10-03,XYZ,EQ,INE2,20,21,19,20.5,20,50,1000
2024-10-03,XYZ,BE,INE2,20,21,19,20.5,20,50,1000
2024-10-03,BOND,GB,INE3,20,21,19,20.5,20,50,1000
"""


def zipped(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("day.csv", text)
    return buffer.getvalue()


def test_urls_prefer_the_format_of_the_period():
    assert "cm01JAN2015bhav" in nse.bhav_urls(date(2015, 1, 1))[0]
    assert "20241003" in nse.bhav_urls(date(2024, 10, 3))[0]


def test_old_format_is_normalized_and_filtered():
    table = nse.parse_bhavcopy(zipped(OLD))
    assert list(table.columns) == nse.COLUMNS
    assert len(table) == 1
    assert table.loc[0, "date"] == pd.Timestamp("2015-01-01")
    assert table.loc[0, "close"] == 10.5


def test_new_format_matches_old_columns():
    table = nse.parse_bhavcopy(zipped(NEW))
    assert list(table.columns) == nse.COLUMNS
    assert sorted(table["series"]) == ["BE", "EQ"]
    assert table.loc[0, "date"] == pd.Timestamp("2024-10-03")


def test_month_ranges_cover_the_period():
    out = nse.month_ranges(date(2020, 1, 15), date(2020, 3, 2))
    assert out[0] == (date(2020, 1, 1), date(2020, 1, 31))
    assert out[-1] == (date(2020, 3, 1), date(2020, 3, 31))


@pytest.mark.parametrize("subject,expected", [
    ("Bonus 1:1", 0.5),
    ("Bonus 1 : 2", 2 / 3),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 0.1),
    ("Fv Splt Frm Rs 10 To Rs 2", 0.2),
    ("Bonus 1:1 And Face Value Split From Rs.10/- To Re.1/-", 0.05),
    ("Bonus Preference Shares 21:1", None),
    ("Face Value Split Rs.10/- To Re.1/- Per Share", 0.1),
    ("Scheme Of Arangement- Bonus - 1 Debenture For 1 Equity Share Held", None),
    ("Dividend - Rs 7 Per Share", None),
])
def test_price_factor(subject, expected):
    result = nse.price_factor(subject)
    assert result == (None if expected is None else pytest.approx(expected))


@pytest.mark.parametrize("subject,expected", [
    (" Annual General Meeting/Dividend - Rs 7 Per Share", 7.0),
    ("Interim Dividend - Re 0.50 Per Share", 0.5),
    ("Dividend - Rs 8 Per Share/Special Dividend - Rs 2 Per Share", 10.0),
    ("Interim Dividend 3 Per Share", 3.0),
    ("Annual General Meeting And Dividend Rs - 3.50 Per Share", 3.5),
    ("Annual General Meeting", 0.0),
    ("Bonus 1:1", 0.0),
])
def test_dividend_amount(subject, expected):
    assert nse.dividend_amount(subject) == pytest.approx(expected)


def test_integrated_rows_keep_only_original_filings():
    base = {"symbol": "A", "broadcast_Date": "14-May-2025 18:00:00", "qe_Date": "31-MAR-2025",
            "consolidated": "Standalone", "xbrl": "https://x/a.xml", "type_Sub": "Original"}
    revised = {**base, "type_Sub": "Revision"}
    table = nse.integrated_rows([base, revised])
    assert len(table) == 1
    assert table["consolidated"].iloc[0] == "Non-Consolidated"
    assert table["toDate"].iloc[0] == "31-Mar-2025"
