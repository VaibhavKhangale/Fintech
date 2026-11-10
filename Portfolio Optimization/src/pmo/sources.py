"""Download raw data from external sources. Parsing is separate so it can be tested offline."""
import io
import re
import time
from datetime import date

import pandas as pd
import requests
import yfinance as yf

from pmo.config import Config, yahoo_ticker

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 60
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _get(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def fetch_prices(symbols: list[str], start: str) -> pd.DataFrame:
    """Daily dividend- and split-adjusted closes, one column per NSE symbol."""
    by_ticker = {yahoo_ticker(s): s for s in symbols}
    data = yf.download(list(by_ticker), start=start, interval="1d", auto_adjust=False,
                       progress=False, threads=True)
    prices = data["Adj Close"].rename(columns=by_ticker)
    missing = sorted(set(symbols) - set(prices.dropna(axis=1, how="all").columns))
    if missing:
        raise RuntimeError(f"No price data returned for: {missing}")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices.index.name = "date"
    prices.columns.name = None
    return prices[symbols]


def fetch_dividends(symbols: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily dividends per share and split-adjusted closes, for valuation."""
    by_ticker = {yahoo_ticker(s): s for s in symbols}
    data = yf.download(list(by_ticker), start=start, interval="1d", auto_adjust=False,
                       actions=True, progress=False, threads=True)
    tables = []
    for field in ("Dividends", "Close"):
        table = data[field].rename(columns=by_ticker)[symbols]
        table.index = pd.DatetimeIndex(table.index).tz_localize(None).normalize()
        table.index.name = "date"
        table.columns.name = None
        tables.append(table)
    return tables[0], tables[1]


def parse_mf_nav(payload: dict) -> pd.Series:
    """Daily net asset values from an mfapi.in response, oldest first."""
    rows = pd.DataFrame(payload["data"])
    dates = pd.to_datetime(rows["date"], dayfirst=True)
    nav = pd.Series(rows["nav"].astype(float).to_numpy(), index=dates, name="nav").sort_index()
    nav.index.name = "date"
    return nav


def fetch_mf_nav(config: Config, code: int) -> pd.Series:
    response = requests.get(config.sources["mf_nav"].format(code=code), headers=HEADERS,
                            timeout=TIMEOUT)
    response.raise_for_status()
    return parse_mf_nav(response.json())


def fetch_constituents(url: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_get(url)))


def latest_iima_file(page_html: str) -> str:
    """Relative path of the newest monthly survivorship-adjusted four-factor file."""
    files = re.findall(
        r"DATA/\d{4}-\d{2}_FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted\.csv",
        page_html)
    if not files:
        raise ValueError("IIMA page layout changed: monthly factor file not found")
    return max(files)


def parse_iima_factors(csv_text: str) -> pd.DataFrame:
    """Monthly factors as decimals, indexed by month end. MF is the market excess return."""
    df = pd.read_csv(io.StringIO(csv_text), na_values="NA")
    df.index = pd.PeriodIndex(df.pop("Date"), freq="M").to_timestamp("M").normalize()
    df.index.name = "date"
    return (df / 100).rename(columns={"MF": "MKT_RF"}).dropna()


def fetch_iima_factors(config: Config) -> tuple[pd.DataFrame, str]:
    page = config.sources["iima_page"]
    file = latest_iima_file(_get(page))
    return parse_iima_factors(_get(page + file)), file


def _clean_html(html: str) -> str:
    text = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text)


def parse_wss_tbill(html: str) -> pd.Series:
    """91-day T-bill primary yields (decimal) from one RBI Weekly Statistical Supplement page.

    The table has one year-ago column followed by five recent weeks. Only the recent weeks are
    used. Weeks with no auction are shown as '..' and skipped.
    """
    text = _clean_html(html)
    published = re.search(r"Date : (\w{3}) \d{1,2}, (\d{4})", text)
    header = re.search(r"Item/Week Ended (?:\d{4} ?)*((?:[A-Z][a-z]{2}\.? \d{1,2} ?)+)", text)
    row = re.search(r"91-Day Treasury Bill \(Primary\) Yield ((?:[\d.]+ ?|\.\. ?)+)", text)
    if not (published and header and row):
        raise ValueError("RBI WSS page layout changed: T-bill row not found")
    pub_month, pub_year = MONTHS[published.group(1)], int(published.group(2))
    labels = re.findall(r"([A-Z][a-z]{2})\.? (\d{1,2})", header.group(1))
    values = row.group(1).split()[:len(labels)]
    if len(labels) != len(values):
        raise ValueError("RBI WSS page layout changed: date and yield columns differ")
    out = {}
    for (month, day), value in list(zip(labels, values))[1:]:
        if value == "..":
            continue
        year = pub_year - 1 if MONTHS[month] > pub_month else pub_year
        out[pd.Timestamp(year, MONTHS[month], int(day))] = float(value) / 100
    return pd.Series(out, name="tbill_91d").sort_index()


def fetch_rbi_tbill(config: Config, since: date, pause: float = 0.5) -> pd.Series:
    """Weekly 91-day yields from `since`. Each page holds five weeks, so read every fourth."""
    ids = re.findall(r"WSSView\.aspx\?Id=(\d+)", _get(config.sources["rbi_wss_index"]))
    collected: list[pd.Series] = []
    for page_id in ids[::4]:
        series = parse_wss_tbill(_get(config.sources["rbi_wss_page"].format(id=page_id)))
        collected.append(series)
        time.sleep(pause)
        if not series.empty and series.index.min().date() < since:
            break
    merged = pd.concat(collected).sort_index()
    return merged[~merged.index.duplicated(keep="last")]
