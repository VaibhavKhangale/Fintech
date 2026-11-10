"""NSE public data: daily bhavcopies, corporate actions, result filings and symbol changes.

These are unofficial routes to public files, so raw responses are archived locally and requests
are paced. Parsing is separate from downloading so it can be tested offline.
"""
import io
import json
import re
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from pmo.config import RAW_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
OLD_URL = ("https://nsearchives.nseindia.com/content/historical/EQUITIES/"
           "{year}/{month}/cm{day}{month}{year}bhav.csv.zip")
NEW_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv.zip"
API = "https://www.nseindia.com/api/"
SYMBOL_CHANGES_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
BHAV_DIR = RAW_DIR / "bhavcopy"
ACTIONS_DIR = RAW_DIR / "nse_actions"
INTEGRATED_DIR = RAW_DIR / "nse_integrated"
RESULTS_DIR = RAW_DIR / "nse_results"
TRI_SITE = "https://www.niftyindices.com"
TRI_REFERER = TRI_SITE + "/reports/historical-data"
FORMAT_CHANGE = date(2024, 7, 8)
SERIES = ("EQ", "BE", "BZ")
COLUMNS = ["date", "symbol", "series", "isin", "open", "high", "low", "close", "prev_close",
           "volume", "value"]
OLD_NAMES = {"TIMESTAMP": "date", "SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin",
             "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
             "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "TOTTRDVAL": "value"}
NEW_NAMES = {"BizDt": "date", "TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin",
             "OpnPric": "open", "HghPric": "high", "LwPric": "low", "ClsPric": "close",
             "PrvsClsgPric": "prev_close", "TtlTradgVol": "volume", "TtlTrfVal": "value"}


INTEGRATED_PAGE = 20000


def bhav_urls(day: date) -> list[str]:
    """Candidate URLs for one day, the expected format first."""
    month = day.strftime("%b").upper()
    old = OLD_URL.format(year=day.year, month=month, day=f"{day.day:02d}")
    new = NEW_URL.format(stamp=day.strftime("%Y%m%d"))
    return [old, new] if day < FORMAT_CHANGE else [new, old]


def parse_bhavcopy(raw: bytes) -> pd.DataFrame:
    """Normalize a zipped bhavcopy of either format, keeping ordinary equity series."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        table = pd.read_csv(archive.open(archive.namelist()[0]))
    names = OLD_NAMES if "SYMBOL" in table.columns else NEW_NAMES
    table = table.rename(columns=names)
    if "isin" not in table.columns:
        table["isin"] = ""
    table["symbol"] = table["symbol"].astype(str).str.strip()
    table["series"] = table["series"].astype(str).str.strip()
    table["date"] = pd.to_datetime(table["date"], format="mixed", dayfirst=names is OLD_NAMES)
    return table.loc[table["series"].isin(SERIES), COLUMNS].reset_index(drop=True)


def _bhav_path(day: date, folder: Path) -> Path:
    return folder / str(day.year) / f"{day:%Y%m%d}.zip"


def download_bhavcopies(start: date, end: date, folder: Path = BHAV_DIR, pause: float = 0.25) -> int:
    """Fetch every day's file, skipping days already done. Resumable; returns files stored."""
    session = requests.Session()
    session.headers.update(HEADERS)
    stored, day = 0, start
    while day <= end:
        path = _bhav_path(day, folder)
        marker = path.with_suffix(".none")
        if not path.exists() and not marker.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = None
            for url in bhav_urls(day):
                response = session.get(url, timeout=60)
                time.sleep(pause)
                if response.status_code == 200 and response.content[:2] == b"PK":
                    payload = response.content
                    break
            if payload is None:
                marker.write_text("", encoding="utf-8")
            else:
                path.write_bytes(payload)
                stored += 1
        day += timedelta(days=1)
    return stored


def load_bhavcopies(folder: Path = BHAV_DIR) -> pd.DataFrame:
    """All stored days as one long table."""
    frames = [parse_bhavcopy(path.read_bytes()) for path in sorted(folder.glob("*/*.zip"))]
    return pd.concat(frames, ignore_index=True)


def api_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=60)
    return session


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months covering the period, as (first day, last day)."""
    firsts = pd.date_range(start.replace(day=1), end, freq="MS")
    return [(first.date(), (first + pd.offsets.MonthEnd(0)).date()) for first in firsts]


def download_monthly(endpoint: str, folder: Path, start: date, end: date,
                     pause: float = 1.0) -> int:
    """Archive an NSE API endpoint month by month. Resumable; returns months stored."""
    folder.mkdir(parents=True, exist_ok=True)
    session = api_session()
    stored = 0
    for first, last in month_ranges(start, end):
        path = folder / f"{first:%Y-%m}.json"
        if path.exists():
            continue
        url = f"{API}{endpoint}&from_date={first:%d-%m-%Y}&to_date={last:%d-%m-%Y}"
        response = session.get(url, timeout=120)
        if response.status_code != 200:
            session = api_session()
            time.sleep(5)
            response = session.get(url, timeout=120)
        response.raise_for_status()
        path.write_text(json.dumps(response.json()), encoding="utf-8")
        stored += 1
        time.sleep(pause)
    return stored


def load_monthly(folder: Path) -> pd.DataFrame:
    frames = [pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
              for path in sorted(folder.glob("*.json"))]
    return pd.concat(frames, ignore_index=True)


def fetch_symbol_changes() -> pd.DataFrame:
    response = requests.get(SYMBOL_CHANGES_URL, headers=HEADERS, timeout=60)
    response.raise_for_status()
    table = pd.read_csv(io.StringIO(response.text), encoding_errors="ignore", header=0,
                        names=["company", "old", "new", "effective"], on_bad_lines="skip")
    table["old"] = table["old"].str.strip()
    table["new"] = table["new"].str.strip()
    table["effective"] = pd.to_datetime(table["effective"], format="%d-%b-%Y", errors="coerce")
    return table.dropna(subset=["effective"])


def fetch_total_return_index(name: str, start: date, end: date, pause: float = 1.0) -> pd.Series:
    """Daily total return index from niftyindices.com, requested a year at a time."""
    session = requests.Session()
    session.headers.update({**HEADERS, "Referer": TRI_REFERER, "Origin": TRI_SITE,
                            "Content-Type": "application/json; charset=utf-8"})
    session.get(TRI_REFERER, timeout=60)
    rows: list[dict] = []
    for year in range(start.year, end.year + 1):
        first = max(start, date(year, 1, 1))
        last = min(end, date(year, 12, 31))
        query = (f"{{'name':'{name}','startDate':'{first:%d-%b-%Y}',"
                 f"'endDate':'{last:%d-%b-%Y}','indexName':'{name}'}}")
        response = session.post(TRI_SITE + "/BackPage/getTotalReturnIndexString",
                                json={"cinfo": query}, timeout=120)
        response.raise_for_status()
        rows.extend(response.json())
        time.sleep(pause)
    table = pd.DataFrame(rows)
    dates = pd.to_datetime(table["Date"], format="%d %b %Y")
    values = table["TotalReturnsIndex"].astype(str).str.replace(",", "").astype(float)
    series = pd.Series(values.to_numpy(), index=dates, name=name).sort_index()
    return series[~series.index.duplicated()]


def price_factor(subject: str) -> float | None:
    """Multiplier on the old price implied by a split or bonus subject, if it has one."""
    text = subject.lower()
    if "preference" in text or "debenture" in text:
        return None
    factor = None
    bonus = re.search(r"bonus\D*(\d+)\s*:\s*(\d+)", text)
    if bonus and int(bonus.group(2)) > 0:
        new, held = int(bonus.group(1)), int(bonus.group(2))
        factor = held / (held + new)
    split = re.search(r"(?:split|splt|sub-?division).*?(\d+(?:\.\d+)?).*?to\D*?(\d+(?:\.\d+)?)", text)
    if split and float(split.group(1)) > 0:
        factor = (factor or 1.0) * float(split.group(2)) / float(split.group(1))
    return factor


def dividend_amount(subject: str) -> float:
    """Total rupees per share across every dividend named in an action subject."""
    text = subject.lower().replace(" ", "")
    if "dividend" not in text and "dvd" not in text:
        return 0.0
    amounts = re.findall(r"(?:rs\.?|re\.?|₹)-?(\d+(?:\.\d+)?)", text)
    if not amounts:
        amounts = re.findall(r"(\d+(?:\.\d+)?)pershare", text)
    return float(sum(float(a) for a in amounts))


def integrated_rows(rows: list[dict]) -> pd.DataFrame:
    """Original quarterly results from the integrated-filing feed, in the older feed's columns."""
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table[table["type_Sub"].eq("Original") & table["broadcast_Date"].notna()]
    basis = {"Standalone": "Non-Consolidated", "Consolidated": "Consolidated"}
    return pd.DataFrame({
        "symbol": table["symbol"], "broadCastDate": table["broadcast_Date"],
        "toDate": table["qe_Date"].str.title(), "consolidated": table["consolidated"].map(basis),
        "xbrl": table["xbrl"], "period": "Quarterly", "cumulative": "Non-cumulative",
    }).dropna(subset=["consolidated"])


def download_integrated(folder: Path, start: date, end: date, pause: float = 1.5) -> int:
    """Archive the integrated-filing feed by month. Listed companies moved to it in early 2025."""
    folder.mkdir(parents=True, exist_ok=True)
    session = api_session()
    stored = 0
    for first, last in month_ranges(start, end):
        path = folder / f"{first:%Y-%m}.json"
        if path.exists():
            continue
        url = (f"{API}integrated-filing-results?index=equities&from_date={first:%d-%m-%Y}"
               f"&to_date={last:%d-%m-%Y}&type=Integrated%20Filing-%20Financials"
               f"&size={INTEGRATED_PAGE}")
        response = session.get(url, timeout=180)
        if response.status_code != 200:
            session = api_session()
            time.sleep(5)
            response = session.get(url, timeout=180)
        response.raise_for_status()
        found = response.json()
        rows = found["data"]
        if len(rows) < found["totalCount"]:
            raise ValueError(f"{first:%Y-%m}: got {len(rows)} of {found['totalCount']} filings")
        path.write_text(json.dumps(rows), encoding="utf-8")
        stored += 1
        time.sleep(pause)
    return stored


def load_integrated(folder: Path = INTEGRATED_DIR) -> pd.DataFrame:
    """All archived integrated filings, in the older feed's columns."""
    rows = [row for path in sorted(folder.glob("*.json"))
            for row in json.loads(path.read_text(encoding="utf-8"))]
    return integrated_rows(rows)
