"""Quarterly results as first published, parsed from NSE's filing pages.

Filings up to 2018 are HTML pages and later ones are XBRL. Each record keeps the time the exchange
first published it, so a signal can use a number only after the market could have seen it.
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

from pmo.config import RAW_DIR
from pmo.nse import HEADERS

FILINGS_DIR = RAW_DIR / "nse_filings"
UNITS = {"lakhs": 1e5, "lacs": 1e5, "crores": 1e7, "crore": 1e7, "millions": 1e6, "million": 1e6,
         "thousands": 1e3, "rupees": 1.0}
MAX_FACE_VALUE = 100.0
MIN_EPS = 0.5
MAX_SHARE_DISAGREEMENT = 3.0
HTML_FIELDS = {
    "revenue": [r"^\(a\) net sales/income from operations", r"^net sales/income from operation",
                r"^total income from operations", r"^interest earned", r"^total income$"],
    "net_profit": [r"^net profit\s*\(\+\)\s*/\s*loss\s*\(-\)\s*for the period",
                   r"^net profit\s*/\s*\(loss\) for the period"],
    "face_value": [r"^face value"],
    "paid_up": [r"^paid-up equity share capital"],
    "eps": [r"^basic eps before extraordinary", r"^basic eps", r"^\(a\) basic"],
}
XBRL_FIELDS = {
    "revenue": ["RevenueFromOperations", "InterestEarned", "Income"],
    "net_profit": ["ProfitLossForPeriod", "ProfitLossForThePeriod"],
    "face_value": ["FaceValueOfEquityShareCapital"],
    "paid_up": ["PaidUpValueOfEquityShareCapital"],
    "eps": ["BasicEarningsLossPerShareFromContinuingOperations",
            "BasicEarningsPerShareBeforeExtraordinaryItems"],
}


def _number(text: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    match = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?", cleaned)
    if not match:
        return None
    value = float(match.group(1))
    return -abs(value) if cleaned.startswith("(") else value


def parse_html(page: str) -> dict:
    """Extract headline figures from an older filing page. Money fields become rupees."""
    cells = [re.sub(r"\s+", " ", c).replace("&nbsp;", " ").strip()
             for c in re.split(r"<[^>]+>", re.sub(r"(?s)<script.*?</script>", " ", page))]
    cells = [c for c in cells if c]
    unit = re.search(r"Rs\.?\s*in\s*(\w+)", " ".join(cells), re.IGNORECASE)
    scale = UNITS.get(unit.group(1).lower(), 1e5) if unit else 1e5
    out: dict[str, float | None] = {}
    for field, patterns in HTML_FIELDS.items():
        for pattern in patterns:
            for position, cell in enumerate(cells[:-1]):
                if re.search(pattern, cell, re.IGNORECASE):
                    value = _number(cells[position + 1])
                    if value is not None and field not in out:
                        out[field] = value
            if field in out:
                break
    for field in ("revenue", "net_profit", "paid_up"):
        if out.get(field) is not None:
            out[field] *= scale
    return out


def parse_xbrl(document: str) -> dict:
    """Extract headline figures for the reported quarter from a newer filing, in rupees."""
    out: dict[str, float] = {}
    for field, tags in XBRL_FIELDS.items():
        for tag in tags:
            pattern = rf'<(?:in-bse-fin|in-capmkt):{tag}\b[^>]*contextRef="([^"]+)"[^>]*>\s*([^<]+?)\s*<'
            matches = re.findall(pattern, document)
            preferred = [m for m in matches if m[0] == "OneD"] or matches
            if preferred:
                value = _number(preferred[0][1])
                if value is not None:
                    out[field] = value
                    break
    return out


def first_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """Keep each company's first quarterly filing per period and reporting basis."""
    table = filings[(filings["period"] == "Quarterly")
                    & (filings["cumulative"] == "Non-cumulative")].copy()
    table["published"] = pd.to_datetime(table["broadCastDate"], format="%d-%b-%Y %H:%M:%S",
                                        errors="coerce")
    table["quarter_end"] = pd.to_datetime(table["toDate"], format="%d-%b-%Y", errors="coerce")
    table["days_to_publish"] = (table["published"] - table["quarter_end"]).dt.days
    table = table.dropna(subset=["published", "quarter_end"]).sort_values("published")
    return table.drop_duplicates(["symbol", "quarter_end", "consolidated"], keep="first")


def filing_url(row: pd.Series) -> str | None:
    """The machine-readable page for a filing, if one exists."""
    xbrl = str(row.get("xbrl", ""))
    if xbrl.endswith(".xml"):
        return xbrl
    link = row.get("resultDetailedDataLink")
    return link if isinstance(link, str) and link.endswith(".html") else None


def download_filings(filings: pd.DataFrame, folder: Path = FILINGS_DIR, pause: float = 0.3) -> int:
    """Archive each filing page. Resumable; returns pages stored."""
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    stored = 0
    for _, row in filings.iterrows():
        url = filing_url(row)
        if url is None:
            continue
        path = folder / url.rsplit("/", 1)[1]
        if path.exists():
            continue
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(30)
            continue
        if response.status_code == 200:
            path.write_bytes(response.content)
            stored += 1
        time.sleep(pause)
    return stored


def share_count(table: pd.DataFrame) -> pd.Series:
    """Shares outstanding, from paid-up capital unless the filing's own EPS contradicts it.

    Filings contain typing errors: a face value of zero, or capital 100 times too large. Profit
    divided by EPS is an independent estimate, usable when EPS is too large to be mostly rounding,
    so it settles disagreements.
    """
    valid = table["face_value"].where(table["face_value"].between(0.01, MAX_FACE_VALUE))
    stated = (table["paid_up"] / valid).where(table["paid_up"] > 0)
    implied = (table["net_profit"] / table["eps"]).where(
        (table["eps"].abs() >= MIN_EPS) & (table["net_profit"] * table["eps"] > 0))
    ratio = stated / implied
    disagrees = (ratio > MAX_SHARE_DISAGREEMENT) | (ratio < 1 / MAX_SHARE_DISAGREEMENT)
    return stated.where(stated.notna() & ~disagrees, implied)


def parse_filings(filings: pd.DataFrame, folder: Path = FILINGS_DIR) -> pd.DataFrame:
    """Turn archived pages into a table, one row per filing."""
    rows = []
    for _, row in filings.iterrows():
        url = filing_url(row)
        if url is None:
            continue
        path = folder / url.rsplit("/", 1)[1]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        figures = parse_xbrl(text) if url.endswith(".xml") else parse_html(text)
        rows.append({"symbol": row["symbol"], "quarter_end": row["quarter_end"],
                     "published": row["published"], "consolidated": row["consolidated"],
                     **figures})
    table = pd.DataFrame(rows)
    table["shares"] = share_count(table)
    return table
