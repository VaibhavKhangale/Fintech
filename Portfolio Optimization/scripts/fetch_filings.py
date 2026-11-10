"""Archive standalone quarterly filings for stocks in the broad liquid set each quarter."""
from datetime import date

import pandas as pd

from pmo import fundamentals as fu
from pmo import nse
from pmo import universe as un
from pmo.config import PROCESSED_DIR, RAW_DIR

BROAD_SET = 300
INTEGRATED_START = date(2025, 1, 1)


def main() -> None:
    mapping = un.symbol_map(pd.read_csv(RAW_DIR / "symbol_changes.csv", parse_dates=["effective"]))
    nse.download_integrated(nse.INTEGRATED_DIR, INTEGRATED_START, date.today())
    reported = pd.concat([nse.load_monthly(nse.RESULTS_DIR), nse.load_integrated()], ignore_index=True)
    filings = fu.first_filings(reported)
    filings = filings[filings["consolidated"] == "Non-Consolidated"].copy()
    filings["stock"] = filings["symbol"].map(lambda s: mapping.get(s, s))
    leaders = un.month_end_universe(pd.read_parquet(PROCESSED_DIR / "nse_liquidity.parquet"), BROAD_SET)
    keys = set(zip(leaders["month"], leaders["symbol"]))
    filings["month"] = filings["quarter_end"].dt.to_period("M").dt.to_timestamp("M").dt.normalize()
    keep = [(m, s) in keys for m, s in zip(filings["month"], filings["stock"])]
    chosen = filings[keep]
    print("filings to fetch", len(chosen), flush=True)
    chosen.to_parquet(PROCESSED_DIR / "filing_index.parquet")
    print("stored", fu.download_filings(chosen), flush=True)


if __name__ == "__main__":
    main()
