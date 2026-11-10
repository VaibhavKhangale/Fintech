"""Parse archived filings and build the market-cap universe."""
import pandas as pd

from pmo import fundamentals as fu
from pmo import universe as un
from pmo.config import PROCESSED_DIR


def main() -> None:
    index = pd.read_parquet(PROCESSED_DIR / "filing_index.parquet")
    table = fu.parse_filings(index.drop(columns="symbol").rename(columns={"stock": "symbol"}))
    table.to_parquet(PROCESSED_DIR / "fundamentals.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "nse_panel.parquet", columns=["date", "symbol", "close"])
    actions = pd.read_parquet(PROCESSED_DIR / "nse_actions.parquet")
    liquid = un.month_end_universe(pd.read_parquet(PROCESSED_DIR / "nse_liquidity.parquet"), 400)
    closes = panel.merge(liquid[["date", "symbol"]], on=["date", "symbol"])
    caps = un.market_caps(closes, table, actions)
    largest = un.largest_stocks(caps, liquid)
    largest["month"] = largest["date"].dt.to_period("M").dt.to_timestamp("M").dt.normalize()
    largest.to_parquet(PROCESSED_DIR / "universe_monthly.parquet")
    print(len(table), largest.groupby("month").size().describe().round(1).to_dict(), flush=True)


if __name__ == "__main__":
    main()
