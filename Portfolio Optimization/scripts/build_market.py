"""Build the survivorship-free price panel, returns and liquidity universe from archived files."""
import pandas as pd
import yaml

from pmo import nse
from pmo import universe as un
from pmo.config import PROCESSED_DIR, RAW_DIR, ROOT


def main() -> None:
    changes = pd.read_csv(RAW_DIR / "symbol_changes.csv", parse_dates=["effective"])
    mapping = un.symbol_map(changes)
    panel = un.apply_symbols(nse.load_bhavcopies(), mapping)
    panel.to_parquet(PROCESSED_DIR / "nse_panel.parquet")
    fixes = yaml.safe_load((ROOT / "config" / "price_corrections.yaml").read_text(encoding="utf-8"))
    actions = un.action_table(nse.load_monthly(nse.ACTIONS_DIR), mapping, fixes["splits"])
    actions.to_parquet(PROCESSED_DIR / "nse_actions.parquet")
    daily = un.daily_returns(panel, actions)
    daily.to_parquet(PROCESSED_DIR / "nse_daily_returns.parquet")
    value = un.liquidity(panel)
    value.to_parquet(PROCESSED_DIR / "nse_liquidity.parquet")
    print(panel.shape, panel["date"].min(), panel["date"].max(), panel["symbol"].nunique(), flush=True)


if __name__ == "__main__":
    main()
