"""Build the study dataset: download or reuse raw data, clean it, check it, save it."""
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from pmo import cleaning, quality, returns, sources
from pmo.config import PROCESSED_DIR, RAW_DIR, Config, load_config


@dataclass(frozen=True)
class Dataset:
    prices: pd.DataFrame        # daily, cleaned, stocks + market proxy + price index
    returns: pd.DataFrame       # monthly simple returns, same columns
    excess: pd.DataFrame        # monthly returns less the risk-free rate
    risk_free: pd.DataFrame     # columns rf, source
    factors: pd.DataFrame       # SMB, HML, WML, MKT_RF (no RF; see risk_free)
    factors_long: pd.DataFrame  # same factors from 1993, for long-run premia
    periods: pd.Series          # 'estimation' or 'holdout' per month
    universe: pd.DataFrame      # symbol, sector, current index membership
    manifest: dict


def _raw_inputs(config: Config, refresh: bool) -> dict:
    paths = {name: RAW_DIR / file for name, file in {
        "prices": "prices.parquet", "factors": "iima_factors.parquet",
        "tbill": "rbi_tbill_weekly.parquet", "nifty50": "nifty50_list.csv",
        "nifty100": "nifty100_list.csv", "meta": "raw_meta.json"}.items()}
    if refresh or not all(p.exists() for p in paths.values()):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        symbols = config.symbols + [config.market_proxy, config.price_index]
        sources.fetch_prices(symbols, config.start).to_parquet(paths["prices"])
        factors, factor_file = sources.fetch_iima_factors(config)
        factors.to_parquet(paths["factors"])
        since = date(factors.index.max().year, 1, 1)
        sources.fetch_rbi_tbill(config, since).to_frame().to_parquet(paths["tbill"])
        sources.fetch_constituents(config.sources["nifty50_list"]).to_csv(
            paths["nifty50"], index=False)
        sources.fetch_constituents(config.sources["nifty100_list"]).to_csv(
            paths["nifty100"], index=False)
        paths["meta"].write_text(json.dumps({
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "iima_file": factor_file}), encoding="utf-8")
    return {
        "prices": pd.read_parquet(paths["prices"]),
        "factors": pd.read_parquet(paths["factors"]),
        "tbill": pd.read_parquet(paths["tbill"])["tbill_91d"],
        "nifty50": pd.read_csv(paths["nifty50"]),
        "nifty100": pd.read_csv(paths["nifty100"]),
        "meta": json.loads(paths["meta"].read_text(encoding="utf-8")),
    }


def clean_prices(raw: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, int]:
    """Repair the market proxy and apply configured demerger adjustments."""
    prices = raw.copy()
    market, index = config.market_proxy, config.price_index
    prices[market] = cleaning.drop_bad_ticks(prices[market], prices[index])
    dropped = int(prices[market].isna().sum() - raw[market].isna().sum())
    for action in config.corporate_actions:
        symbol = action["symbol"]
        prices[symbol] = cleaning.neutralize_event(prices[symbol], prices[index],
                                                   action["ex_date"])
    return prices, dropped


def _table(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build(refresh: bool = False, config: Config | None = None) -> Dataset:
    config = config or load_config()
    raw = _raw_inputs(config, refresh)

    prices, ticks_dropped = clean_prices(raw["prices"], config)
    as_of = prices.index.max()
    monthly = returns.monthly_returns(returns.complete_month_prices(prices, as_of))

    rbi_rf = returns.yield_to_monthly_return(raw["tbill"])
    risk_free = returns.splice_risk_free(raw["factors"]["RF"], rbi_rf).reindex(monthly.index)
    if risk_free["rf"].isna().any():
        missing = risk_free.index[risk_free["rf"].isna()].strftime("%Y-%m").tolist()
        raise RuntimeError(f"No risk-free rate for months: {missing}")
    excess = monthly.sub(risk_free["rf"], axis=0)
    factors = raw["factors"].drop(columns="RF")
    factors = factors[factors.index.isin(monthly.index)]
    factors_long = raw["factors"].drop(columns="RF")
    periods = returns.split_periods(monthly.index, config.holdout_months)

    universe = pd.DataFrame({"symbol": config.symbols,
                             "sector": [config.sectors[s] for s in config.symbols]})
    universe["in_nifty50_now"] = universe["symbol"].isin(raw["nifty50"]["Symbol"])
    universe["in_nifty100_now"] = universe["symbol"].isin(raw["nifty100"]["Symbol"])

    stocks = prices[config.symbols]
    checks = {
        "missing_days": quality.missing_share(stocks, config.max_missing_share),
        "late_starts": quality.late_starts(stocks, pd.Timestamp(config.start)),
        "monthly_gaps": quality.monthly_gaps(monthly),
        "rf_out_of_range": quality.rf_out_of_range(risk_free["rf"]),
        "not_in_nifty50": universe.loc[~universe["in_nifty50_now"], ["symbol"]],
    }
    jumps = quality.market_adjusted_jumps(stocks, prices[config.price_index],
                                          config.jump_threshold)
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "downloaded_at": raw["meta"]["downloaded_at"],
        "iima_file": raw["meta"]["iima_file"],
        "price_as_of": as_of.date().isoformat(),
        "months": len(monthly),
        "first_month": monthly.index.min().date().isoformat(),
        "last_month": monthly.index.max().date().isoformat(),
        "holdout_start": periods[periods == "holdout"].index.min().date().isoformat(),
        "factors_end": factors.index.max().date().isoformat(),
        "bad_market_ticks_dropped": ticks_dropped,
        "corporate_actions": list(config.corporate_actions),
        "rf_months_by_source": risk_free["source"].value_counts().to_dict(),
        "failed_checks": {k: _table(v.reset_index()) for k, v in checks.items() if not v.empty},
        "jumps_for_review": _table(jumps),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PROCESSED_DIR / "prices_daily.parquet")
    monthly.to_parquet(PROCESSED_DIR / "returns_monthly.parquet")
    excess.to_parquet(PROCESSED_DIR / "excess_returns_monthly.parquet")
    risk_free.to_parquet(PROCESSED_DIR / "risk_free_monthly.parquet")
    factors.to_parquet(PROCESSED_DIR / "factors_monthly.parquet")
    factors_long.to_parquet(PROCESSED_DIR / "factors_long_monthly.parquet")
    periods.to_frame().to_parquet(PROCESSED_DIR / "periods.parquet")
    universe.to_csv(PROCESSED_DIR / "universe.csv", index=False)
    (PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return Dataset(prices, monthly, excess, risk_free, factors, factors_long, periods, universe,
                   manifest)


def dividend_data(refresh: bool = False, config: Config | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dividends per share and closes, downloaded once and reused."""
    config = config or load_config()
    dividends_path, closes_path = RAW_DIR / "dividends.parquet", RAW_DIR / "closes.parquet"
    if refresh or not (dividends_path.exists() and closes_path.exists()):
        dividends, closes = sources.fetch_dividends(config.symbols, config.start)
        dividends.to_parquet(dividends_path)
        closes.to_parquet(closes_path)
    return pd.read_parquet(dividends_path), pd.read_parquet(closes_path)


def asset_class_prices(refresh: bool = False, config: Config | None = None) -> pd.DataFrame:
    """Daily gold, government bond and liquid fund prices, downloaded once and reused."""
    config = config or load_config()
    path = RAW_DIR / "asset_classes.parquet"
    if refresh or not path.exists():
        names = config.asset_classes
        gold = sources.fetch_prices([names["gold_etf"]], config.start)[names["gold_etf"]]
        table = pd.DataFrame({
            "Gold": cleaning.drop_bad_ticks(gold, pd.Series(1.0, index=gold.index)),
            "Bonds": sources.fetch_mf_nav(config, names["bond_scheme"]),
            "Liquid fund": cleaning.remove_unit_change(
                sources.fetch_mf_nav(config, names["liquid_scheme"])),
        })
        table.loc[config.start:].to_parquet(path)
    return pd.read_parquet(path)


def load() -> Dataset:
    """Load the processed dataset written by `build`."""
    read = lambda name: pd.read_parquet(PROCESSED_DIR / name)
    return Dataset(
        prices=read("prices_daily.parquet"),
        returns=read("returns_monthly.parquet"),
        excess=read("excess_returns_monthly.parquet"),
        risk_free=read("risk_free_monthly.parquet"),
        factors=read("factors_monthly.parquet"),
        factors_long=read("factors_long_monthly.parquet"),
        periods=read("periods.parquet")["period"],
        universe=pd.read_csv(PROCESSED_DIR / "universe.csv"),
        manifest=json.loads((PROCESSED_DIR / "manifest.json").read_text(encoding="utf-8")),
    )
