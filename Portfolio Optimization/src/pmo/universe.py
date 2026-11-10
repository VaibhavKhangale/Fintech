"""Survivorship-free stock history built from NSE daily files.

Every stock that ever traded is kept, including those later delisted. Each month-end universe uses
only data available on that date.
"""
import numpy as np
import pandas as pd

from pmo.nse import dividend_amount, price_factor

# Events whose ex-date price drop is not a loss to holders, valued separately or not at all.
SPIN_OFF_WORDS = ("demerger", "scheme of arrangement", "spin off", "spin-off", "rights")
MINIMUM_PRICE = 20.0


def symbol_map(changes: pd.DataFrame) -> dict[str, str]:
    """Map every old symbol to its latest symbol, following chains of renames in date order."""
    ordered = changes.sort_values("effective")
    mapping: dict[str, str] = {}
    for old, new in zip(ordered["old"], ordered["new"]):
        if old == new:
            continue
        mapping[old] = new
        for key, value in mapping.items():
            if value == old:
                mapping[key] = new
    return mapping


def apply_symbols(panel: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Give each stock one identity across renames, keeping its busiest line each day."""
    table = panel.copy()
    table["symbol"] = table["symbol"].map(lambda s: mapping.get(s, s))
    table = table.sort_values(["date", "symbol", "value"])
    return table.drop_duplicates(["date", "symbol"], keep="last").reset_index(drop=True)


def action_table(actions: pd.DataFrame, mapping: dict[str, str],
                 corrections: list[dict] | None = None) -> pd.DataFrame:
    """Parse raw corporate actions into price factors, dividends and spin-off flags.

    `corrections` adds splits that NSE's action list omits.
    """
    table = actions.loc[actions["exDate"].astype(str).str.len() > 1,
                        ["symbol", "exDate", "subject"]].copy()
    table["date"] = pd.to_datetime(table["exDate"], format="%d-%b-%Y", errors="coerce")
    table = table.dropna(subset=["date"]).drop_duplicates(["symbol", "date", "subject"])
    table["symbol"] = table["symbol"].str.strip().map(lambda s: mapping.get(s, s))
    subject = table["subject"].astype(str)
    table["factor"] = subject.map(price_factor).fillna(1.0)
    table["dividend"] = subject.map(dividend_amount)
    table["spin_off"] = subject.str.lower().map(lambda s: any(w in s for w in SPIN_OFF_WORDS))
    extra = pd.DataFrame(corrections or [], columns=["symbol", "date", "factor"])
    extra["date"] = pd.to_datetime(extra["date"])
    extra[["dividend", "spin_off"]] = [0.0, False]
    table = pd.concat([table.astype({"spin_off": bool}), extra.astype({"spin_off": bool})],
                      ignore_index=True)
    return (table.groupby(["symbol", "date"], as_index=False)
            .agg(factor=("factor", "prod"), dividend=("dividend", "sum"),
                 spin_off=("spin_off", "any")))


def daily_returns(panel: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Total daily returns per stock, adjusted for splits, bonuses and dividends.

    Returns span any gap between a stock's trading days. On a spin-off's ex-date the return is
    replaced by that day's median stock return, as the spun-off value is not priced that day.
    """
    table = panel.sort_values(["symbol", "date"]).merge(actions, on=["symbol", "date"], how="left")
    table[["factor", "dividend"]] = table[["factor", "dividend"]].fillna({"factor": 1.0,
                                                                          "dividend": 0.0})
    table["spin_off"] = table["spin_off"].astype("boolean").fillna(False).astype(bool)
    previous = table.groupby("symbol")["close"].shift()
    table["return"] = (table["close"] + table["dividend"]) / (previous * table["factor"]) - 1
    median = table.groupby("date")["return"].transform("median")
    table.loc[table["spin_off"], "return"] = median[table["spin_off"]]
    return table[["date", "symbol", "return"]].pivot(index="date", columns="symbol",
                                                     values="return")


def unexplained_jumps(returns: pd.DataFrame, limit: float = 0.4) -> pd.DataFrame:
    """Daily moves too large to accept without review, often a missing corporate action."""
    rows = returns.stack().rename("return").reset_index()
    rows.columns = ["date", "symbol", "return"]
    return rows[rows["return"].abs() > limit].sort_values("return")


def liquidity(panel: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Trailing median daily traded value of each operating company's shares.

    Quiet days count as zero. ETFs (INF codes), other series and penny stocks are left out.
    """
    funds = set(panel.loc[panel["isin"].astype(str).str.startswith("INF"), "symbol"])
    shares = panel[(panel["series"] == "EQ") & ~panel["symbol"].isin(funds)
                   & (panel["close"] >= MINIMUM_PRICE)]
    wide = (shares.pivot(index="date", columns="symbol", values="value")
            .reindex(panel["date"].drop_duplicates().sort_values()))
    return wide.fillna(0.0).rolling(window, min_periods=window).median()


MAX_SHARE_JUMP = 3.0
SAME_COUNT = 0.10


def month_end_universe(value: pd.DataFrame, size: int = 200) -> pd.DataFrame:
    """Most liquid stocks at each month end, using only data up to that date."""
    ends = value.groupby(value.index.to_period("M")).tail(1)
    rows = []
    for day, row in ends.iterrows():
        leaders = row[row > 0].nlargest(size)
        rows.append(pd.DataFrame({"date": day, "symbol": leaders.index, "rank": range(1, len(leaders) + 1)}))
    table = pd.concat(rows, ignore_index=True)
    table["month"] = table["date"].dt.to_period("M").dt.to_timestamp("M").dt.normalize()
    return table


def monthly_total_returns(daily: pd.DataFrame, final_loss: float = 0.0) -> pd.DataFrame:
    """Compound daily returns by month. A stock's last month can take a delisting loss.

    A stock that stops trading is assumed to pay out its last close, less `final_loss`.
    """
    growth = (1 + daily.fillna(0.0)).groupby(daily.index.to_period("M")).prod()
    traded = daily.notna().groupby(daily.index.to_period("M")).any()
    out = (growth - 1).where(traded)
    out.index = out.index.to_timestamp("M").normalize()
    out.index.name = "date"
    if final_loss:
        last_month = traded.index.to_timestamp("M").normalize()
        for symbol in out:
            valid = out[symbol].last_valid_index()
            if valid is not None and valid < last_month[-1]:
                out.loc[valid, symbol] = (1 + out.loc[valid, symbol]) * (1 - final_loss) - 1
    return out


def share_levels(actions: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share multiplier for each stock after each split or bonus.

    Spin-off days are left out. They don't change the share count, and the factors recorded on them
    can describe something else, such as a bonus of preference shares.
    """
    moves = actions["factor"].ne(1.0) & ~actions["spin_off"]
    steps = (actions.loc[moves, ["symbol", "date", "factor"]]
             .astype({"symbol": object, "date": "datetime64[ns]"}).sort_values("date"))
    steps["level"] = (1 / steps["factor"]).groupby(steps["symbol"]).cumprod()
    return steps[["symbol", "date", "level"]].rename(columns={"date": "effective"})


def _level_on(keys: pd.DataFrame, column: str, levels: pd.DataFrame) -> pd.Series:
    """The share multiplier in force on `keys[column]`, aligned to `keys`."""
    frame = keys[["symbol", column]].astype({"symbol": object, column: "datetime64[ns]"})
    frame = frame.assign(row=range(len(frame))).sort_values(column, kind="stable")
    found = pd.merge_asof(frame, levels, left_on=column, right_on="effective", by="symbol")
    return found.set_index("row")["level"].sort_index().fillna(1.0).set_axis(keys.index)


def steady_shares(filings: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Filed share counts in split-free units, with unexplained jumps held back.

    A jump beyond MAX_SHARE_JUMP that no split explains is usually a typing error, so the previous
    count stays until a second filing agrees with the new one. A real change, such as a merger, is
    therefore accepted one filing late, and nothing here looks ahead.
    """
    table = (filings.dropna(subset=["shares", "quarter_end"]).sort_values(["symbol", "published"])
             .reset_index(drop=True))
    units = table["shares"] / _level_on(table, "quarter_end", levels)
    held = []
    accepted = pending = last_symbol = None
    for symbol, value in zip(table["symbol"], units):
        if symbol != last_symbol:
            accepted, pending, last_symbol = value, None, symbol
        elif 1 / MAX_SHARE_JUMP <= value / accepted <= MAX_SHARE_JUMP:
            accepted, pending = value, None
        elif pending is not None and abs(value / pending - 1) <= SAME_COUNT:
            accepted, pending = value, None
        else:
            pending = value
        held.append(accepted)
    return table[["symbol", "published"]].assign(units=held)


def market_caps(closes: pd.DataFrame, filings: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Market value at each date in `closes`, using only share counts already published.

    `closes` has date, symbol and close. Filings give shares as of their quarter end; any split or
    bonus between then and the date scales that count.
    """
    levels = share_levels(actions)
    known = (steady_shares(filings, levels).assign(filed=lambda t: t["published"].dt.normalize())
             .astype({"symbol": object, "filed": "datetime64[ns]"}).sort_values("filed")
             [["symbol", "filed", "units"]])
    left = closes.astype({"symbol": object, "date": "datetime64[ns]"}).sort_values("date")
    merged = pd.merge_asof(left, known, left_on="date", right_on="filed", by="symbol",
                           allow_exact_matches=False)
    merged["market_cap"] = merged["close"] * merged["units"] * _level_on(merged, "date", levels)
    return merged[["date", "symbol", "market_cap"]]


def largest_stocks(caps: pd.DataFrame, liquid: pd.DataFrame, size: int = 200,
                   liquid_rank_limit: int = 400) -> pd.DataFrame:
    """Largest stocks by market value at each date, among those liquid enough to trade."""
    table = caps.dropna(subset=["market_cap"]).merge(
        liquid.loc[liquid["rank"] <= liquid_rank_limit, ["date", "symbol"]], on=["date", "symbol"])
    table["rank"] = table.groupby("date")["market_cap"].rank(ascending=False, method="first")
    return table[table["rank"] <= size].sort_values(["date", "rank"]).reset_index(drop=True)


def overlap(rule_based: set[str], official: set[str]) -> float:
    """Share of an official list that the rule-based list also holds."""
    return len(rule_based & official) / len(official) if official else np.nan
