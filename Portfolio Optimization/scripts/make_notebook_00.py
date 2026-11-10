"""Generate notebooks/00_data_overview.ipynb from source cells."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

MD, CODE = "md", "code"
CELLS = [
    (MD, """# Stage 0: Data foundation

This notebook checks the dataset every later stage builds on. It uses 25 NIFTY 50 stocks across
12 sectors, with monthly returns in INR.

Rebuild the data with `python scripts/build_data.py --refresh`."""),
    (CODE, """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmo.config import ROOT, load_config
from pmo.pipeline import load
from pmo.plotting import FIGURES, MUTED, SERIES_1, SERIES_2, apply_style

apply_style()
config = load_config()
data = load()
stocks = data.universe["symbol"].tolist()
pd.Series({k: v for k, v in data.manifest.items()
           if k in ("price_as_of", "months", "first_month", "last_month", "holdout_start",
                    "factors_end", "iima_file")}).to_frame("value")"""),
    (MD, """## 1. Universe

The stocks are today's NIFTY 50 members with history covering the whole window, so the sample
contains only companies that succeeded. **Survivorship bias lifts their past returns**, and
results in later stages should be read with that in mind."""),
    (CODE, """months = data.periods.index
summary = pd.DataFrame({
    "sector": data.universe.set_index("symbol")["sector"],
    "ann_return": data.returns[stocks].mean() * 12,
    "ann_volatility": data.returns[stocks].std() * np.sqrt(12),
})
summary.sort_values("ann_return", ascending=False).round(3)"""),
    (MD, """## 2. Data defects found and fixed

- **Market proxy misprint.** Yahoo shows NIFTYBEES at one-tenth of its price on 19 and 20 December
  2019. These two ticks are removed.
- **Unadjusted demergers.** ITC (ITC Hotels) and Hindustan Unilever (Kwality Wall's) show false
  price drops on their ex-dates. Earlier prices are back-adjusted so the ex-date return equals the
  market's. This is an approximation, because exact spun-off values are not available.

The table shows raw and cleaned returns on those dates."""),
    (CODE, """raw = pd.read_parquet(ROOT / "data" / "raw" / "prices.parquet")
rows = []
for action in config.corporate_actions:
    symbol, ex_date = action["symbol"], pd.Timestamp(action["ex_date"])
    rows.append({
        "symbol": symbol, "event": action["event"], "ex_date": ex_date.date(),
        "raw_return": raw[symbol].pct_change()[ex_date],
        "clean_return": data.prices[symbol].pct_change()[ex_date],
        "index_return": raw[config.price_index].pct_change()[ex_date],
    })
pd.DataFrame(rows).round(4)"""),
    (MD, """Daily moves more than 15% away from the market, kept in the data for review. These are not
treated as errors unless a corporate action explains them."""),
    (CODE, """jumps = pd.DataFrame(data.manifest["jumps_for_review"])
jumps["date"] = pd.to_datetime(jumps["date"]).dt.date
jumps.round(3)"""),
    (MD, """## 3. Market proxy

The Nifty 50 Total Return Index could not be downloaded, so NIFTYBEES stands in for it.

**Known weakness:** the ETF only beats the price index from 2016, by about 1% a year. Before that
the gap is close to zero, which suggests dividends paid out by the fund are missing from Yahoo's
adjusted prices. Market returns are therefore understated by about one dividend yield in 2011 to
2015. This has little effect on betas, but it slightly overstates alphas in Stages 7 and 8."""),
    (CODE, """market = data.returns[[config.market_proxy, config.price_index]]
growth = 100 * (1 + market).cumprod()
gap = (market[config.market_proxy] - market[config.price_index])
print(f"Average return gap: {gap.mean() * 12:.2%} a year; tracking difference {gap.std() * 12 ** .5:.2%}")

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(growth.index, growth[config.market_proxy], color=SERIES_1, label="NIFTYBEES")
ax.plot(growth.index, growth[config.price_index], color=SERIES_2, label="Nifty 50 price index")
ax.set_yscale("log")
ticks = [100, 200, 400, 800]
ax.set_yticks(ticks, [f"₹{t}" for t in ticks])
ax.minorticks_off()
ax.set_title("NIFTYBEES only pulls ahead of the price index from 2016")
ax.set_ylabel("Value of ₹100, log scale")
ax.legend(frameon=False, loc="upper left")
fig.savefig(FIGURES / "00_market_proxy.png")
plt.show()"""),
    (MD, """## 4. Risk-free rate

IIMA derives its monthly RF from RBI 91-day T-bill yields but stops at December 2025. RBI's Weekly
Statistical Supplement fills the months after that. Over 2025, where both exist, the two can be
compared directly."""),
    (CODE, """from pmo import returns as ret

weekly = pd.read_parquet(ROOT / "data" / "raw" / "rbi_tbill_weekly.parquet")["tbill_91d"]
rbi = ret.yield_to_monthly_return(weekly)
iima = pd.read_parquet(ROOT / "data" / "raw" / "iima_factors.parquet")["RF"]
overlap = pd.concat([iima.rename("iima"), rbi.rename("rbi")], axis=1).dropna()
print(f"Overlap: {len(overlap)} months, mean annual gap {(overlap.iima - overlap.rbi).mean() * 12:.3%}, "
      f"largest monthly gap {(overlap.iima - overlap.rbi).abs().max() * 12:.2%} annualized")

annual = data.risk_free["rf"] * 12
splice = data.risk_free.index[data.risk_free["source"] == "RBI WSS"].min()
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(annual.index, annual * 100, color=SERIES_1)
ax.axvline(splice, color=MUTED, linewidth=1, linestyle=":")
ax.annotate("RBI data from here", (splice, annual.max() * 100), xytext=(-8, 0),
            textcoords="offset points", ha="right", va="top", color=MUTED)
ax.set_title("The risk-free rate halved by 2021, then partly recovered")
ax.set_ylabel("Annualized %")
ax.set_ylim(0)
fig.savefig(FIGURES / "00_risk_free.png")
plt.show()"""),
    (MD, """## 5. Factors

`MKT_RF` is IIMA's market factor, a broad market that is wider than the Nifty 50. Its definition
was not confirmed in IIMA's FAQ, so this cell tests whether it is an excess return. If it is,
adding RF should give a total market return close to NIFTYBEES.
Factors end at December 2025, so regressions that use them cannot reach the end of the holdout."""),
    (CODE, """joined = pd.concat([data.factors, data.risk_free["rf"], data.returns[config.market_proxy]],
                   axis=1).dropna()
total = joined["MKT_RF"] + joined["rf"]
print(f"Annual mean: MKT_RF {joined.MKT_RF.mean() * 12:.1%}, MKT_RF + RF {total.mean() * 12:.1%}, "
      f"NIFTYBEES {joined[config.market_proxy].mean() * 12:.1%}")
print(f"Correlation with NIFTYBEES: {total.corr(joined[config.market_proxy]):.3f}")
pd.DataFrame({"ann_mean": data.factors.mean() * 12,
              "ann_volatility": data.factors.std() * np.sqrt(12)}).round(3)"""),
    (MD, """## 6. Estimation and holdout split

Optimization in Stages 1 to 9 uses only the estimation months. The holdout stays untouched until
the Stage 10 backtest."""),
    (CODE, """split = data.periods.groupby(data.periods).agg(
    months="size",
    start=lambda s: s.index.min().date(),
    end=lambda s: s.index.max().date(),
)
split"""),
]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(text) if kind == MD else nbf.v4.new_code_cell(text)
        for kind, text in CELLS
    ]
    (ROOT / "notebooks" / "00_data_overview.ipynb").write_text(nbf.writes(notebook),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()
