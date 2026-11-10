# Portfolio Management and Optimization (India)

A study of modern portfolio theory on Indian equities, built one stage at a time and ending in an
interactive portfolio builder. This is an educational and research project, not investment advice.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe scripts/build_data.py --refresh
.venv/Scripts/python.exe -m pytest
```

`requirements-lock.txt` records the exact package versions used.

If notebooks fail with `Bad file descriptor (epoll.cpp)`, Anaconda's `Library/bin` is on `PATH`
and its ZeroMQ library is being loaded. Remove it from `PATH` for that session.

## Layout

| Path | Purpose |
|---|---|
| `config/project.yaml` | Universe, dates, sources and corrections |
| `src/pmo/` | Shared engine, used by both notebooks and the final tool |
| `scripts/` | Data build and notebook generation |
| `notebooks/` | One notebook per stage |
| `tests/` | Offline tests |
| `data/` | Raw and processed data, not tracked |
| `reports/figures/` | Saved charts |

## Data

| Dataset | Source | Coverage and notes |
|---|---|---|
| Stock prices | Yahoo Finance via yfinance | Dividend- and split-adjusted daily closes |
| Market proxy | NIFTYBEES ETF | Stands in for the Nifty 50 Total Return Index, which could not be downloaded |
| Factors | [IIMA Fama-French and momentum library](https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/) | Survivorship-adjusted; ends December 2025 |
| Risk-free rate | IIMA RF, then RBI Weekly Statistical Supplement | 91-day T-bill basis throughout |
| Index members | niftyindices.com | Current membership only |
| Gold | GOLDBEES ETF via yfinance | One December 2019 misprint removed |
| Bonds | SBI 10 Year Constant Maturity Gilt Fund NAV via mfapi.in | Stand-in for gilts; its risk looks understated |
| Cash check | HDFC Liquid Fund NAV via mfapi.in | Unit-size change in 2015 removed; tracks T-bills closely |

Limitations:

- The universe is today's index members, so returns carry survivorship bias.
- Two demergers are adjusted by assumption (see `corporate_actions` in the config).
- Yahoo is an unofficial feed. A few misprints were corrected and others may remain.
- NIFTYBEES appears to miss dividends before 2016, so market returns are slightly low in 2011 to 2015.
- Debt ETFs have short histories, so a gilt fund stands in for bonds, and its volatility looks too low.

## Stage 12: Data foundation

`notebooks/12_data_foundation.ipynb` passes every check, using development-period returns only. The data
covers every NSE stock from June 2010, 17,510 standalone quarterly filings stamped with their publication
time, and a monthly universe of the 200 largest stocks among the 400 most traded. Rebuild it with:

```bash
python scripts/fetch_nse.py bhavcopy
python scripts/fetch_nse.py api
python scripts/build_market.py
python scripts/fetch_filings.py
python scripts/build_fundamentals.py
```
