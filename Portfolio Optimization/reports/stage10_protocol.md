# Stage 10 protocol

Written on 2026-11-07, before any holdout result was calculated. Nothing below changes after the test
runs. Failures get reported, not tuned away.

## Test period

The 36 months from November 2023 to October 2026. Portfolios are set at the end of October 2023, 2024
and 2025, using only data available on those dates, and held for 12 months with holdings drifting.

## Frozen method

- **Stock fund:** long-only, 10% stock cap, tangency portfolio on practical inputs (50% historical
  mean, 50% CAPM, single-index covariance), trailing 60 months, 3-point survivorship haircut, trailing
  T-bill rate.
- **Policies (stock fund / gold / bonds):** Cautious 25/15/60, Balanced 60/20/20, Growth 80/10/10.

## Portfolios tested

| Group | Portfolios |
|---|---|
| Stock level | Stock fund, capped lowest risk, equal weight, market tracker, Nifty |
| Policies | The three policies with the stock fund, and with the Nifty in its place |
| Chance benchmark | 2,000 random 12-stock equal-weight portfolios, redrawn yearly |

## Costs and tax

- **Stocks:** 0.20% per rupee traded, covering STT, stamp duty, exchange fees and spread.
- **ETFs:** 0.05%.
- **Bond fund and cash:** none, as fund fees are already in the NAV.
- **Tax scenario:** 12.5% on realized gains for stocks, the Nifty ETF and gold, and 30% for the bond
  fund, with no annual exemption. This is an illustrative worst case on rebalancing gains only.

## Measures

Excess return, volatility, Sharpe, maximum drawdown, turnover, alpha against the Nifty and against the
tracker, and percentile against random portfolios. Promised and realized volatility are compared.

## Decision rules

1. **Equity sleeve in the tool.** Keep the stock fund as the default only if, after costs, its maximum
   drawdown is smaller than both equal weight's and the Nifty's, and its Sharpe is not below the
   Nifty's. Otherwise the default becomes a Nifty ETF.
2. **Policies.** A policy passes if, after costs, its maximum drawdown is smaller than the Nifty's and
   ordered by risk: Cautious smallest, Growth largest. If a policy fails, the tool shows it with that
   warning rather than changing the weights.
3. **Claims.** The tool makes no claim of outperformance unless the fund's alpha against the tracker
   has a t-statistic above 2 and it ranks above the 95th percentile of random portfolios on Sharpe.

## Known limits

- 36 months is short: the standard error of a Sharpe ratio is about 0.6.
- The universe was chosen from the November 2026 index list, so stocks that left the index by then
  were excluded. The holdout therefore keeps some survivorship, and the tracker and random
  portfolios are the fair comparisons.
- Summary statistics over all months, including the holdout, were displayed in Stage 0, and the audit
  listed dates of large holdout moves. Nothing was tuned on them, and the method predates this file.
