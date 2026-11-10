# Research protocol: beating the Nifty

Status: proposed on 2026-11-07, for approval. Once approved, changes need your agreement and are
recorded at the end of this file.

## Goal

Find a stock portfolio that can beat the Nifty 50 after costs, keeps updating its view of each stock,
and reacts to news and price changes. No method is assumed to work, and stopping is an acceptable
outcome.

## Benchmark and universe

- **Benchmark:** Nifty 50 Total Return Index. NIFTYBEES is the fallback, but it understates the
  benchmark before 2016, which would flatter any strategy.
- **Universe:** Nifty 200 using each date's actual members, so failed and removed stocks are included.

## Date splits

| Split | Period | Use |
|---|---|---|
| Development | Jan 2011 to Dec 2021 | All research. About 9 usable years after warm-up |
| Validation | Jan 2022 to Oct 2023 | Opened once per candidate, after your approval |
| Seen period | Nov 2023 to Oct 2026 | Labeled second check only, because Stage 10 inspected it |
| Paper trading | From the end of Stage 17 | The only fully unseen evidence |

Validation is not fully clean. Stage 2's factor premia and Stages 1 to 9 used those months, though
on a different stock list and for other purposes.

The engine refuses protected splits until an approval is recorded in
`reports/validation_unlocks.yaml`.

## Limit on ideas tried

- **At most 10 distinct ideas in development.** Variants of an idea, such as different lookbacks,
  count as separate ideas unless they were listed before testing.
- Every idea is written to `reports/research_log.csv`, including failures.
- Going past 10 needs your approval, and raises the hurdle: 2.81 at 10 ideas, 3.02 at 20, 3.29 at 50.

## Gates

| Stage | Passes if | If it fails |
|---|---|---|
| 12 Data | Past index members rebuilt for the whole period, with no future information in any field | Report the gap; a paid source may be needed |
| 13 Forecast signals | At least one long-only signal shows t of 2.0 or more against the Nifty in development, after costs | Stop and review with you |
| 14 Event signals | An event's price effect, measured from the next possible trade, has t of 2.0 or more after costs | Drop that event from forecasts; it may stay as a risk alert |
| 15 Portfolio | Rank correlation of weights and forecasts is at least 0.5, and active share is at least 50% | Redesign the optimizer |
| 16 Dynamic rules | Each rule beats plain holding after costs and tax, in development | Drop that rule |
| 17 Full test | All targets below are met in development, and validation active return is positive | Stop; don't start paper trading |
| 18 Paper trading | At least 6 months running reliably, alerts within one trading day, results near the backtest | Fix, then restart the clock |
| 19 Tool | Built only after Stage 18 passes | Not built |

## Targets for Stage 17

| Measure | Target |
|---|---|
| Active return against the Nifty, after costs | At least 3 points a year |
| Information ratio | At least 0.5, on about 6% tracking error |
| t-statistic in development | At least 2.0 |
| Random portfolio rank | Top 5%, on a matching stock count and turnover |
| Turnover | At most 100% a year one-way, unless a higher rate clearly pays for itself |
| After-tax active return | Positive, reported separately |

Results on the seen period are reported, but they don't decide anything alone.

## Costs and tax

| Stock group | Cost per side | Stress case |
|---|---|---|
| Nifty 100 | 0.20% | 0.40% |
| Other Nifty 200 | 0.35% | 0.70% |

Trades happen at the next day's open or later, never at the price that caused the signal. Tax
scenario: 20% on gains held 12 months or less, 12.5% beyond that, with no annual exemption.

## Stop rules during paper trading

Review with you if the portfolio falls 15 points behind the Nifty, or a data feed fails for more
than one trading day.

## What these gates can and can't show

Even a truly good strategy can fail them. With t of 2.0 in development and positive validation, one
with an information ratio of 0.5 passes only about 23% of the time, and 0.75 about 51%. Across 10
independent skill-free ideas, the chance that one passes is about 11%, and paper trading is the
further check on those.

## Changes

None yet.
