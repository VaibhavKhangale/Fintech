"""Research safeguards: date splits, trial logging, hurdles, gate power and cost budgets."""
import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from pmo.config import ROOT

SPLITS_PATH = ROOT / "config" / "research.yaml"
UNLOCKS_PATH = ROOT / "reports" / "validation_unlocks.yaml"
LOG_PATH = ROOT / "reports" / "research_log.csv"
LOG_FIELDS = ["logged_at", "stage", "trial", "split", "metric", "value", "notes"]


def load_splits(path: Path = SPLITS_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["splits"]


def unlocked(candidate: str, path: Path = UNLOCKS_PATH) -> bool:
    """True if the user has approved a one-time validation run for this candidate."""
    if not path.exists():
        return False
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return candidate in entries


def window(frame: pd.DataFrame, split: str, candidate: str | None = None,
           splits: dict | None = None, unlocks_path: Path = UNLOCKS_PATH) -> pd.DataFrame:
    """Rows of `frame` in a date split. Protected splits need a recorded approval."""
    settings = (splits or load_splits())[split]
    if settings.get("protected") and not (candidate and unlocked(candidate, unlocks_path)):
        raise PermissionError(f"The {split} split needs an approved unlock for this candidate")
    return frame.loc[settings["start"]:settings["end"]]


def hurdle_t(trials: int, level: float = 0.05) -> float:
    """Two-sided t hurdle after a Bonferroni correction for the number of trials."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    return float(norm.ppf(1 - level / (2 * trials)))


def log_trial(stage: str, trial: str, split: str, metric: str, value: float, notes: str = "",
              path: Path = LOG_PATH) -> None:
    """Append one tried idea to the log, so failed attempts are counted as well."""
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
                         "stage": stage, "trial": trial, "split": split, "metric": metric,
                         "value": value, "notes": notes})


def count_trials(path: Path = LOG_PATH) -> int:
    """Number of distinct ideas tried so far."""
    if not path.exists():
        return 0
    return int(pd.read_csv(path)["trial"].nunique())


def pass_rate(information_ratio: float, years: float, threshold_t: float) -> float:
    """Chance that a strategy's measured t-statistic clears a threshold over `years`."""
    return float(norm.sf(threshold_t - information_ratio * np.sqrt(years)))


def gate_pass_rate(information_ratio: float, gates: list[tuple[float, float]]) -> float:
    """Chance of clearing independent gates, each given as (years, threshold t)."""
    return float(np.prod([pass_rate(information_ratio, y, t) for y, t in gates]))


def family_false_positive(single_rate: float, trials: int) -> float:
    """Chance that at least one of `trials` skill-free ideas passes."""
    return float(1 - (1 - single_rate) ** trials)


def annual_cost(turnover: float, cost_per_side: float) -> float:
    """Yearly drag. `turnover` is one-way: buying and selling each happen on that share."""
    return 2 * turnover * cost_per_side


def tax_drag(gross_return: float, turnover: float, short_rate: float = 0.20,
             long_rate: float = 0.125) -> float:
    """Rough yearly tax on realized gains, compared with holding and deferring.

    Turnover sets the share of gains realized each year. Selling more than once a year means
    holdings are under 12 months old, so gains take the short-term rate.
    """
    realized = min(turnover, 1.0)
    rate = short_rate if turnover > 1.0 else long_rate
    return float(max(gross_return, 0.0) * realized * rate)
