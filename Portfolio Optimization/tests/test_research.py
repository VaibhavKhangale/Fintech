"""Tests for research safeguards."""
import numpy as np
import pandas as pd
import pytest

from pmo import research

SPLITS = {"dev": {"start": "2020-01-01", "end": "2020-12-31"},
          "val": {"start": "2021-01-01", "end": "2021-12-31", "protected": True}}
FRAME = pd.DataFrame({"A": range(24)}, index=pd.date_range("2020-01-31", periods=24, freq="ME"))


def test_window_returns_only_the_split(tmp_path):
    out = research.window(FRAME, "dev", splits=SPLITS, unlocks_path=tmp_path / "none.yaml")
    assert len(out) == 12
    assert out.index.max() == pd.Timestamp("2020-12-31")


def test_protected_split_is_refused_without_approval(tmp_path):
    with pytest.raises(PermissionError):
        research.window(FRAME, "val", "signal-a", SPLITS, tmp_path / "none.yaml")


def test_protected_split_opens_only_for_the_approved_candidate(tmp_path):
    path = tmp_path / "unlocks.yaml"
    path.write_text("signal-a: {approved_on: 2027-01-01}\n", encoding="utf-8")
    assert len(research.window(FRAME, "val", "signal-a", SPLITS, path)) == 12
    with pytest.raises(PermissionError):
        research.window(FRAME, "val", "signal-b", SPLITS, path)


def test_project_splits_do_not_overlap_and_protect_later_periods():
    splits = research.load_splits()
    assert splits["development"]["end"] < splits["validation"]["start"]
    assert splits["validation"]["end"] < splits["seen_period"]["start"]
    assert splits["validation"]["protected"] and splits["seen_period"]["protected"]
    assert not splits["development"].get("protected")


def test_hurdle_rises_with_the_number_of_trials():
    assert research.hurdle_t(1) == pytest.approx(1.96, abs=0.01)
    assert research.hurdle_t(20) == pytest.approx(3.02, abs=0.01)
    with pytest.raises(ValueError):
        research.hurdle_t(0)


def test_log_counts_distinct_trials(tmp_path):
    path = tmp_path / "log.csv"
    assert research.count_trials(path) == 0
    research.log_trial("13", "momentum", "development", "t", 2.1, path=path)
    research.log_trial("13", "momentum", "development", "ic", 0.03, path=path)
    research.log_trial("13", "value", "development", "t", 0.4, path=path)
    assert research.count_trials(path) == 2


def test_pass_rate_at_the_mean_is_one_half():
    assert research.pass_rate(0.5, 4, 1.0) == pytest.approx(0.5)


def test_gate_pass_rate_multiplies_independent_gates():
    single = research.pass_rate(0.5, 4, 1.0)
    assert research.gate_pass_rate(0.5, [(4, 1.0), (4, 1.0)]) == pytest.approx(single ** 2)


def test_family_false_positive():
    assert research.family_false_positive(0.05, 1) == pytest.approx(0.05)
    assert research.family_false_positive(0.05, 20) == pytest.approx(1 - 0.95 ** 20)


def test_annual_cost_counts_both_sides():
    assert research.annual_cost(1.0, 0.0025) == pytest.approx(0.005)


def test_tax_drag_switches_rate_above_full_turnover():
    assert research.tax_drag(0.15, 0.5) == pytest.approx(0.15 * 0.5 * 0.125)
    assert research.tax_drag(0.15, 2.0) == pytest.approx(0.15 * 0.20)
    assert research.tax_drag(-0.1, 2.0) == 0
    assert np.isfinite(research.tax_drag(0.0, 0.0))
