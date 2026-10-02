import numpy as np
import pandas as pd
import pytest
from pdt.ab import build_report, per_cluster, select_target, undesirable_rate
from pdt.cluster import TAGS

FIELDS = [
    "scenario_id", "tag", "max_lateral_deviation_m", "mean_lateral_deviation_m",
    "final_position_gap_m", "min_ttc_rule_s", "min_ttc_ml_s", "ttc_delta_s",
    "max_jerk_rule", "max_jerk_ml", "jerk_delta", "mean_abs_accel_rule",
    "mean_abs_accel_ml", "decision_flip_count", "first_flip_time_s", "flip_kind",
    "completion_progress_rule_m", "completion_progress_ml_m", "hard_brake_rule",
    "hard_brake_ml", "collision_rule", "collision_ml", "ade_rule_vs_human_m",
    "ade_ml_vs_human_m", "divergence_score",
]


def make_frame(n=12, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "scenario_id": f"s{i:03d}",
            "tag": TAGS[i % len(TAGS)],
            "max_lateral_deviation_m": rng.uniform(0, 5),
            "mean_lateral_deviation_m": rng.uniform(0, 2),
            "final_position_gap_m": rng.uniform(0, 8),
            "min_ttc_rule_s": rng.uniform(0.2, 5),
            "min_ttc_ml_s": rng.uniform(0.2, 5),
            "ttc_delta_s": rng.uniform(-2, 2),
            "max_jerk_rule": rng.uniform(0, 30),
            "max_jerk_ml": rng.uniform(0, 30),
            "jerk_delta": rng.uniform(-10, 10),
            "mean_abs_accel_rule": rng.uniform(0, 1),
            "mean_abs_accel_ml": rng.uniform(0, 1),
            "decision_flip_count": int(rng.integers(0, 81)),
            "first_flip_time_s": 1.0,
            "flip_kind": "rule_FOLLOW_ml_ASSERT",
            "completion_progress_rule_m": rng.uniform(20, 80),
            "completion_progress_ml_m": rng.uniform(20, 80),
            "hard_brake_rule": bool(i % 5 == 0),
            "hard_brake_ml": bool(i % 7 == 0),
            "collision_rule": bool(i % 3 == 0),
            "collision_ml": bool(i % 4 == 0),
            "ade_rule_vs_human_m": rng.uniform(1, 10),
            "ade_ml_vs_human_m": rng.uniform(1, 10),
            "divergence_score": rng.uniform(0.4, 0.9),
        })
    return pd.DataFrame(rows, columns=FIELDS)


def test_undesirable_rate_counts_collision_or_hard_brake():
    df = make_frame(20)
    expected = float((df["collision_rule"] | df["hard_brake_rule"]).mean())
    assert undesirable_rate(df) == pytest.approx(expected)


def test_select_target_picks_desirable_with_best_ttc_improvement():
    before = {
        2: {"size": 47, "ttc_delta_mean_s": 1.01},
        3: {"size": 63, "ttc_delta_mean_s": 0.10},
        9: {"size": 21, "ttc_delta_mean_s": 0.90},
        5: {"size": 2, "ttc_delta_mean_s": 9.9},
    }
    labels = {"clusters": {
        2: {"label": "desirable"}, 3: {"label": "desirable"},
        9: {"label": "desirable"}, 5: {"label": "undesirable"},
    }}
    assert select_target(before, labels, min_size=10) == 2
    assert select_target({}, labels, min_size=10) is None
    assert select_target(before, {"clusters": {}}, min_size=10) is None
    assert select_target(before, labels, min_size=100) is None


def test_per_cluster_and_build_report():
    df = make_frame(24)
    labels = np.array([i % 3 for i in range(24)])
    per = per_cluster(df, labels)
    assert set(per) == {0, 1, 2}
    assert sum(c["size"] for c in per.values()) == 24

    events = pd.DataFrame({
        "name": ["early_braking"] * 6,
        "active": [True, True, True, True, False, False],
        "reason": ["activated"] * 4 + ["veto_jerk", "veto_pedestrian_buffer"],
    })
    lab = {"clusters": {0: {"label": "desirable"}, 1: {"label": "mixed"}, 2: {"label": "undesirable"}}}
    report = build_report(per, per, {"n_rows": 24}, {"n_rows": 24}, lab, 0, ["early_braking"], events)
    assert report["target_cluster"] == 0
    assert report["overrides"] == ["early_braking"]
    assert report["override_stats"]["early_braking"]["activations"] == 4
    assert report["override_stats"]["early_braking"]["vetoes"] == 2
    assert report["override_stats"]["early_braking"]["veto_reasons"] == {"veto_jerk": 1, "veto_pedestrian_buffer": 1}
    assert report["headline"]["undesirable_rate_before"] == report["headline"]["undesirable_rate_after"]
