import json

import pytest

from pdt.gate import check, load_config


def deep_merge(base, patch):
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def base_report(**overrides):
    report = {
        "target_cluster": 2,
        "override": "early_braking",
        "before": {
            "global": {"collisions": 100, "hard_brakes": 0, "jerk_p95": 20.0, "mean_progress_m": 55.0, "mean_min_ttc_s": 1.0},
            "clusters": {
                2: {"size": 47, "undesirable_rate": 0.340, "collisions_rule": 16, "collisions_ml": 12},
                1: {"size": 149, "undesirable_rate": 0.10, "collisions_rule": 5, "collisions_ml": 6},
                9: {"size": 21, "undesirable_rate": 0.095, "collisions_rule": 2, "collisions_ml": 0},
            },
        },
        "after": {
            "global": {"collisions": 95, "hard_brakes": 0, "jerk_p95": 19.0, "mean_progress_m": 56.0, "mean_min_ttc_s": 1.2},
            "clusters": {
                2: {"size": 50, "undesirable_rate": 0.200, "collisions_rule": 10, "collisions_ml": 12},
                1: {"size": 148, "undesirable_rate": 0.09, "collisions_rule": 5, "collisions_ml": 6},
                9: {"size": 18, "undesirable_rate": 0.05, "collisions_rule": 1, "collisions_ml": 0},
            },
        },
        "no_regression": [],
        "override_stats": {"activations": 100, "vetoes": 10, "veto_reasons": {}},
        "headline": {"undesirable_rate_before": 0.34, "undesirable_rate_after": 0.20, "drop_pp": 0.14},
    }
    for key, value in overrides.items():
        if isinstance(value, dict):
            report[key] = deep_merge(report[key], value)
        else:
            report[key] = value
    return report


CFG = {
    "hard_brake_increase_tolerance": 2,
    "jerk_p95_rel_tolerance": 0.05,
    "non_target_undesirable_rate_rise_tolerance": 0.03,
    "non_target_cluster_min_size": 5,
    "new_cluster_min_size": 5,
}


def test_gate_passes_on_clean_report():
    ok, rows = check(base_report(), CFG)
    assert ok is True
    assert all(r["pass"] for r in rows)


def test_gate_fails_on_collision_increase():
    report = base_report(after={"global": {"collisions": 101}})
    ok, rows = check(report, CFG)
    assert ok is False
    failed = [r["check"] for r in rows if not r["pass"]]
    assert "collision count" in failed


def test_gate_fails_on_hard_brake_increase_beyond_tolerance():
    report = base_report(after={"global": {"hard_brakes": 3}})
    ok, rows = check(report, CFG)
    assert ok is False
    assert "hard-brake count" in [r["check"] for r in rows if not r["pass"]]

    report = base_report(after={"global": {"hard_brakes": 2}})
    ok, _ = check(report, CFG)
    assert ok is True


def test_gate_fails_on_jerk_regression():
    report = base_report(after={"global": {"jerk_p95": 23.0}})
    ok, rows = check(report, CFG)
    assert ok is False
    assert any("jerk" in r["check"] for r in rows if not r["pass"])


def test_gate_fails_on_non_target_cluster_rise():
    report = base_report(after={"clusters": {"1": {"size": 148, "undesirable_rate": 0.16}}})
    ok, rows = check(report, CFG)
    assert ok is False
    assert any("cluster 1" in r["check"] for r in rows if not r["pass"])


def test_gate_fails_on_new_cluster():
    report = base_report()
    report["after"]["clusters"][7] = {"size": 9, "undesirable_rate": 0.5}
    ok, rows = check(report, CFG)
    assert ok is False
    assert any("new cluster 7" in r["check"] for r in rows if not r["pass"])


def test_gate_config_round_trip(tmp_path):
    p = tmp_path / "gate_config.yaml"
    p.write_text("thresholds:\n  hard_brake_increase_tolerance: 4\n")
    assert load_config(p) == {"hard_brake_increase_tolerance": 4}
