"""Unit tests for CLI plumbing: query, harness helpers, review rendering, ab
rendering, and metric edge cases."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pdt_core

from pdt.ab import assign_through_model, render_md
from pdt.harness import line_offsets, scenario_from_dict, state_from_dict
from pdt.metrics import compute_metrics_row
from pdt.query import CANNED_QUERIES, connect
from pdt.review import render_exemplar

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scenarios.jsonl"


def test_query_connect_and_canned_queries(tmp_path):
    df = pd.DataFrame({
        "scenario_id": ["a", "b"], "tag": ["other", "other"],
        "divergence_score": [0.9, 0.1], "decision_flip_count": [3, 1],
        "flip_kind": ["rule_YIELD_ml_ASSERT", ""],
        "hard_brake_rule": [False, False], "hard_brake_ml": [True, False],
        "collision_rule": [False, False], "collision_ml": [False, False],
    })
    p = tmp_path / "divergence.parquet"
    df.to_parquet(p)
    con = connect(str(p))
    for name in ("top50", "per_tag", "flips", "hard_brake"):
        out = con.execute(CANNED_QUERIES[name]).fetchdf()
        assert len(out) >= 0
    top = con.execute(CANNED_QUERIES["top50"]).fetchdf()
    assert top.iloc[0]["scenario_id"] == "a"


def test_harness_state_and_scenario_roundtrip():
    d = json.loads(open(FIXTURES).readline())
    sc = scenario_from_dict(d)
    assert sc.id == d["id"]
    assert sc.tag == d["tag"]
    assert len(sc.agents) == len(d["agents"])
    assert len(sc.logged_ego) == len(d["logged_ego"])
    assert len(sc.centerline) == len(d["centerline"])
    s = state_from_dict(d["ego_init"])
    assert (s.t, s.x, s.y, s.heading, s.v, s.a) == pytest.approx(
        (d["ego_init"]["t"], d["ego_init"]["x"], d["ego_init"]["y"],
         d["ego_init"]["heading"], d["ego_init"]["v"], d["ego_init"]["a"])
    )


def test_harness_line_offsets_split():
    offsets_all = line_offsets(FIXTURES, "all")
    assert len(offsets_all) == 20
    offsets_held = line_offsets(FIXTURES, "held_out")
    offsets_train = line_offsets(FIXTURES, "train")
    assert len(offsets_held) + len(offsets_train) == 20
    assert len(offsets_held) > 0 and len(offsets_train) > 0


def test_review_render_exemplar_contains_both_grids():
    d = json.loads(open(FIXTURES).readline())
    sc = scenario_from_dict(d)
    traj = pdt_core.RulePlanner().plan(sc)
    n = len(traj.states)
    df = pd.DataFrame({
        "scenario_id": [d["id"]] * (2 * n),
        "source": ["rule"] * n + ["ml"] * n,
        "step": list(range(n)) * 2,
        "t": [s.t for s in traj.states] * 2,
        "x": [s.x for s in traj.states] * 2,
        "y": [s.y for s in traj.states] * 2,
        "heading": [s.heading for s in traj.states] * 2,
        "v": [s.v for s in traj.states] * 2,
        "a": [s.a for s in traj.states] * 2,
        "decision": [0] * (2 * n), "decision_name": ["FOLLOW"] * (2 * n),
    })
    div_row = pd.DataFrame({"scenario_id": [d["id"]], "divergence_score": [0.7],
                            "min_ttc_rule_s": [1.0], "min_ttc_ml_s": [1.2],
                            "hard_brake_rule": [False], "hard_brake_ml": [False],
                            "collision_rule": [False], "collision_ml": [False],
                            "completion_progress_rule_m": [50.0], "completion_progress_ml_m": [55.0]})
    out = render_exemplar(d["id"], df, div_row)
    assert "x-y path" in out
    assert "speed profile" in out
    assert "|" in out


def test_metrics_no_centerline_and_no_human_fallbacks():
    scenario = {
        "id": "t", "tag": "other",
        "ego_init": {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "v": 10.0, "a": 0.0},
        "agents": [], "centerline": [], "speed_limit": 11.18, "logged_ego": [],
    }
    states = np.zeros((81, 6))
    for i in range(81):
        states[i] = [i * 0.1, i, 0.0, 0.0, 10.0, 0.0]
    pair = {"states": states, "decisions": [0] * 81}
    row = compute_metrics_row(scenario, pair, {"states": states.copy(), "decisions": [0] * 81})
    assert row["completion_progress_rule_m"] == pytest.approx(80.0)
    assert np.isnan(row["ade_rule_vs_human_m"])
    assert np.isnan(row["ttc_delta_s"])
    assert row["divergence_score"] == 0.0


def test_metrics_human_ade_aligned():
    scenario = {
        "id": "t", "tag": "other",
        "ego_init": {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "v": 10.0, "a": 0.0},
        "agents": [], "centerline": [[x, 0.0] for x in range(0, 61, 5)], "speed_limit": 11.18,
        "logged_ego": [],
    }
    human = np.zeros((81, 6))
    for i in range(81):
        human[i] = [i * 0.1, i, 0.0, 0.0, 10.0, 0.0]
    scenario["logged_ego"] = human.tolist()
    states = human.copy()
    row = compute_metrics_row(scenario, {"states": states, "decisions": [0] * 81},
                              {"states": states.copy(), "decisions": [0] * 81})
    assert row["ade_rule_vs_human_m"] == pytest.approx(0.0)
    assert row["ade_ml_vs_human_m"] == pytest.approx(0.0)


def test_ab_render_md(tmp_path):
    report = {
        "override": "early_braking",
        "target_cluster": 2,
        "before": {"global": {"collisions": 5, "hard_brakes": 0, "jerk_p95": 30.0,
                              "mean_progress_m": 50.0, "mean_min_ttc_s": 1.0},
                   "clusters": {2: {"size": 8, "undesirable_rate": 0.5}}},
        "after": {"global": {"collisions": 2, "hard_brakes": 0, "jerk_p95": 28.0,
                             "mean_progress_m": 51.0, "mean_min_ttc_s": 1.1},
                  "clusters": {2: {"size": 8, "undesirable_rate": 0.25}}},
        "headline": {"undesirable_rate_before": 0.5, "undesirable_rate_after": 0.25,
                     "drop_pp": 0.25, "collisions_before": 5, "collisions_after": 2,
                     "fixed_population_undesirable_rate_after": 0.25,
                     "fixed_population_collisions_after": 2, "drop_pp_fixed_population": 0.25},
        "override_stats": {"activations": 3, "vetoes": 1, "veto_reasons": {"veto_jerk": 1}},
        "no_regression": [],
        "cluster_labels": {2: "desirable"},
    }
    md = render_md(report)
    assert "# A/B report" in md
    assert "25.0 pp" in md
    assert "early_braking" in md
