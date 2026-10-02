"""End-to-end test: the 20-scenario fixture subset through the whole pipeline
(train -> shadow -> metrics -> cluster -> labels -> ab -> gate) asserting the
gate passes. Uses the committed tests/fixtures/scenarios.jsonl (no AV2)."""

import argparse
from pathlib import Path

import pytest
from pdt.ab import build_report_from_runs
from pdt.cluster import run_clustering
from pdt.gate import check
from pdt.harness import run as run_shadow
from pdt.review import save_labels
from pdt.train_policy import run as run_train

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scenarios.jsonl"
GATE_CFG = {
    "hard_brake_increase_tolerance": 2,
    "jerk_p95_rel_tolerance": 0.05,
    "non_target_undesirable_rate_rise_tolerance": 0.03,
    "non_target_cluster_min_size": 5,
    "new_cluster_min_size": 5,
}


def harness_args(out_dir: Path, scenarios: Path, checkpoint: Path, overrides: str | None):
    return argparse.Namespace(
        scenarios=scenarios, checkpoint=checkpoint,
        out=out_dir / "trajectories.parquet", metrics_out=out_dir / "divergence.parquet",
        events_out=out_dir / "override_events.parquet", overrides=overrides,
        workers=4, seed=0, split="all", limit=None,
    )


@pytest.mark.slow
def test_fixture_pipeline_end_to_end_gate_passes(tmp_path):
    pytest.importorskip("torch")

    checkpoint = tmp_path / "policy.pt"
    run_train(argparse.Namespace(
        scenarios=FIXTURES, out=checkpoint, log=tmp_path / "train_log.jsonl",
        epochs=2, batch_size=64, lr=1e-3, seed=0,
    ))
    assert checkpoint.exists()

    base_dir = tmp_path / "run_base"
    cand_dir = tmp_path / "run_cand"
    run_shadow(harness_args(base_dir, FIXTURES, checkpoint, None))
    run_shadow(harness_args(cand_dir, FIXTURES, checkpoint, '{"early_braking": true, "intersection_caution": true}'))

    assert (base_dir / "divergence.parquet").exists()
    assert (cand_dir / "divergence.parquet").exists()
    assert (cand_dir / "override_events.parquet").exists()

    summary = run_clustering(base_dir / "divergence.parquet", tmp_path / "clusters",
                             threshold=0.0, seed=0, refit=True, quiet=True)
    cluster_ids = [c["cluster_id"] for c in summary["clusters"]]
    assert cluster_ids

    labels = {"version": 1, "clusters": {
        cid: {"label": "desirable", "rationale": "fixture e2e", "ported_rule_hint": None}
        for cid in cluster_ids
    }}
    labels_path = tmp_path / "labels.yaml"
    save_labels(labels_path, labels)

    report = build_report_from_runs(base_dir, cand_dir, tmp_path / "clusters" / "cluster_model.joblib",
                                    {"clusters": labels["clusters"]}, ["early_braking", "intersection_caution"],
                                    min_target_size=1)
    assert report["target_cluster"] in cluster_ids
    assert sum(st["activations"] for st in report["override_stats"].values()) > 0

    ok, rows = check(report, GATE_CFG)
    failed = [r["check"] for r in rows if not r["pass"]]
    assert ok, f"gate failed: {failed}"
    assert report["before"]["global"]["collisions"] >= report["after"]["global"]["collisions"]
