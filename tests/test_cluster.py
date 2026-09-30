import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from pdt.cluster import TAGS, build_features, run_clustering
from pdt.review import generate_cluster_png, load_labels, save_labels

FIELDS = [
    "scenario_id", "tag", "max_lateral_deviation_m", "mean_lateral_deviation_m",
    "final_position_gap_m", "min_ttc_rule_s", "min_ttc_ml_s", "ttc_delta_s",
    "max_jerk_rule", "max_jerk_ml", "jerk_delta", "mean_abs_accel_rule",
    "mean_abs_accel_ml", "decision_flip_count", "first_flip_time_s", "flip_kind",
    "completion_progress_rule_m", "completion_progress_ml_m", "hard_brake_rule",
    "hard_brake_ml", "collision_rule", "collision_ml", "ade_rule_vs_human_m",
    "ade_ml_vs_human_m", "divergence_score",
]


def make_divergence_df(n: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        blob = i % 3
        if blob == 0:
            lat = rng.uniform(4.0, 9.0)
            ttc = rng.uniform(-3.0, -1.0)
            flip = "rule_YIELD_ml_ASSERT"
            prog = rng.uniform(2.0, 8.0)
            jerk = rng.uniform(-1.0, 1.0)
        elif blob == 1:
            lat = rng.uniform(0.2, 1.5)
            ttc = rng.uniform(0.5, 2.0)
            flip = "rule_FOLLOW_ml_STOP"
            prog = rng.uniform(-6.0, -1.0)
            jerk = rng.uniform(2.0, 5.0)
        else:
            lat = rng.uniform(1.5, 4.0)
            ttc = float("nan")
            flip = ""
            prog = rng.uniform(-1.0, 1.0)
            jerk = rng.uniform(-5.0, -2.0)
        rows.append({
            "scenario_id": f"synth_{i:04d}",
            "tag": TAGS[i % len(TAGS)],
            "max_lateral_deviation_m": lat,
            "mean_lateral_deviation_m": lat * 0.4,
            "final_position_gap_m": lat * 1.6,
            "min_ttc_rule_s": 5.0,
            "min_ttc_ml_s": 5.0 + ttc,
            "ttc_delta_s": ttc,
            "max_jerk_rule": 3.0,
            "max_jerk_ml": 3.0 + jerk,
            "jerk_delta": jerk,
            "mean_abs_accel_rule": 0.4,
            "mean_abs_accel_ml": 0.4 + (0.3 if blob == 0 else -0.2),
            "decision_flip_count": int(rng.integers(0, 81)),
            "first_flip_time_s": 1.0,
            "flip_kind": flip,
            "completion_progress_rule_m": 60.0,
            "completion_progress_ml_m": 60.0 + prog,
            "hard_brake_rule": False,
            "hard_brake_ml": bool(blob == 1),
            "collision_rule": False,
            "collision_ml": bool(blob == 1),
            "ade_rule_vs_human_m": 1.0,
            "ade_ml_vs_human_m": 1.0 + (0.8 if blob == 0 else 0.2),
            "divergence_score": rng.uniform(0.55, 0.95),
        })
    return pd.DataFrame(rows, columns=FIELDS)


def write_divergence(path, df):
    df.to_parquet(path)


def test_clustering_reproducible_under_fixed_seed(tmp_path):
    df = make_divergence_df()
    src = tmp_path / "divergence.parquet"
    write_divergence(src, df)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    s1 = run_clustering(src, out1, threshold=0.5, seed=3, refit=True, quiet=True)
    s2 = run_clustering(src, out2, threshold=0.5, seed=3, refit=True, quiet=True)

    c1 = pd.read_parquet(out1 / "clusters.parquet")
    c2 = pd.read_parquet(out2 / "clusters.parquet")
    pd.testing.assert_frame_equal(c1, c2)
    assert s1 == s2
    assert s1["assigned_with_existing_model"] is False
    assert len(s1["clusters"]) >= 3
    total = sum(c["size"] for c in s1["clusters"])
    assert total == (df.divergence_score > 0.5).sum()


def test_existing_model_path_does_not_refit(tmp_path):
    df = make_divergence_df()
    src = tmp_path / "divergence.parquet"
    write_divergence(src, df)

    out = tmp_path / "out"
    run_clustering(src, out, threshold=0.5, seed=5, refit=True, quiet=True)
    model_path = out / "cluster_model.joblib"
    h1 = joblib.hash(joblib.load(model_path))
    c1 = pd.read_parquet(out / "clusters.parquet")

    s2 = run_clustering(src, out, threshold=0.5, seed=999, refit=False, quiet=True)
    h2 = joblib.hash(joblib.load(model_path))
    c2 = pd.read_parquet(out / "clusters.parquet")

    assert h1 == h2
    assert s2["assigned_with_existing_model"] is True
    pd.testing.assert_frame_equal(c1, c2)
    assert s2["seed"] == 5


def test_label_file_round_trip(tmp_path):
    labels = {
        "version": 1,
        "clusters": {
            3: {"label": "desirable", "rationale": "ML clears the intersection sooner", "ported_rule_hint": None},
            0: {"label": "undesirable", "rationale": "ML hard brakes", "ported_rule_hint": "port IDM gap"},
        },
    }
    path = tmp_path / "labels.yaml"
    save_labels(path, labels)
    loaded = load_labels(path)
    assert loaded == labels
    text = path.read_text()
    assert text.startswith("#")


def test_review_non_interactive_generates_plots(tmp_path):
    df = make_divergence_df(n=24)
    div_path = tmp_path / "divergence.parquet"
    write_divergence(div_path, df)
    out = tmp_path / "out"
    summary = run_clustering(div_path, out, threshold=0.5, seed=11, refit=True, quiet=True)

    n = 24
    traj_rows = []
    for i in range(n):
        sid = f"synth_{i:04d}"
        for step in range(40):
            for src in ("rule", "ml"):
                traj_rows.append({
                    "scenario_id": sid, "source": src, "step": step, "t": step * 0.1,
                    "x": step * 0.5, "y": (0.0 if src == "rule" else step * 0.02),
                    "heading": 0.0, "v": 8.0, "a": 0.0, "decision": 0, "decision_name": "FOLLOW",
                })
    traj_path = tmp_path / "trajectories.parquet"
    pd.DataFrame(traj_rows).to_parquet(traj_path)

    labels = {
        "version": 1,
        "clusters": {
            c["cluster_id"]: {"label": "desirable", "rationale": "ci auto-label", "ported_rule_hint": None}
            for c in summary["clusters"]
        },
    }
    labels_path = tmp_path / "labels_from.yaml"
    save_labels(labels_path, labels)

    plots_dir = tmp_path / "plots"
    for cluster in summary["clusters"]:
        p = generate_cluster_png(cluster, pd.read_parquet(traj_path), df, plots_dir)
        assert p.exists()

    from pdt.review import run as review_run
    import argparse

    args = argparse.Namespace(
        clusters=str(out / "clusters.parquet"),
        summary=str(out / "cluster_summary.json"),
        trajectories=str(traj_path),
        divergence=str(div_path),
        labels=str(tmp_path / "labels_main.yaml"),
        labels_from=str(labels_path),
        plots_dir=str(plots_dir),
        non_interactive=True,
    )
    rc = review_run(args)
    assert rc == 0
    for cluster in summary["clusters"]:
        assert (plots_dir / f"cluster_{cluster['cluster_id']:02d}.png").exists()
