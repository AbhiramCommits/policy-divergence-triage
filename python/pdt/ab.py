#!/usr/bin/env python3
"""A/B study (CLI: pdt-ab): port a desirable ML behavior into the rule planner
and measure it.

Runs the full shadow suite twice on the held-out split — baseline (overrides
OFF) and candidate (overrides ON) — assigns both runs through the *persisted*
cluster model, and emits artifacts/ab_report.md + ab_report.json.

Target cluster selection (documented, data-driven): among clusters labeled
`desirable` in labels/cluster_labels.yaml with at least `min_target_size`
scenarios, pick the one where the ML behavior removes the most rule collisions
(before-run collision_rule count - collision_ml count), tie-broken by size.

Per-cluster undesirable rate: fraction of scenarios in the cluster flagged
unsafe for the RULE planner (collision_rule OR hard_brake_rule).

Report contents:
- per-cluster undesirable rate before vs after
- the headline target-cluster undesirable rate drop
- a no-regression check across every other cluster
- global safety metrics: collision count, hard-brake count, p95 max jerk,
  mean progress, mean min TTC
- override activation count and veto count with veto reason breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pdt.cluster import assign
from pdt.review import load_labels

MIN_TARGET_SIZE = 10


def undesirable_rate(g: pd.DataFrame) -> float:
    if len(g) == 0:
        return 0.0
    return float((g["collision_rule"] | g["hard_brake_rule"]).mean())


def before_target_rate(per_before: dict[int, dict], target: int) -> float:
    return per_before[target]["undesirable_rate"]


def assign_through_model(model_path: Path, divergence: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    bundle = joblib.load(model_path)
    d, labels, _dists = assign(bundle, divergence)
    above = d["divergence_score"] > bundle["meta"]["threshold"]
    return d[above].reset_index(drop=True), labels[above]


def per_cluster(d: pd.DataFrame, labels: np.ndarray) -> dict[int, dict]:
    out = {}
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        g = d.iloc[idx]
        out[int(cid)] = {
            "size": int(len(g)),
            "undesirable": int((g["collision_rule"] | g["hard_brake_rule"]).sum()),
            "undesirable_rate": undesirable_rate(g),
            "collisions_rule": int(g["collision_rule"].sum()),
            "collisions_ml": int(g["collision_ml"].sum()),
        }
    return out


def global_metrics(d: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(d)),
        "collisions": int(d["collision_rule"].sum()),
        "hard_brakes": int(d["hard_brake_rule"].sum()),
        "jerk_p95": float(np.percentile(d["max_jerk_rule"], 95)),
        "mean_progress_m": float(d["completion_progress_rule_m"].mean()),
        "mean_min_ttc_s": float(d["min_ttc_rule_s"].mean()),
    }


def select_target(before: dict[int, dict], labels: dict, min_size: int = MIN_TARGET_SIZE) -> int | None:
    candidates = []
    for cid, entry in labels.get("clusters", {}).items():
        if entry.get("label") != "desirable":
            continue
        b = before.get(cid)
        if b is None or b["size"] < min_size:
            continue
        candidates.append((b["collisions_rule"] - b["collisions_ml"], -b["size"], cid))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][2])


def build_report(before: dict[int, dict], after: dict[int, dict], glob_before: dict, glob_after: dict,
                 labels: dict, target: int, override_name: str, events: pd.DataFrame) -> dict:
    no_regression = []
    for cid, entry in labels.get("clusters", {}).items():
        if cid == target:
            continue
        b = before.get(cid)
        a = after.get(cid)
        if b is None or a is None or b["size"] == 0 or a["size"] == 0:
            continue
        delta = a["undesirable_rate"] - b["undesirable_rate"]
        no_regression.append({
            "cluster_id": cid,
            "label": entry.get("label"),
            "size_before": b["size"],
            "size_after": a["size"],
            "undesirable_rate_before": b["undesirable_rate"],
            "undesirable_rate_after": a["undesirable_rate"],
            "delta": delta,
        })
    no_regression.sort(key=lambda r: -abs(r["delta"]))

    veto_reasons = {}
    if len(events):
        ev = events[events["name"] == override_name]
        veto_reasons = {k: int(v) for k, v in ev[~ev["active"]]["reason"].value_counts().items()}
    return {
        "target_cluster": target,
        "override": override_name,
        "before": {"clusters": before, "global": glob_before},
        "after": {"clusters": after, "global": glob_after},
        "headline": {
            "undesirable_rate_before": before[target]["undesirable_rate"],
            "undesirable_rate_after": after[target]["undesirable_rate"],
            "drop_pp": before[target]["undesirable_rate"] - after[target]["undesirable_rate"],
            "collisions_before": before[target]["collisions_rule"],
            "collisions_after": after[target]["collisions_rule"],
        },
        "override_stats": {
            "activations": int((events["active"] & (events["name"] == override_name)).sum()) if len(events) else 0,
            "vetoes": int((~events["active"] & (events["name"] == override_name)).sum()) if len(events) else 0,
            "veto_reasons": veto_reasons,
        },
        "no_regression": no_regression,
    }


def render_md(report: dict) -> str:
    h = report["headline"]
    lines = [
        "# A/B report: `{}` override".format(report["override"]),
        "",
        f"Target cluster: **{report['target_cluster']}** (desirable).",
        "",
        "## Headline",
        "",
        f"- target-cluster undesirable rate: {h['undesirable_rate_before']:.3f} -> {h['undesirable_rate_after']:.3f} "
        f"(drop of **{h['drop_pp'] * 100:.1f} pp** over the scenarios still assigned to the cluster)",
        f"- fixed-population rate (before-cluster scenario ids): {h['undesirable_rate_before']:.3f} -> "
        f"{h['fixed_population_undesirable_rate_after']:.3f} (drop of **{h['drop_pp_fixed_population'] * 100:.1f} pp**)",
        f"- target-cluster rule collisions: {h['collisions_before']} -> {h['collisions_after']} "
        f"(fixed population: {h['fixed_population_collisions_after']})",
        "",
        "## Global safety metrics (rule planner, held-out split)",
        "",
        "| metric | before | after |",
        "|---|---|---|",
    ]
    b = report["before"]["global"]
    a = report["after"]["global"]
    lines += [
        f"| collision count | {b['collisions']} | {a['collisions']} |",
        f"| hard-brake count | {b['hard_brakes']} | {a['hard_brakes']} |",
        f"| p95 max jerk (m/s^3) | {b['jerk_p95']:.2f} | {a['jerk_p95']:.2f} |",
        f"| mean progress (m) | {b['mean_progress_m']:.2f} | {a['mean_progress_m']:.2f} |",
        f"| mean min TTC (s) | {b['mean_min_ttc_s']:.2f} | {a['mean_min_ttc_s']:.2f} |",
        "",
        "## Per-cluster undesirable rate",
        "",
        "| cluster | label | size before/after | rate before | rate after | delta |",
        "|---|---|---|---|---|---|",
    ]
    labels = {cid: e.get("label") for cid, e in report_labels_map(report).items()}
    all_ids = sorted(set(report["before"]["clusters"]) | set(report["after"]["clusters"]))
    for cid in all_ids:
        bb = report["before"]["clusters"].get(cid, {"size": 0, "undesirable_rate": 0.0})
        aa = report["after"]["clusters"].get(cid, {"size": 0, "undesirable_rate": 0.0})
        mark = " *target*" if cid == report["target_cluster"] else ""
        lines.append(
            f"| {cid}{mark} | {labels.get(cid, '?')} | {bb['size']}/{aa['size']} | "
            f"{bb['undesirable_rate']:.3f} | {aa['undesirable_rate']:.3f} | {aa['undesirable_rate'] - bb['undesirable_rate']:+.3f} |"
        )
    lines += [
        "",
        "## Override stats",
        "",
        f"- activations: {report['override_stats']['activations']}",
        f"- vetoes: {report['override_stats']['vetoes']}",
        f"- veto reasons: {report['override_stats']['veto_reasons']}",
        "",
        "## No-regression check (non-target clusters)",
        "",
        "| cluster | label | rate before | rate after | delta |",
        "|---|---|---|---|---|",
    ]
    for r in report["no_regression"]:
        lines.append(f"| {r['cluster_id']} | {r['label']} | {r['undesirable_rate_before']:.3f} | "
                     f"{r['undesirable_rate_after']:.3f} | {r['delta']:+.3f} |")
    return "\n".join(lines) + "\n"


def report_labels_map(report: dict) -> dict:
    return {cid: {"label": report["cluster_labels"].get(cid)} for cid in report["cluster_labels"]}


def run_shadow_subprocess(overrides_json: str | None, out_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "pdt.harness",
        "--scenarios", str(args.scenarios),
        "--checkpoint", str(args.checkpoint),
        "--out", str(out_dir / "trajectories.parquet"),
        "--metrics-out", str(out_dir / "divergence.parquet"),
        "--events-out", str(out_dir / "override_events.parquet"),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--split", "held_out",
    ]
    if overrides_json:
        cmd += ["--overrides", overrides_json]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"harness run failed with exit code {proc.returncode}")


def run_ab(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    a_dir = out_dir / "ab" / "baseline"
    b_dir = out_dir / "ab" / "candidate"

    labels = load_labels(args.labels)

    if not args.report_only:
        print("=== run A: baseline (overrides OFF) ===")
        run_shadow_subprocess(None, a_dir, args)
        print("=== run B: candidate (override ON) ===")
        run_shadow_subprocess(json.dumps({args.override: True}), b_dir, args)

    div_before = pd.read_parquet(a_dir / "divergence.parquet")
    div_after = pd.read_parquet(b_dir / "divergence.parquet")
    d_before, labels_before = assign_through_model(Path(args.model), div_before)
    d_after, labels_after = assign_through_model(Path(args.model), div_after)

    per_before = per_cluster(d_before, labels_before)
    per_after = per_cluster(d_after, labels_after)

    target = select_target(per_before, labels, min_size=args.min_target_size)
    if target is None:
        raise SystemExit("no desirable cluster with enough scenarios found in the baseline run")

    target_ids = set(d_before[labels_before == target]["scenario_id"])
    target_after = d_after[d_after["scenario_id"].isin(target_ids)]
    fixed_rate_after = undesirable_rate(target_after)
    fixed_collisions_after = int(target_after["collision_rule"].sum())

    events = pd.read_parquet(b_dir / "override_events.parquet")
    report = build_report(per_before, per_after, global_metrics(d_before), global_metrics(d_after),
                          labels, target, args.override, events)
    report["cluster_labels"] = {cid: (labels["clusters"].get(cid) or {}).get("label") for cid in
                                set(per_before) | set(per_after)}
    report["headline"]["fixed_population_undesirable_rate_after"] = fixed_rate_after
    report["headline"]["fixed_population_collisions_after"] = fixed_collisions_after
    report["headline"]["drop_pp_fixed_population"] = (
        before_target_rate(per_before, target) - fixed_rate_after
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "ab_report.md").write_text(render_md(report))
    print(f"wrote {out_dir / 'ab_report.md'} and {out_dir / 'ab_report.json'}")
    h = report["headline"]
    print(f"headline: target cluster {target} undesirable rate {h['undesirable_rate_before']:.3f} -> "
          f"{h['fixed_population_undesirable_rate_after']:.3f} fixed-population "
          f"(drop {h['drop_pp_fixed_population'] * 100:.1f} pp)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios/logs.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/policy.pt"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/cluster_model.joblib"))
    parser.add_argument("--labels", type=Path, default=Path("labels/cluster_labels.yaml"))
    parser.add_argument("--override", type=str, default="early_braking")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-target-size", type=int, default=MIN_TARGET_SIZE)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-only", action="store_true",
                        help="skip the harness runs and rebuild the report from artifacts/ab/")
    args = parser.parse_args()
    run_ab(args)


if __name__ == "__main__":
    main()
