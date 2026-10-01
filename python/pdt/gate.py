#!/usr/bin/env python3
"""Regression gate (CLI: pdt-gate): block merges when the override regresses
safety. Consumes artifacts/ab_report.json and gate_config.yaml.

Fails (non-zero exit) when any of:
- collision count increased (any increase fails)
- hard-brake count increased more than hard_brake_increase_tolerance
- p95 max jerk regressed beyond jerk_p95_rel_tolerance (relative)
- any non-target cluster's undesirable rate rose beyond
  non_target_undesirable_rate_rise_tolerance (clusters present in both runs
  with size >= non_target_cluster_min_size)
- a new cluster appeared with size above new_cluster_min_size
  (unmodeled behavior)
Prints a pass/fail table naming which check failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("gate_config.yaml")


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return raw.get("thresholds", {})


def check(report: dict, cfg: dict) -> tuple[bool, list[dict]]:
    before_g = report["before"]["global"]
    after_g = report["after"]["global"]
    before_c = report["before"]["clusters"]
    after_c = report["after"]["clusters"]
    target = report["target_cluster"]

    rows = []

    def row(name: str, before, after, threshold: str, ok: bool) -> None:
        rows.append({"check": name, "before": before, "after": after, "threshold": threshold, "pass": bool(ok)})

    tol_hard = float(cfg.get("hard_brake_increase_tolerance", 2))
    row("collision count", before_g["collisions"], after_g["collisions"], "no increase",
        after_g["collisions"] <= before_g["collisions"])
    row("hard-brake count", before_g["hard_brakes"], after_g["hard_brakes"],
        f"+{tol_hard:g} tolerance", after_g["hard_brakes"] <= before_g["hard_brakes"] + tol_hard)
    tol_jerk = float(cfg.get("jerk_p95_rel_tolerance", 0.05))
    row("p95 max jerk (m/s^3)", round(before_g["jerk_p95"], 2), round(after_g["jerk_p95"], 2),
        f"+{tol_jerk * 100:.1f}% tolerance", after_g["jerk_p95"] <= before_g["jerk_p95"] * (1 + tol_jerk))

    tol_rate = float(cfg.get("non_target_undesirable_rate_rise_tolerance", 0.03))
    min_size = int(cfg.get("non_target_cluster_min_size", 5))
    for cid in sorted(set(before_c) & set(after_c)):
        if cid == target:
            continue
        b = before_c[cid]
        a = after_c[cid]
        if b["size"] < min_size or a["size"] < min_size:
            continue
        delta = a["undesirable_rate"] - b["undesirable_rate"]
        row(f"cluster {cid} undesirable rate", round(b["undesirable_rate"], 4), round(a["undesirable_rate"], 4),
            f"+{tol_rate:g} tolerance", delta <= tol_rate)

    new_min = int(cfg.get("new_cluster_min_size", 5))
    new_ids = sorted(set(after_c) - set(before_c))
    for cid in new_ids:
        row(f"new cluster {cid} appeared", 0, after_c[cid]["size"],
            f"size < {new_min}", after_c[cid]["size"] < new_min)

    ok = all(r["pass"] for r in rows)
    return ok, rows


def print_table(rows: list[dict]) -> None:
    print(f"{'check':42s} {'before':>10s} {'after':>10s} {'threshold':>18s}  result")
    for r in rows:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"{r['check']:42s} {str(r['before']):>10s} {str(r['after']):>10s} {r['threshold']:>18s}  {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("artifacts/ab_report.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    cfg = load_config(args.config)
    ok, rows = check(report, cfg)
    print_table(rows)
    failed = [r["check"] for r in rows if not r["pass"]]
    if ok:
        print("\nGATE PASS")
    else:
        print(f"\nGATE FAIL: {failed}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
