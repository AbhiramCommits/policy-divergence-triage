#!/usr/bin/env python3
"""Shadow-mode harness (CLI: pdt-shadow).

Replays each scenario through the rule planner (pdt_core, pybind11) and the
learned ML policy on identical initial conditions, with other agents replaying
their logged tracks (non-reactive) for both systems. Scenarios are processed
in parallel (multiprocessing) with a fixed chunk order, so output row order is
stable across runs. Writes paired trajectories to artifacts/trajectories.parquet
(long format) and per-scenario divergence metrics to
artifacts/divergence.parquet. By default only the held-out 30% (by scenario id)
is analyzed, so downstream results are never computed on training data.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pdt_core
import torch

from pdt.metrics import FIELDS, compute_metrics_row
from pdt.policy import DECISION_NAMES, POLICY_CFG, MLPolicy, is_train_id, rollout

_CTX: dict = {}


def _worker_init(checkpoint_path: str, seed: int, overrides: dict) -> None:
    torch.set_num_threads(1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("cfg", POLICY_CFG)
    model = MLPolicy(cfg["input_dim"], cfg["hidden"], 2 * cfg["out_steps"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    planner_cfg = pdt_core.PlannerConfig()
    if overrides:
        planner_cfg.overrides = {k: bool(v) for k, v in overrides.items()}
    _CTX["model"] = model
    _CTX["cfg"] = cfg
    _CTX["seed"] = seed
    _CTX["planner"] = pdt_core.RulePlanner(planner_cfg)


def state_from_dict(d: dict) -> pdt_core.State:
    s = pdt_core.State()
    s.t = float(d["t"])
    s.x = float(d["x"])
    s.y = float(d["y"])
    s.heading = float(d["heading"])
    s.v = float(d["v"])
    s.a = float(d["a"])
    return s


def scenario_from_dict(d: dict) -> pdt_core.Scenario:
    sc = pdt_core.Scenario()
    sc.id = d["id"]
    sc.tag = d.get("tag", "other")
    sc.ego_init = state_from_dict(d["ego_init"])
    agents = []
    for a in d.get("agents", []):
        ag = pdt_core.Agent()
        ag.id = int(a["id"])
        ag.type = a["type"]
        ag.track = [state_from_dict(s) for s in a["track"]]
        agents.append(ag)
    sc.agents = agents
    sc.centerline = [[float(x), float(y)] for x, y in d.get("centerline", [])]
    sc.speed_limit = float(d.get("speed_limit", 0.0))
    sc.logged_ego = [state_from_dict(s) for s in d.get("logged_ego", [])]
    return sc


def _run_task(task: tuple[str, int]) -> dict:
    path, offset = task
    with open(path, "rb") as f:
        f.seek(offset)
        line = f.readline().decode()
    d = json.loads(line)

    sc = scenario_from_dict(d)
    t0 = time.perf_counter()
    traj = _CTX["planner"].plan(sc)
    t_rule = time.perf_counter() - t0
    rule_states = np.array([[s.t, s.x, s.y, s.heading, s.v, s.a] for s in traj.states], dtype=float)
    rule_decisions = [int(getattr(x, "value", x)) for x in traj.decisions]
    events = [{"step": e.step, "name": e.name, "active": bool(e.active), "reason": e.reason} for e in _CTX["planner"].events()]

    t0 = time.perf_counter()
    ml_states, ml_decisions = rollout(d, _CTX["model"], _CTX["cfg"], _CTX["seed"])
    t_ml = time.perf_counter() - t0

    row = compute_metrics_row(
        d,
        {"states": rule_states, "decisions": rule_decisions},
        {"states": ml_states, "decisions": ml_decisions},
    )
    return {
        "scenario_id": d["id"],
        "rule_states": rule_states,
        "rule_decisions": rule_decisions,
        "ml_states": ml_states,
        "ml_decisions": ml_decisions,
        "metrics": row,
        "events": events,
        "t_rule": t_rule,
        "t_ml": t_ml,
    }


def default_workers() -> int:
    return int(os.environ.get("PDT_WORKERS", max(1, (os.cpu_count() or 2) - 1)))


def line_offsets(path: Path, split: str) -> list[int]:
    offsets = []
    with open(path, "rb") as f:
        pos = 0
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if line and not line.startswith("#"):
                sid = None
                try:
                    sid = json.loads(line).get("id")
                except json.JSONDecodeError:
                    pass
                if sid is None or split == "held_out" and is_train_id(sid) or split == "train" and not is_train_id(sid):
                    pass
                else:
                    offsets.append(pos)
            pos += len(raw)
    return offsets


def run(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint} (run python -m pdt.train_policy first)")

    overrides = json.loads(args.overrides) if args.overrides else {}

    offsets = line_offsets(args.scenarios, args.split)
    if args.limit:
        offsets = offsets[: args.limit]
    tasks = [(str(args.scenarios), off) for off in offsets]
    print(f"running shadow replay on {len(tasks)} scenarios (split={args.split}, workers={args.workers}, overrides={overrides or 'none'})")

    # fork is fast but deadlocks with torch's background threads (OpenMP) on
    # Linux; fall back to spawn whenever torch has been imported.
    ctx = mp.get_context("spawn" if "torch" in sys.modules else "fork")
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(str(args.checkpoint), args.seed, overrides)) as pool:
        results = list(pool.imap(_run_task, tasks, chunksize=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for source, states, decs in (
            ("rule", r["rule_states"], r["rule_decisions"]),
            ("ml", r["ml_states"], r["ml_decisions"]),
        ):
            for i in range(len(states)):
                rows.append(
                    {
                        "scenario_id": r["scenario_id"],
                        "source": source,
                        "step": i,
                        "t": states[i, 0],
                        "x": states[i, 1],
                        "y": states[i, 2],
                        "heading": states[i, 3],
                        "v": states[i, 4],
                        "a": states[i, 5],
                        "decision": decs[i],
                        "decision_name": DECISION_NAMES[decs[i]],
                    }
                )
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(results)} paired trajectories ({len(rows)} rows) to {args.out}")

    div = pd.DataFrame([r["metrics"] for r in results], columns=FIELDS)
    div.to_parquet(args.metrics_out)
    print(f"wrote {len(div)} divergence rows to {args.metrics_out}")

    event_rows = []
    for r in results:
        for e in r["events"]:
            event_rows.append({"scenario_id": r["scenario_id"], **e})
    pd.DataFrame(event_rows, columns=["scenario_id", "step", "name", "active", "reason"]).to_parquet(args.events_out)
    print(f"wrote {len(event_rows)} override events to {args.events_out}")

    if results:
        t_rule_total = sum(r["t_rule"] for r in results)
        t_ml_total = sum(r["t_ml"] for r in results)
        n = len(results)
        print(f"timing: rule planner {n / t_rule_total:.1f} scenarios/s "
              f"({t_rule_total / n * 1e3:.2f} ms/scenario, wall over {args.workers} workers)")
        print(f"timing: ml policy    {n / t_ml_total:.1f} scenarios/s "
              f"({t_ml_total / n * 1e3:.2f} ms/scenario, wall over {args.workers} workers)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios/logs.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/policy.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/trajectories.parquet"))
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/divergence.parquet"))
    parser.add_argument("--events-out", type=Path, default=Path("artifacts/override_events.parquet"))
    parser.add_argument("--overrides", type=str, default=None, help='JSON dict, e.g. \'{"early_braking": true}\'')
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["held_out", "train", "all"], default="held_out")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
