"""Human triage of divergence clusters (CLI: pdt-review).

Walks clusters one at a time in descending size. For each cluster it prints the
auto-generated signature, the centroid metrics, and an ASCII side-by-side plot
of the rule vs ML trajectory (x-y path and speed profile) for each of the 5
medoid exemplars, then prompts the reviewer for:

  label   : desirable | undesirable | mixed
  rationale: free text (required)
  ported_rule_hint: free text (optional)

Labels persist to labels/cluster_labels.yaml (human-editable, version
controlled; comment header included). Re-running shows existing labels and only
prompts for unlabeled clusters. `--non-interactive --labels-from <file>` runs
without prompts for CI (every cluster must then be covered by the labels file).

Also exports a matplotlib PNG per cluster to artifacts/plots/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_LABELS_PATH = Path("labels/cluster_labels.yaml")
LABEL_CHOICES = {"d": "desirable", "u": "undesirable", "m": "mixed"}
LABELS_HEADER = (
    "# Cluster labels for divergence triage. Human-editable and version-controlled.\n"
    "# These are proposals; review and edit before using them in the regression gate.\n"
    "# label: desirable | undesirable | mixed\n"
    "# rationale: required free text. ported_rule_hint: optional free text.\n"
    "# Cluster ids refer to the persisted cluster model in artifacts/cluster_model.joblib;\n"
    "# re-fitting the model invalidates them.\n"
)


def load_labels(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"version": 1, "clusters": {}}
    raw = yaml.safe_load(path.read_text()) or {}
    clusters = {}
    for k, v in (raw.get("clusters") or {}).items():
        clusters[int(k)] = {
            "label": v.get("label"),
            "rationale": v.get("rationale", ""),
            "ported_rule_hint": v.get("ported_rule_hint"),
        }
    return {"version": raw.get("version", 1), "clusters": clusters}


def save_labels(path: str | Path, labels: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": labels.get("version", 1), "clusters": {str(k): v for k, v in sorted(labels["clusters"].items())}}
    path.write_text(LABELS_HEADER + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def _grid(rows: np.ndarray, w: int, h: int, xmin: float, xmax: float, ymin: float, ymax: float, mark: str) -> list[list[str]]:
    grid = [[" "] * w for _ in range(h)]
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    for r in rows:
        gx = int(np.clip((r[0] - xmin) / xspan * (w - 1), 0, w - 1))
        gy = int(np.clip((r[1] - ymin) / yspan * (h - 1), 0, h - 1))
        if grid[gy][gx] == " ":
            grid[gy][gx] = mark
        elif grid[gy][gx] != mark:
            grid[gy][gx] = "+"
    return grid


def ascii_path(xr: np.ndarray, yr: np.ndarray, xm: np.ndarray, ym: np.ndarray, w: int = 52, h: int = 13) -> list[str]:
    xs = np.concatenate([xr, xm])
    ys = np.concatenate([yr, ym])
    pad = max((xs.max() - xs.min()), (ys.max() - ys.min()), 1.0) * 0.05
    grid_r = _grid(np.stack([xr, yr], axis=1), w, h, xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad, "R")
    grid_m = _grid(np.stack([xm, ym], axis=1), w, h, xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad, "M")
    grid = [[("+" if grid_r[gy][gx] == "R" and grid_m[gy][gx] == "M" else (grid_r[gy][gx] if grid_r[gy][gx] != " " else grid_m[gy][gx])) for gx in range(w)] for gy in range(h)]
    return [f"x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}]", f"+{'-' * w}+"] + ["|" + "".join(row) + "|" for row in grid] + [f"+{'-' * w}+"]


def ascii_speed(tr: np.ndarray, vr: np.ndarray, tm: np.ndarray, vm: np.ndarray, w: int = 52, h: int = 13) -> list[str]:
    tmin = min(tr.min(), tm.min())
    tmax = max(tr.max(), tm.max())
    vmax = max(vr.max(), vm.max(), 1.0)
    grid_r = _grid(np.stack([tr, vr], axis=1), w, h, tmin, tmax, 0.0, vmax, "R")
    grid_m = _grid(np.stack([tm, vm], axis=1), w, h, tmin, tmax, 0.0, vmax, "M")
    grid = [[("+" if grid_r[gy][gx] == "R" and grid_m[gy][gx] == "M" else (grid_r[gy][gx] if grid_r[gy][gx] != " " else grid_m[gy][gx])) for gx in range(w)] for gy in range(h)]
    return [f"t[{tmin:.0f},{tmax:.0f}]s v[0,{vmax:.0f}]", f"+{'-' * w}+"] + ["|" + "".join(row) + "|" for row in grid] + [f"+{'-' * w}+"]


def render_exemplar(sid: str, traj: pd.DataFrame, div_row: pd.Series) -> str:
    rule = traj[(traj.scenario_id == sid) & (traj.source == "rule")]
    ml = traj[(traj.scenario_id == sid) & (traj.source == "ml")]
    if rule.empty or ml.empty:
        return f"{sid}: no trajectory rows"
    path_lines = ascii_path(rule.x.to_numpy(), rule.y.to_numpy(), ml.x.to_numpy(), ml.y.to_numpy())
    speed_lines = ascii_speed(rule.t.to_numpy(), rule.v.to_numpy(), ml.t.to_numpy(), ml.v.to_numpy())
    n = max(len(path_lines), len(speed_lines))
    out = [f"{sid}  (x-y path [R=rule, M=ml]            |  speed profile [R=rule, M=ml])"]
    for i in range(n):
        left = path_lines[i] if i < len(path_lines) else " " * 55
        right = speed_lines[i] if i < len(speed_lines) else ""
        out.append(f"{left:<55} | {right}")
    if div_row is not None and not div_row.empty:
        d = div_row.iloc[0]
        out.append(
            f"  score={d['divergence_score']:.3f}  ttc rule/ml={d['min_ttc_rule_s']:.2f}/{d['min_ttc_ml_s']:.2f}s "
            f" hard_brake={int(d['hard_brake_rule'])}/{int(d['hard_brake_ml'])} "
            f" collision={int(d['collision_rule'])}/{int(d['collision_ml'])} "
            f" progress={d['completion_progress_rule_m']:.0f}/{d['completion_progress_ml_m']:.0f}m"
        )
    return "\n".join(out)


def generate_cluster_png(cluster: dict, traj: pd.DataFrame, div: pd.DataFrame, out_dir: Path) -> Path:
    exemplars = cluster["exemplars"][:5]
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(10, 2.6 * len(exemplars)))
    if len(exemplars) == 1:
        axes = axes[None, :]
    for i, sid in enumerate(exemplars):
        rule = traj[(traj.scenario_id == sid) & (traj.source == "rule")]
        ml = traj[(traj.scenario_id == sid) & (traj.source == "ml")]
        ax = axes[i, 0]
        ax.plot(rule.x, rule.y, "-", color="tab:red", label="rule")
        ax.plot(ml.x, ml.y, "-", color="tab:blue", label="ml")
        ax.set_title(sid)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
        ax = axes[i, 1]
        ax.plot(rule.t, rule.v, "-", color="tab:red", label="rule")
        ax.plot(ml.t, ml.v, "-", color="tab:blue", label="ml")
        ax.set_ylabel("v (m/s)")
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=7)
    fig.suptitle(f"cluster {cluster['cluster_id']} (n={cluster['size']}): {cluster['signature']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cluster_{cluster['cluster_id']:02d}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> int:
    summary = json.loads(Path(args.summary).read_text())
    clusters = summary["clusters"]
    labels = load_labels(args.labels)
    if args.labels_from:
        extra = load_labels(args.labels_from)
        for cid, entry in extra["clusters"].items():
            labels["clusters"].setdefault(cid, entry)

    traj = pd.read_parquet(args.trajectories)
    div = pd.read_parquet(args.divergence)

    missing = []
    for cluster in clusters:
        cid = cluster["cluster_id"]
        generate_cluster_png(cluster, traj, div, Path(args.plots_dir))
        print(f"\n=== cluster {cid}  size={cluster['size']}  centroid tag={cluster['centroid']['dominant_tag']} ===")
        print(f"signature: {cluster['signature']}")
        print("centroid metrics: " + ", ".join(
            f"{k}={v:.3g}" for k, v in cluster["centroid"].items()
            if isinstance(v, float) and k not in ("mean_lateral_deviation_m",)
        ))
        existing = labels["clusters"].get(cid)
        if existing is not None:
            print(f"already labeled: {existing['label']} - {existing['rationale'][:100]}")
            continue
        if args.non_interactive:
            missing.append(cid)
            continue
        for sid in cluster["exemplars"]:
            row = div[div.scenario_id == sid]
            print(render_exemplar(sid, traj, row))
            print()
        while True:
            ans = input(f"[cluster {cid}] label [d]esirable/[u]ndesirable/[m]ixed: ").strip().lower()
            if ans in LABEL_CHOICES:
                label = LABEL_CHOICES[ans]
                break
        while True:
            rationale = input("rationale: ").strip()
            if rationale:
                break
        hint = input("ported_rule_hint (optional, empty to skip): ").strip() or None
        labels["clusters"][cid] = {"label": label, "rationale": rationale, "ported_rule_hint": hint}
        save_labels(args.labels, labels)
        print(f"saved label for cluster {cid} -> {args.labels}")

    if missing:
        print(f"non-interactive: missing labels for clusters {missing}; provide them via --labels-from", flush=True)
        return 1
    n_labeled = sum(1 for c in clusters if c["cluster_id"] in labels["clusters"])
    print(f"\nlabeled {n_labeled}/{len(clusters)} clusters; plots in {args.plots_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=str, default="artifacts/clusters.parquet")
    parser.add_argument("--summary", type=str, default="artifacts/cluster_summary.json")
    parser.add_argument("--trajectories", type=str, default="artifacts/trajectories.parquet")
    parser.add_argument("--divergence", type=str, default="artifacts/divergence.parquet")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--labels-from", type=str, default=None)
    parser.add_argument("--plots-dir", type=str, default="artifacts/plots")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
