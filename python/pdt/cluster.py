"""Clustering of divergence signatures (CLI: pdt-cluster).

Feature vector per scenario: 9 numeric metric columns (documented below), a
one-hot of the first decision flip kind (12 fixed rule/ml decision pairs), and
a one-hot of the scenario tag (7 fixed tags). Numeric columns are standardized
with StandardScaler; the one-hots are appended unscaled. Rows are filtered to
divergence_score strictly above a configurable threshold.

Numeric features (deltas are ML - rule):
  max_lateral_deviation_m, mean_lateral_deviation_m, final_position_gap_m,
  ttc_delta_s (NaN, i.e. no agent on a collision course for either system,
  imputed as 0), jerk_delta, accel_delta (mean_abs_accel_ml -
  mean_abs_accel_rule), decision_flip_count, progress_delta
  (completion_progress_ml_m - completion_progress_rule_m), ade_delta
  (ade_ml_vs_human_m - ade_rule_vs_human_m).

Two algorithms run: KMeans (k swept 3..12, k chosen by silhouette score) and
DBSCAN (eps swept 0.5..1.5). The full sweep is reported; the persisted model
is KMeans because it assigns every row to a cluster, which the regression gate
needs. The fitted scaler + model are persisted to
artifacts/cluster_model.joblib so new runs ASSIGN to existing clusters instead
of re-fitting; cluster ids are only meaningful relative to the persisted model
(re-fitting changes them).

Outputs:
  artifacts/clusters.parquet    scenario_id, cluster_id, distance_to_centroid
  artifacts/cluster_summary.json  per-cluster size, centroid in original metric
                                units, 5 medoid exemplar scenario_ids, and an
                                auto-generated human-readable signature
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pdt.metrics import DECISION_NAMES

FLIP_KINDS = [f"rule_{r}_ml_{m}" for r in DECISION_NAMES.values() for m in DECISION_NAMES.values() if r != m]
TAGS = [
    "left_turn",
    "right_turn",
    "straight_through_intersection",
    "lead_vehicle_braking",
    "ped_or_cyclist_interaction",
    "lane_change",
    "other",
]
NUMERIC_FEATURES = [
    "max_lateral_deviation_m",
    "mean_lateral_deviation_m",
    "final_position_gap_m",
    "ttc_delta_s",
    "jerk_delta",
    "accel_delta",
    "decision_flip_count",
    "progress_delta",
    "ade_delta",
]
DEFAULT_THRESHOLD = 0.5
DBSCAN_EPS_SWEEP = [0.5, 0.75, 1.0, 1.25, 1.5]
DBSCAN_MIN_SAMPLES = 5


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    d = df.copy()
    d["ttc_delta_s"] = d["ttc_delta_s"].fillna(0.0)
    d["accel_delta"] = d["mean_abs_accel_ml"] - d["mean_abs_accel_rule"]
    d["progress_delta"] = d["completion_progress_ml_m"] - d["completion_progress_rule_m"]
    d["ade_delta"] = d["ade_ml_vs_human_m"] - d["ade_rule_vs_human_m"]
    d["decision_flip_count"] = d["decision_flip_count"].astype(float)
    numeric = d[NUMERIC_FEATURES].to_numpy(dtype=float)

    flip = np.zeros((len(d), len(FLIP_KINDS)))
    for i, fk in enumerate(d["flip_kind"]):
        if fk in FLIP_KINDS:
            flip[i, FLIP_KINDS.index(fk)] = 1.0
    tag = np.zeros((len(d), len(TAGS)))
    for i, tg in enumerate(d["tag"]):
        if tg in TAGS:
            tag[i, TAGS.index(tg)] = 1.0
    return d, numeric, np.hstack([flip, tag])


def fit_bundle(
    df: pd.DataFrame, threshold: float, seed: int, quiet: bool = False
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    filtered = df[df["divergence_score"] > threshold].reset_index(drop=True)
    if len(filtered) < 4:
        raise ValueError(f"only {len(filtered)} rows above threshold {threshold}; need at least 4 to cluster")

    d, numeric, onehot = build_features(filtered)
    scaler = StandardScaler().fit(numeric)
    X = np.hstack([scaler.transform(numeric), onehot])

    k_max = min(12, len(X) - 1)
    sweep: dict[str, float] = {}
    best_k, best_sil, best_model = 0, -np.inf, None
    for k in range(3, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels, random_state=seed))
        sweep[str(k)] = sil
        if sil > best_sil:
            best_k, best_sil, best_model = k, sil, model

    dbs: dict[str, dict] = {}
    for eps in DBSCAN_EPS_SWEEP:
        model = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        labels = model.fit_predict(X)
        core = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil = float("nan")
        if n_clusters >= 2 and int(core.sum()) >= 2:
            sil = float(silhouette_score(X[core], labels[core], random_state=seed))
        dbs[str(eps)] = {"silhouette": sil, "n_clusters": n_clusters, "n_noise": int((~core).sum())}

    if not quiet:
        print("kmeans silhouette sweep:", ", ".join(f"k={k}:{s:.4f}" for k, s in sweep.items()))
        print("dbscan sweep:", ", ".join(
            f"eps={e}:sil={m['silhouette']:.4f},c={m['n_clusters']},noise={m['n_noise']}" for e, m in dbs.items()
        ))
        print(f"selected: kmeans k={best_k} silhouette={best_sil:.4f} (persisted; dbscan reported only)")

    bundle = {
        "method": "kmeans",
        "k": best_k,
        "scaler": scaler,
        "model": best_model,
        "feature_columns": NUMERIC_FEATURES,
        "flip_kinds": FLIP_KINDS,
        "tags": TAGS,
        "meta": {
            "threshold": threshold,
            "seed": seed,
            "n_fit_rows": len(filtered),
            "silhouette_sweep": sweep,
            "dbscan": dbs,
            "best_silhouette": best_sil,
        },
    }
    return bundle, filtered, X, best_model.predict(X)


def assign(bundle: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    d, numeric, onehot = build_features(df)
    X = np.hstack([bundle["scaler"].transform(numeric), onehot])
    labels = bundle["model"].predict(X)
    dists = bundle["model"].transform(X).min(axis=1)
    return d, labels, dists


def signature(centroid: dict, size: int) -> str:
    parts = []
    ttc = centroid.get("ttc_delta_s")
    if ttc is not None and abs(ttc) > 0.5:
        parts.append(f"ML min TTC {abs(ttc):.1f} s {'lower' if ttc < 0 else 'higher'}")
    flip = centroid.get("dominant_flip_kind", "")
    if flip.startswith("rule_") and "_ml_" in flip:
        rule_dec, ml_dec = flip[5:].split("_ml_")
        parts.append(f"ML {ml_dec} where rule {rule_dec}")
    prog = centroid.get("progress_delta")
    if prog is not None and abs(prog) > 1.0:
        parts.append(f"ML progress {prog:+.1f} m")
    lat = centroid.get("max_lateral_deviation_m")
    if lat is not None and lat > 1.5:
        parts.append(f"max lateral deviation {lat:.1f} m")
    jerk = centroid.get("jerk_delta")
    if jerk is not None and abs(jerk) > 2.0:
        parts.append(f"ML jerk {jerk:+.1f} m/s^3")
    ade = centroid.get("ade_delta")
    if ade is not None and abs(ade) > 1.0:
        parts.append(f"ML ADE vs human {ade:+.1f} m")
    if not parts:
        parts.append(f"moderate divergence across {size} scenarios")
    return "; ".join(parts)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def summarize(
    bundle: dict, filtered: pd.DataFrame, labels: np.ndarray, dists: np.ndarray
) -> dict:
    clusters = []
    n_oh = len(FLIP_KINDS)
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        center = bundle["model"].cluster_centers_[cid]
        num_centroid = bundle["scaler"].inverse_transform(center[: len(NUMERIC_FEATURES)][None, :])[0]
        centroid = {name: float(v) for name, v in zip(NUMERIC_FEATURES, num_centroid)}
        flip_part = center[len(NUMERIC_FEATURES): len(NUMERIC_FEATURES) + n_oh]
        tag_part = center[len(NUMERIC_FEATURES) + n_oh:]
        centroid["dominant_flip_kind"] = FLIP_KINDS[int(flip_part.argmax())] if flip_part.max() > 0 else ""
        centroid["dominant_tag"] = TAGS[int(tag_part.argmax())]
        order = np.argsort(dists[idx], kind="stable")
        exemplars = [filtered.iloc[int(idx[j])]["scenario_id"] for j in order[:5]]
        size = int(len(idx))
        clusters.append({
            "cluster_id": int(cid),
            "size": size,
            "centroid": centroid,
            "exemplars": exemplars,
            "signature": signature(centroid, size),
        })
    clusters.sort(key=lambda c: (-c["size"], c["cluster_id"]))
    return _sanitize({
        "method": bundle["method"],
        "k": bundle["k"],
        "threshold": bundle["meta"]["threshold"],
        "seed": bundle["meta"]["seed"],
        "n_rows_filtered": bundle["meta"]["n_fit_rows"],
        "silhouette_sweep": bundle["meta"]["silhouette_sweep"],
        "dbscan": bundle["meta"]["dbscan"],
        "best_silhouette": bundle["meta"]["best_silhouette"],
        "clusters": clusters,
    })


def run_clustering(
    divergence_path: str | Path,
    out_dir: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 0,
    refit: bool = False,
    quiet: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "cluster_model.joblib"
    df = pd.read_parquet(divergence_path)

    if model_path.exists() and not refit:
        bundle = joblib.load(model_path)
        d, labels, dists = assign(bundle, df)
        above = d["divergence_score"] > bundle["meta"]["threshold"]
        d, labels, dists = d[above].reset_index(drop=True), labels[above], dists[above]
        summary = summarize(bundle, d, labels, dists)
        summary["assigned_with_existing_model"] = True
        if not quiet:
            print(f"loaded persisted cluster model (k={bundle['k']}, "
                  f"fit on {bundle['meta']['n_fit_rows']} rows); assigned {len(d)} new rows")
    else:
        bundle, filtered, X, labels = fit_bundle(df, threshold, seed, quiet=quiet)
        dists = bundle["model"].transform(X).min(axis=1)
        d = filtered
        joblib.dump(bundle, model_path)
        summary = summarize(bundle, filtered, labels, dists)
        summary["assigned_with_existing_model"] = False

    clusters_df = pd.DataFrame({
        "scenario_id": d["scenario_id"].to_numpy(),
        "cluster_id": labels,
        "distance_to_centroid": dists,
    })
    clusters_df.to_parquet(out_dir / "clusters.parquet", index=False)
    (out_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2))

    if not quiet:
        for c in summary["clusters"]:
            print(f"cluster {c['cluster_id']:2d}  size={c['size']:3d}  {c['signature']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divergence", type=Path, default=Path("artifacts/divergence.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refit", action="store_true", help="re-fit instead of using the persisted model")
    args = parser.parse_args()
    run_clustering(args.divergence, args.out_dir, threshold=args.threshold, seed=args.seed, refit=args.refit)


if __name__ == "__main__":
    main()
