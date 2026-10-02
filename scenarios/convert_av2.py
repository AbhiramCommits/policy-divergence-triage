#!/usr/bin/env python3
"""Convert fetched Argoverse 2 scenarios into the JSONL Scenario format.

Reads each scenario parquet under scenarios/av2_raw/val/<id>/ with the av2 API,
localizes all geometry to the ego frame (ego start at origin, heading 0),
maps tracks to {vehicle, pedestrian, cyclist}, extracts the lane centerline(s)
the logged ego actually traversed from the scenario HD map, derives a scenario
tag with documented heuristics, and applies a documented per-lane-type default
speed limit (AV2 maps carry no speed limits). Writes scenarios/logs.jsonl.

The logged ego trajectory is kept as `logged_ego` (human-driver ground truth
used for behavior cloning and the "vs human" metrics).

Track replay note: AV2 tracks are real logged data split into an observed
window and a future window. We keep the full logged track for every agent so
the replay uses ground-truth motion throughout the horizon. Tracks containing
non-finite values are dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "av2_raw" / "val"
OUT_FILE = ROOT / "logs.jsonl"

# Documented per-lane-type default speed limits (m/s). Argoverse 2 maps do not
# carry posted speed limits, so we apply these defaults by ego lane type:
SPEED_LIMIT_BY_LANE_TYPE = {
    "VEHICLE": 11.18,  # 25 mph - default urban
    "BUS": 11.18,      # 25 mph
    "BIKE": 6.71,      # 15 mph
}
DEFAULT_SPEED_LIMIT = 11.18

# Documented tag heuristics (checked in priority order):
TAG_CFG = {
    "turn_min_heading_deg": 35.0,
    "lane_change_max_heading_deg": 20.0,
    "lane_change_lateral_m": 1.5,
    "interaction_max_dist_m": 6.0,
    "lead_brake_dv_mps": 3.0,
    "lead_brake_window_s": 2.0,
    "lead_brake_max_ahead_m": 40.0,
    "near_lane_dist_m": 2.5,
}


def object_type_to_kind(ot) -> str | None:
    if ot in (ObjectType.VEHICLE, ObjectType.BUS):
        return "vehicle"
    if ot == ObjectType.PEDESTRIAN:
        return "pedestrian"
    if ot in (ObjectType.CYCLIST, ObjectType.MOTORCYCLIST, ObjectType.RIDERLESS_BICYCLE):
        return "cyclist"
    return None


def to_states(timestamps_ns: np.ndarray, pos: np.ndarray, heading: np.ndarray, vel: np.ndarray) -> list[dict]:
    t0 = timestamps_ns[0]
    t = (timestamps_ns - t0) / 1e9
    v = np.clip(np.linalg.norm(vel, axis=1), 0.0, None)
    a = np.zeros_like(v)
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        a[1:] = np.where(dt > 1e-9, np.diff(v) / dt, 0.0)
    a = np.nan_to_num(a, nan=0.0, posinf=10.0, neginf=-10.0)
    a = np.clip(a, -10.0, 10.0)
    return [
        {"t": float(t[i]), "x": float(pos[i, 0]), "y": float(pos[i, 1]),
         "heading": float(heading[i]), "v": float(v[i]), "a": float(a[i])}
        for i in range(len(t))
    ]


def localize(points: np.ndarray, origin: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (points - origin) @ rot.T


def load_map_lanes(map_json_path: Path) -> dict:
    """Lane segments from the scenario's log_map_archive_*.json (av2 >= 0.3
    keeps the HD map out of the scenario parquet). Keys are lane ids as
    strings; values expose centerline (N x 3), is_intersection, lane_type."""
    d = json.loads(map_json_path.read_text())
    lanes = {}
    for lane_id, seg in d.get("lane_segments", {}).items():
        cl = np.asarray([[p["x"], p["y"], p["z"]] for p in seg.get("centerline", [])], dtype=float)
        if len(cl) < 2:
            continue
        lanes[str(lane_id)] = SimpleNamespace(
            centerline=cl,
            is_intersection=bool(seg.get("is_intersection", False)),
            lane_type=str(seg.get("lane_type", "VEHICLE")),
        )
    return lanes


def lane_point_table(map_lanes: dict) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    ids = []
    for lane_id, lane in map_lanes.items():
        cl = np.asarray(lane.centerline, dtype=float)[:, :2]
        chunks.append(cl)
        ids.extend([lane_id] * len(cl))
    if not chunks:
        return np.zeros((0, 2)), np.zeros(0, dtype=object)
    return np.concatenate(chunks, axis=0), np.asarray(ids, dtype=object)


def ego_lanes(ego_xy_city: np.ndarray, map_lanes: dict) -> list[str]:
    """Match the (city-coordinate) ego positions to map lane centerlines."""
    pts, lane_of_pt = lane_point_table(map_lanes)
    if len(pts) == 0:
        return []
    ordered: list[str] = []
    for p in ego_xy_city[::5]:
        d = np.linalg.norm(pts - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < TAG_CFG["near_lane_dist_m"]:
            lane_id = lane_of_pt[j]
            if lane_id not in ordered:
                ordered.append(lane_id)
    return ordered


def stitch_centerline(lane_ids: list[str], map_lanes: dict, origin: np.ndarray, rot: np.ndarray) -> list[list[float]]:
    if not lane_ids:
        return []
    pts: list[list[float]] = []
    for lane_id in lane_ids:
        lane = map_lanes.get(lane_id)
        if lane is None:
            continue
        cl = localize(np.asarray(lane.centerline, dtype=float)[:, :2], origin, rot)
        if pts and np.hypot(cl[0, 0] - pts[-1][0], cl[0, 1] - pts[-1][1]) > 10.0:
            continue
        pts.extend([[float(x), float(y)] for x, y in cl])
    if len(pts) > 200:
        idx = np.linspace(0, len(pts) - 1, 200).astype(int)
        pts = [pts[i] for i in idx]
    return pts


def default_speed_limit(lane_ids: list[str], map_lanes: dict) -> float:
    if not lane_ids:
        return DEFAULT_SPEED_LIMIT
    lane = map_lanes.get(lane_ids[0])
    lt = getattr(lane, "lane_type", None) if lane is not None else None
    lt_str = lt.value if hasattr(lt, "value") else str(lt) if lt is not None else None
    return SPEED_LIMIT_BY_LANE_TYPE.get(lt_str, DEFAULT_SPEED_LIMIT)


def valid(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)))


def derive_tag(ego_states: list[dict], agents: list[dict], ego_lane_ids: list[str], map_lanes: dict,
               origin: np.ndarray, rot: np.ndarray) -> str:
    h = np.unwrap([s["heading"] for s in ego_states])
    heading_change_deg = math.degrees(float(h[-1] - h[0]))

    if heading_change_deg > TAG_CFG["turn_min_heading_deg"]:
        return "left_turn"
    if heading_change_deg < -TAG_CFG["turn_min_heading_deg"]:
        return "right_turn"

    ego_pos = np.array([[s["x"], s["y"]] for s in ego_states[::5]], dtype=float)

    if abs(heading_change_deg) < TAG_CFG["lane_change_max_heading_deg"] and ego_lane_ids:
        lanes = [map_lanes.get(lid) for lid in ego_lane_ids]
        lanes = [l for l in lanes if l is not None]
        if lanes:
            cl_all = np.concatenate(
                [localize(np.asarray(l.centerline, dtype=float)[:, :2], origin, rot) for l in lanes], axis=0
            )
            signs = []
            for p in ego_pos:
                d = cl_all - p
                dist = np.linalg.norm(d, axis=1)
                j = int(np.argmin(dist))
                j2 = j + 1 if j < len(cl_all) - 1 else j - 1
                t = cl_all[j2] - cl_all[j]
                tn = np.linalg.norm(t)
                if tn < 1e-9:
                    continue
                t = t / tn
                signs.append(t[0] * (p[1] - cl_all[j, 1]) - t[1] * (p[0] - cl_all[j, 0]))
            if signs and min(signs) < -TAG_CFG["lane_change_lateral_m"] and max(signs) > TAG_CFG["lane_change_lateral_m"]:
                return "lane_change"

    for ag in agents:
        if ag["type"] in ("pedestrian", "cyclist"):
            ap = np.array([[s["x"], s["y"]] for s in ag["track"][::5]], dtype=float)
            d = np.linalg.norm(ego_pos[:, None, :] - ap[None, :, :], axis=2)
            if d.min() < TAG_CFG["interaction_max_dist_m"]:
                return "ped_or_cyclist_interaction"

    for ag in agents:
        if ag["type"] != "vehicle":
            continue
        v = np.array([s["v"] for s in ag["track"]], dtype=float)
        w = int(TAG_CFG["lead_brake_window_s"] / 0.1)
        dv = v[w:] - v[:-w]
        bad = np.nonzero(dv < -TAG_CFG["lead_brake_dv_mps"])[0]
        if len(bad):
            i = int(bad[0]) + w
            p = np.array([ag["track"][i]["x"], ag["track"][i]["y"]])
            ahead = float(np.min(np.linalg.norm(ego_pos - p, axis=1)))
            if ahead < TAG_CFG["lead_brake_max_ahead_m"]:
                return "lead_vehicle_braking"

    for lane_id in ego_lane_ids:
        lane = map_lanes.get(lane_id)
        if lane is not None and getattr(lane, "is_intersection", False):
            return "straight_through_intersection"

    return "other"


def convert_scenario(parquet_path: Path) -> dict | None:
    scenario = scenario_serialization.load_argoverse_scenario_parquet(str(parquet_path))

    map_archives = sorted(parquet_path.parent.glob("log_map_archive_*.json"))
    map_lanes = load_map_lanes(map_archives[0]) if map_archives else {}

    timestamps_ns = np.asarray(scenario.timestamps_ns, dtype=np.int64)
    tracks = scenario.tracks
    focal_id = scenario.focal_track_id

    def track_states(track) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if not track.object_states:
            return None
        idx = np.array([os.timestep for os in track.object_states], dtype=np.int64)
        t = (timestamps_ns[idx] - timestamps_ns[0]) / 1e9
        pos = np.asarray([os.position for os in track.object_states], dtype=float)
        heading = np.asarray([os.heading for os in track.object_states], dtype=float)
        vel = np.asarray([os.velocity for os in track.object_states], dtype=float)
        return t, pos, heading, vel

    ego_track = next((tr for tr in tracks if str(tr.track_id) == focal_id), None)
    if ego_track is None:
        return None
    ego_data = track_states(ego_track)
    if ego_data is None:
        return None
    t_ego, pos_ego, h_ego, vel_ego = ego_data

    origin = pos_ego[0].astype(float)
    yaw = float(h_ego[0])
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

    ego_xy = localize(pos_ego, origin, rot)
    ego_h = np.unwrap(h_ego - yaw)
    ego_vel = localize(vel_ego, np.zeros(2), rot)
    if not (valid(ego_xy) and valid(ego_h) and valid(ego_vel)):
        return None
    ego_states = to_states(t_ego, np.column_stack([ego_xy, np.zeros(len(ego_xy))]), ego_h, ego_vel)

    agents = []
    for tr in tracks:
        sid = str(tr.track_id)
        if sid == focal_id or sid == "AV":
            continue
        kind = object_type_to_kind(tr.object_type)
        if kind is None:
            continue
        try:
            agent_id = int(sid)
        except ValueError:
            continue
        data = track_states(tr)
        if data is None:
            continue
        t_a, pos_a, h_a, vel_a = data
        pos = localize(pos_a, origin, rot)
        h = np.unwrap(h_a - yaw)
        vel = localize(vel_a, np.zeros(2), rot)
        if not (valid(pos) and valid(h) and valid(vel)):
            continue
        agents.append({
            "id": agent_id,
            "type": kind,
            "track": to_states(t_a, np.column_stack([pos, np.zeros(len(pos))]), h, vel),
        })

    lane_ids = ego_lanes(pos_ego, map_lanes)
    centerline = stitch_centerline(lane_ids, map_lanes, origin, rot)
    speed_limit = default_speed_limit(lane_ids, map_lanes)
    tag = derive_tag(ego_states, agents, lane_ids, map_lanes, origin, rot)

    return {
        "id": str(scenario.scenario_id),
        "tag": tag,
        "ego_init": ego_states[0],
        "agents": agents,
        "centerline": centerline,
        "speed_limit": speed_limit,
        "logged_ego": ego_states,
    }


def _worker(path: Path) -> tuple[str | None, dict | None]:
    try:
        sc = convert_scenario(path)
    except Exception as e:  # noqa: BLE001
        return str(e), None
    return None, sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR, help="directory of raw scenario folders")
    parser.add_argument("--output", type=Path, default=OUT_FILE, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N scenarios")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    parquet_files = sorted(args.input.glob("*/*.parquet"))
    if not parquet_files:
        sys.exit(f"no scenario parquets found under {args.input}; run scenarios/fetch_av2.py first")

    manifest = ROOT / "manifest.txt"
    if manifest.exists():
        wanted = {
            line.strip() for line in manifest.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if wanted:
            parquet_files = [p for p in parquet_files if p.parent.name in wanted]
    if args.limit:
        parquet_files = parquet_files[: args.limit]

    tag_counts: Counter = Counter()
    written = 0
    failed = 0
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with open(args.output, "w") as f, ctx.Pool(args.workers) as pool:
        for err, sc in pool.imap(_worker, parquet_files, chunksize=4):
            if sc is None:
                failed += 1
                if err:
                    print(f"failed: {err}", file=sys.stderr)
                continue
            f.write(json.dumps(sc, separators=(",", ":")) + "\n")
            tag_counts[sc["tag"]] += 1
            written += 1

    print(f"converted {written} scenarios (failed: {failed}) -> {args.output}")
    print("tag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag:28s} {count:5d}")


if __name__ == "__main__":
    main()
