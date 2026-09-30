#!/usr/bin/env python3
"""Convert fetched Argoverse 2 scenarios into the JSONL Scenario format.

Reads each scenario parquet under scenarios/av2_raw/val/<id>/ with the av2 API,
localizes all geometry to the ego frame (ego start at origin, heading 0),
maps tracks to {vehicle, pedestrian, cyclist}, extracts the lane centerline(s)
the logged ego actually traversed from the scenario HD map, derives a scenario
tag with documented heuristics, and applies a documented per-lane-type default
speed limit (AV2 maps carry no speed limits). Writes scenarios/logs.jsonl.

The logged ego trajectory is kept as `logged_ego` (human-driver ground truth
used later for training and the "vs human" metrics).

Track replay note: AV2 tracks are real logged data split into an observed
window and a future window. We keep the full logged track for every agent so
the replay uses ground-truth motion throughout the horizon.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory

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
    "lane_change_min_lateral_m": 3.0,
    "lane_change_max_heading_deg": 20.0,
    "interaction_max_dist_m": 6.0,
    "lead_brake_dv_mps": 3.0,
    "lead_brake_window_s": 2.0,
    "lead_brake_max_ahead_m": 40.0,
    "near_lane_dist_m": 2.5,
}


def object_type_to_kind(ot) -> str | None:
    if ot == ObjectType.VEHICLE:
        return "vehicle"
    if ot == ObjectType.PEDESTRIAN:
        return "pedestrian"
    if ot in (ObjectType.CYCLIST, ObjectType.BICYCLIST, ObjectType.MOTORCYCLIST):
        return "cyclist"
    return None


def to_states(timestamps_ns: np.ndarray, pos: np.ndarray, heading: np.ndarray, vel: np.ndarray) -> list[dict]:
    t0 = timestamps_ns[0]
    t = (timestamps_ns - t0) / 1e9
    v = np.linalg.norm(vel, axis=1)
    a = np.zeros_like(v)
    a[1:] = np.diff(v) / np.diff(t)
    return [
        {"t": float(t[i]), "x": float(pos[i, 0]), "y": float(pos[i, 1]),
         "heading": float(heading[i]), "v": float(v[i]), "a": float(a[i])}
        for i in range(len(t))
    ]


def localize(points: np.ndarray, origin: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (points - origin) @ rot.T


def ego_lanes(ego_xy: np.ndarray, map_lanes: dict) -> list[str]:
    ordered: list[str] = []
    for p in ego_xy[::5]:
        for lane_id, lane in map_lanes.items():
            if lane_id in ordered:
                continue
            cl = lane.centerline[:, :2]
            if np.min(np.linalg.norm(cl - p, axis=1)) < TAG_CFG["near_lane_dist_m"]:
                ordered.append(lane_id)
                break
    return ordered


def stitch_centerline(lane_ids: list[str], map_lanes: dict) -> list[list[float]]:
    if not lane_ids:
        return []
    pts: list[list[float]] = []
    for lane_id in lane_ids:
        cl = map_lanes[lane_id].centerline[:, :2]
        if pts and len(pts[-1]) == 2:
            start = cl[0]
            if np.hypot(start[0] - pts[-1][0], start[1] - pts[-1][1]) > 10.0:
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


def derive_tag(ego_states: list[dict], agents: list[dict], ego_lane_ids: list[str], map_lanes: dict) -> str:
    h = np.unwrap([s["heading"] for s in ego_states])
    heading_change_deg = math.degrees(float(h[-1] - h[0]))

    if heading_change_deg > TAG_CFG["turn_min_heading_deg"]:
        return "left_turn"
    if heading_change_deg < -TAG_CFG["turn_min_heading_deg"]:
        return "right_turn"

    if abs(heading_change_deg) < TAG_CFG["lane_change_max_heading_deg"] and ego_lane_ids:
        lane = map_lanes.get(ego_lane_ids[0])
        if lane is not None:
            cl = lane.centerline[:, :2]
            lateral = []
            for s in ego_states[::5]:
                d = cl - np.array([s["x"], s["y"]])
                lateral.append(float(np.min(np.linalg.norm(d, axis=1))))
            if max(lateral) - min(lateral) > TAG_CFG["lane_change_min_lateral_m"]:
                return "lane_change"

    for ag in agents:
        if ag["type"] in ("pedestrian", "cyclist"):
            for s in ego_states[::5]:
                for t in ag["track"][::5]:
                    if math.hypot(t["x"] - s["x"], t["y"] - s["y"]) < TAG_CFG["interaction_max_dist_m"]:
                        return "ped_or_cyclist_interaction"

    ego_arr = np.array([[s["x"], s["y"]] for s in ego_states])
    for ag in agents:
        if ag["type"] != "vehicle":
            continue
        tr = ag["track"]
        for i in range(len(tr)):
            if i == 0:
                continue
            j = max(0, i - int(TAG_CFG["lead_brake_window_s"] / 0.1))
            dv = tr[i]["v"] - tr[j]["v"]
            if dv < -TAG_CFG["lead_brake_dv_mps"]:
                p = np.array([tr[i]["x"], tr[i]["y"]])
                ahead = float(np.min(np.linalg.norm(ego_arr - p, axis=1)))
                if ahead < TAG_CFG["lead_brake_max_ahead_m"]:
                    return "lead_vehicle_braking"

    for lane_id in ego_lane_ids:
        lane = map_lanes.get(lane_id)
        if lane is not None and getattr(lane, "is_intersection", False):
            return "straight_through_intersection"

    return "other"


def convert_scenario(parquet_path: Path) -> dict | None:
    scenario = scenario_serialization.load_argoverse_scenario_parquet(str(parquet_path))

    focal_id = scenario.focal_track_id
    tracks = scenario.tracks
    map_lanes = scenario.map_lanes

    ego_rows = tracks[tracks["track_id"] == focal_id]
    if ego_rows.empty:
        return None
    ego_row = ego_rows.iloc[0]

    origin = ego_row["position"][0].astype(float)
    yaw = float(ego_row["heading"][0])
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

    ego_xy = localize(ego_row["position"].astype(float), origin, rot)
    ego_h = np.unwrap(ego_row["heading"].astype(float) - yaw)
    ego_states = to_states(ego_row["timestamps_ns"], np.column_stack([ego_xy, np.zeros(len(ego_xy))]), ego_h,
                           localize(ego_row["velocity"].astype(float), np.zeros(2), rot))

    agents = []
    for _, row in tracks.iterrows():
        if row["track_id"] == focal_id:
            continue
        kind = object_type_to_kind(row["object_type"])
        if kind is None:
            continue
        pos = localize(row["position"].astype(float), origin, rot)
        h = np.unwrap(row["heading"].astype(float) - yaw)
        vel = localize(row["velocity"].astype(float), np.zeros(2), rot)
        agents.append({
            "id": int(row["track_id"], 16) % (2 ** 31) if isinstance(row["track_id"], str) else int(row["track_id"]),
            "type": kind,
            "track": to_states(row["timestamps_ns"], np.column_stack([pos, np.zeros(len(pos))]), h, vel),
        })

    lane_ids = ego_lanes(ego_xy, map_lanes)
    centerline = stitch_centerline(lane_ids, map_lanes)
    speed_limit = default_speed_limit(lane_ids, map_lanes)
    tag = derive_tag(ego_states, agents, lane_ids, map_lanes)

    return {
        "id": str(scenario.scenario_id),
        "tag": tag,
        "ego_init": ego_states[0],
        "agents": agents,
        "centerline": centerline,
        "speed_limit": speed_limit,
        "logged_ego": ego_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR, help="directory of raw scenario folders")
    parser.add_argument("--output", type=Path, default=OUT_FILE, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N scenarios")
    args = parser.parse_args()

    parquet_files = sorted(args.input.glob("*/*.parquet"))
    if not parquet_files:
        sys.exit(f"no scenario parquets found under {args.input}; run scenarios/fetch_av2.py first")
    if args.limit:
        parquet_files = parquet_files[: args.limit]

    tag_counts: Counter = Counter()
    written = 0
    failed = 0
    with open(args.output, "w") as f:
        for pf in parquet_files:
            try:
                sc = convert_scenario(pf)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"failed {pf.name}: {e}", file=sys.stderr)
                continue
            if sc is None:
                failed += 1
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
