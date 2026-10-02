"""Learned ML policy (PyTorch).

Model: a small MLP. Input = flattened ego state + the 6 nearest agents'
relative pose/velocity + 10 centerline points ahead of the ego, all expressed
in the ego frame. Hidden 256-256, ReLU. Output = 80 values = 40 steps x
(accel, steer_rate), interleaved [a0, sr0, a1, sr1, ...]. The model plans 40
steps (4.0 s) per query; the closed-loop rollout re-queries every 0.1 s step
and applies only the first action (receding horizon).

Rollout dynamics: kinematic bicycle model with steer-rate control.
  delta_k = clip(delta_{k-1} + steer_rate * dt, +-steer_angle_max)
  theta  += (v / wheelbase) * tan(delta) * dt
  v      = max(0, v + accel * dt)
  x += v cos(theta) dt;  y += v sin(theta) dt
  accel      clipped to [accel_min, accel_max]
  steer_rate clipped to +-steer_rate_max

Other agents replay their logged tracks (non-reactive) — the same non-reactive
environment the rule planner sees.

Feature normalization: positions /50 m, velocities /20 m/s, accel /5 m/s^2.

Decisions: the same enum as the planner {FOLLOW, YIELD, ASSERT, STOP} is
derived post-hoc from the ML rollout via a kinematic classifier (documented in
`classify_decisions`), so decision sequences are comparable to the planner's.

Inference is deterministic: model.eval(), torch.no_grad(), fixed seed.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn

from pdt.geom import (
    agents_at_t,
    arc_lengths,
    interpolate_agent_grid,
    point_at_arc,
    project_to_centerline,
    states_to_array,
)

POLICY_CFG: dict = {
    "input_dim": 49,
    "hidden": [256, 256],
    "out_steps": 40,
    "dt": 0.1,
    "wheelbase": 2.8,
    "n_agents": 6,
    "n_cl_points": 10,
    "cl_spacing": 5.0,
    "accel_min": -6.0,
    "accel_max": 5.0,
    "steer_rate_max": 1.0,
    "steer_angle_max": 0.6,
    "pos_scale": 50.0,
    "vel_scale": 20.0,
    "acc_scale": 5.0,
    "sr_scale": 1.0,
}

DECISION_CFG: dict = {
    "stop_speed": 0.05,
    "yield_accel": -1.0,
    "assert_accel_eps": 0.02,
    "assert_min_speed": 1.0,
    "leader_gap_m": 20.0,
    "vehicle_length": 4.5,
    "lane_half_width": 2.5,
}

FOLLOW, YIELD, ASSERT, STOP = 0, 1, 2, 3
DECISION_NAMES = {FOLLOW: "FOLLOW", YIELD: "YIELD", ASSERT: "ASSERT", STOP: "STOP"}


class MLPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_scenario_ids(ids: list[str], train_frac: float = 0.7) -> tuple[list[str], list[str]]:
    train: list[str] = []
    held_out: list[str] = []
    for sid in ids:
        h = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], "big")
        (train if h % 100 < round(train_frac * 100) else held_out).append(sid)
    return train, held_out


def is_train_id(sid: str, train_frac: float = 0.7) -> bool:
    h = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], "big")
    return h % 100 < round(train_frac * 100)


def scenario_arrays(scenario: dict) -> dict:
    cl = np.asarray(scenario.get("centerline") or [], dtype=float).reshape(-1, 2)
    tracks = []
    for a in scenario.get("agents", []):
        tr = states_to_array(a["track"])
        if len(tr) == 0:
            continue
        tr[:, 3] = np.unwrap(tr[:, 3])
        tracks.append((a["type"], tr))
    logged = states_to_array(scenario.get("logged_ego") or [])
    if len(logged):
        logged[:, 3] = np.unwrap(logged[:, 3])

    max_t = max(8.0, float(logged[-1, 0]) if len(logged) else 8.0)
    n_grid = round(max_t / POLICY_CFG["dt"]) + 1
    grid_types, agents_grid = interpolate_agent_grid(tracks, np.arange(n_grid) * POLICY_CFG["dt"])

    veh_idx = [i for i, t in enumerate(grid_types) if t == "vehicle"]
    veh_s: np.ndarray | None = None
    veh_d: np.ndarray | None = None
    if veh_idx and len(cl) >= 2:
        pts = agents_grid[veh_idx][:, :, :2].reshape(-1, 2)
        s_all, d_all = project_to_centerline(pts, cl, arc_lengths(cl))
        veh_s = s_all.reshape(len(veh_idx), n_grid)
        veh_d = d_all.reshape(len(veh_idx), n_grid)

    return {
        "cl": cl,
        "cum": arc_lengths(cl),
        "tracks": tracks,
        "logged_ego": logged,
        "speed_limit": float(scenario.get("speed_limit") or 0.0),
        "ego_init": scenario.get("ego_init") or {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "v": 0.0},
        "id": scenario.get("id", ""),
        "grid_types": grid_types,
        "agents_grid": agents_grid,
        "veh_idx": veh_idx,
        "veh_s": veh_s,
        "veh_d": veh_d,
    }


def agents_at(scn: dict, t: float, cfg: dict) -> np.ndarray:
    grid = scn.get("agents_grid")
    if grid is not None and len(grid):
        tidx = min(grid.shape[1] - 1, max(0, round(t / cfg["dt"])))
        return grid[:, tidx, :]
    return agents_at_t(scn["tracks"], t)


def build_features(scn: dict, state: np.ndarray, cfg: dict) -> np.ndarray:
    t, x, y, h, v, a = state[0], state[1], state[2], state[3], state[4], state[5]
    c = math.cos(h)
    s = math.sin(h)

    feats = [0.0, 0.0, 0.0, v / cfg["vel_scale"], a / cfg["acc_scale"]]

    ag = agents_at(scn, t, cfg)
    if len(ag):
        d = ag[:, :2] - np.array([x, y])
        dx = d[:, 0] * c + d[:, 1] * s
        dy = -d[:, 0] * s + d[:, 1] * c
        vx_a = ag[:, 3] * np.cos(ag[:, 2])
        vy_a = ag[:, 3] * np.sin(ag[:, 2])
        dvx = vx_a - v * c
        dvy = vy_a - v * s
        rel = np.stack([dx, dy, dvx, dvy], axis=1)
        dist = np.linalg.norm(d, axis=1)
        order = np.argsort(dist, kind="stable")[: cfg["n_agents"]]
        for j in order:
            feats.extend(
                [
                    rel[j, 0] / cfg["pos_scale"],
                    rel[j, 1] / cfg["pos_scale"],
                    rel[j, 2] / cfg["vel_scale"],
                    rel[j, 3] / cfg["vel_scale"],
                ]
            )
    if len(ag) < cfg["n_agents"]:
        feats.extend([0.0] * (4 * (cfg["n_agents"] - len(ag))))

    cl = scn["cl"]
    if len(cl) >= 2:
        s_ego, _ = project_to_centerline(np.array([[x, y]]), cl, scn["cum"])
        for k in range(cfg["n_cl_points"]):
            p = point_at_arc(cl, scn["cum"], s_ego[0] + k * cfg["cl_spacing"])
            dx = p[0] - x
            dy = p[1] - y
            feats.extend([(dx * c + dy * s) / cfg["pos_scale"], (-dx * s + dy * c) / cfg["pos_scale"]])
    else:
        feats.extend([0.0] * (2 * cfg["n_cl_points"]))

    return np.clip(np.asarray(feats, dtype=np.float32), -100.0, 100.0)


def integrate_step(state: np.ndarray, accel: float, steer_rate: float, cfg: dict) -> np.ndarray:
    dt = cfg["dt"]
    L = cfg["wheelbase"]
    t, x, y, h, v = state[0], state[1], state[2], state[3], state[4]
    delta = state[6] if len(state) > 6 else 0.0
    delta = float(np.clip(delta + steer_rate * dt, -cfg["steer_angle_max"], cfg["steer_angle_max"]))
    h = h + (v / L) * math.tan(delta) * dt
    v = max(0.0, v + accel * dt)
    x = x + v * math.cos(h) * dt
    y = y + v * math.sin(h) * dt
    t = t + dt
    return np.array([t, x, y, h, v, accel, delta], dtype=float)


def leader_gap_at(scn: dict, state: np.ndarray, dcfg: dict) -> float:
    x, y = state[1], state[2]
    cl = scn["cl"]
    if len(cl) < 2:
        return float("inf")
    s_ego, _ = project_to_centerline(np.array([[x, y]]), cl, scn["cum"])
    t = state[0]
    gap = float("inf")
    veh_s = scn.get("veh_s")
    veh_d = scn.get("veh_d")
    if veh_s is not None and veh_d is not None:
        tidx = min(veh_s.shape[1] - 1, max(0, round(t / dcfg.get("dt", 0.1))))
        for j in range(veh_s.shape[0]):
            s_v = veh_s[j, tidx]
            d_v = veh_d[j, tidx]
            if d_v > dcfg["lane_half_width"] or s_v <= s_ego[0]:
                continue
            g = s_v - s_ego[0] - dcfg["vehicle_length"]
            gap = min(gap, g)
        return gap
    if scn.get("agents_grid") is not None and len(scn["agents_grid"]):
        grid = scn["agents_grid"]
        types = scn["grid_types"]
        tidx = min(grid.shape[1] - 1, max(0, round(t / dcfg.get("dt", 0.1))))
        sts = grid[:, tidx, :]
        for typ, st in zip(types, sts):
            if typ != "vehicle":
                continue
            s_v, d_v = project_to_centerline(np.array([[st[0], st[1]]]), cl, scn["cum"])
            if d_v[0] > dcfg["lane_half_width"] or s_v[0] <= s_ego[0]:
                continue
            g = s_v[0] - s_ego[0] - dcfg["vehicle_length"]
            gap = min(gap, g)
        return gap
    for typ, tr in scn["tracks"]:
        if typ != "vehicle":
            continue
        st = tr[np.searchsorted(tr[:, 0], t, side="right") - 1] if t >= tr[0, 0] else tr[0]
        if t >= tr[-1, 0]:
            st = tr[-1]
        s_v, d_v = project_to_centerline(np.array([[st[1], st[2]]]), cl, scn["cum"])
        if d_v[0] > dcfg["lane_half_width"] or s_v[0] <= s_ego[0]:
            continue
        g = s_v[0] - s_ego[0] - dcfg["vehicle_length"]
        gap = min(gap, g)
    return gap


def classify_decisions(states: np.ndarray, scn: dict, dcfg: dict = DECISION_CFG) -> list[int]:
    """Post-hoc decision classifier for the ML rollout, based on the speed
    profile and gap to the nearest in-lane leader (the ML policy has no
    explicit conflict concept, so this is an observable-kinematics proxy for
    the planner's decision enum):

    STOP   : v <= stop_speed
    YIELD  : a <= yield_accel AND no in-lane leader within leader_gap_m
             (hard braking that is not explained by car-following)
    ASSERT : |a| <= assert_accel_eps AND v >= assert_min_speed AND no leader
             (exactly holding speed, matching the planner's ASSERT behavior)
    FOLLOW : everything else (free-flow, car-following, mild braking)
    """
    decs = []
    for st in states:
        v = st[4]
        a = st[5]
        if v <= dcfg["stop_speed"]:
            decs.append(STOP)
            continue
        gap = leader_gap_at(scn, st, dcfg)
        open_road = gap > dcfg["leader_gap_m"]
        if a <= dcfg["yield_accel"] and open_road:
            decs.append(YIELD)
        elif abs(a) <= dcfg["assert_accel_eps"] and v >= dcfg["assert_min_speed"] and open_road:
            decs.append(ASSERT)
        else:
            decs.append(FOLLOW)
    return decs


def rollout(scenario: dict, model: nn.Module, cfg: dict, seed: int, n_steps: int = 80) -> tuple[np.ndarray, list[int]]:
    torch.manual_seed(seed)
    model.eval()
    scn = scenario_arrays(scenario)
    ei = scenario["ego_init"]
    state = np.array([ei["t"], ei["x"], ei["y"], ei["heading"], max(0.0, ei["v"]), 0.0, 0.0], dtype=float)
    states_arr = [state.copy()]
    with torch.no_grad():
        for _ in range(n_steps):
            f = build_features(scn, state, cfg)
            out = model(torch.from_numpy(f).float().unsqueeze(0))
            a = float(out[0, 0].item()) * cfg["acc_scale"]
            sr = float(out[0, 1].item()) * cfg["sr_scale"]
            a = float(np.clip(a, cfg["accel_min"], cfg["accel_max"]))
            sr = float(np.clip(sr, -cfg["steer_rate_max"], cfg["steer_rate_max"]))
            state = integrate_step(state, a, sr, cfg)
            states_arr.append(state.copy())
    states = np.asarray(states_arr)
    decisions = classify_decisions(states, scn)
    return states[:, :6], decisions


def open_loop_ade_fde(scn: dict, model: nn.Module, cfg: dict) -> tuple[float, float]:
    """Open-loop validation: one model query at t0, apply the 40 predicted
    actions sequentially with no re-query and no state feedback, then compare
    against the logged human ego. `scn` is a scenario-arrays dict."""
    model.eval()
    human = scn["logged_ego"]
    if len(human) < cfg["out_steps"] + 1:
        return float("nan"), float("nan")
    ei = scn["ego_init"]
    state = np.array([ei["t"], ei["x"], ei["y"], ei["heading"], max(0.0, ei["v"]), 0.0, 0.0], dtype=float)
    with torch.no_grad():
        f = build_features(scn, state, cfg)
        out = model(torch.from_numpy(f).float().unsqueeze(0)).numpy()[0]
    errs = []
    for k in range(cfg["out_steps"]):
        a = float(np.clip(out[2 * k] * cfg["acc_scale"], cfg["accel_min"], cfg["accel_max"]))
        sr = float(np.clip(out[2 * k + 1] * cfg["sr_scale"], -cfg["steer_rate_max"], cfg["steer_rate_max"]))
        state = integrate_step(state, a, sr, cfg)
        errs.append(math.hypot(state[1] - human[k + 1, 1], state[2] - human[k + 1, 2]))
    return float(np.mean(errs)), float(errs[-1])
