"""Divergence metrics: one row per scenario comparing the rule planner and ML
policy trajectories. Every formula is documented here.

Inputs per scenario:
  rule/states, ml/states : (N, 6) arrays [t, x, y, heading, v, a], N = 81 steps
                           at dt = 0.1 s (t = 0 .. 8.0 s)
  rule/decisions         : int lists, enum {0=FOLLOW, 1=YIELD, 2=ASSERT, 3=STOP}
  scenario               : dict with agents (logged tracks), centerline,
                           logged_ego (human ground truth), tag, id

Formulas (R = rule, M = ml, H = human logged ego):

lateral deviation (step i): e = M_i - R_i (position); with R's unit heading
  vector u = (cos th_R, sin th_R) and normal n = (-sin th_R, cos th_R):
    cross_i = e . n
    max_lateral_deviation_m = max_i |cross_i|
    mean_lateral_deviation_m = mean_i |cross_i|

final_position_gap_m = ||M_last - R_last|| (position only)

TTC vs agents (per system, per step): for each agent state (x_a, y_a, th_a, v_a)
    d = p_a - p_ego; dist = ||d||
    v_rel = v_a (cos th_a, sin th_a) - v_ego (cos th_e, sin th_e)
    closing = -(d . v_rel) / dist
    ttc = dist / closing  if closing > 1e-3, else +inf
    min_ttc_X_s = min over steps and agents (inf -> NaN when no agent exists)
  ttc_delta_s = min_ttc_ml_s - min_ttc_rule_s (positive => ML safer;
    NaN (both infinite) is treated as 0 in the score)

jerk (step i): j_i = (a_{i+1} - a_i) / dt
    max_jerk_X = max_i |j_i|
    jerk_delta = max_jerk_ml - max_jerk_rule

mean_abs_accel_X = mean_i |a_i|

decision_flip_count = count of steps where rule decision != ml decision
first_flip_time_s = t at the first differing step (None if never)
flip_kind = "rule_<D>_ml_<D>" at the first differing step ("" if never)

completion_progress_X_m = arc length along the scenario centerline of the
  projection of the final position (fallback: cumulative path length of the
  system's own trajectory when no centerline is available)

hard_brake_X = any step with a_i < -4.0 m/s^2
collision_X  = any step where the distance to the nearest agent position
  (agents replay their logged tracks) is below 2.0 m

ade_X_vs_human_m = mean over aligned steps (same 0.1 s grid, t <= 8.0 s) of
  ||X_i - H_i||

divergence_score = sum_k w_k * n_k, each component normalized to [0, 1]:
    n_lateral  = tanh(max_lateral_deviation_m / 2.0)     w = 0.20
    n_position = tanh(final_position_gap_m / 3.0)        w = 0.15
    n_ttc      = tanh(|ttc_delta_s| / 2.0)  (0 if NaN)   w = 0.10
    n_jerk     = tanh(|jerk_delta| / 5.0)                w = 0.10
    n_accel    = tanh(|mean_abs_accel_ml - mean_abs_accel_rule| / 1.0)  w = 0.10
    n_decision = decision_flip_count / N                 w = 0.20
    n_ade      = tanh(|ade_ml - ade_rule| / 2.0)         w = 0.15
  score in [0, 1]; identical trajectories score exactly 0.0.
"""

from __future__ import annotations

import math

import numpy as np

from pdt.geom import arc_lengths, interpolate_agent_grid, project_to_centerline, states_to_array

DT = 0.1
COLLISION_RADIUS_M = 2.0
HARD_BRAKE_DECEL = 4.0
TTC_CLOSING_EPS = 1e-3

SCORE_WEIGHTS = {
    "lateral": 0.20,
    "position": 0.15,
    "ttc": 0.10,
    "jerk": 0.10,
    "accel": 0.10,
    "decision": 0.20,
    "ade": 0.15,
}

DECISION_NAMES = {0: "FOLLOW", 1: "YIELD", 2: "ASSERT", 3: "STOP"}

FIELDS = [
    "scenario_id", "tag", "max_lateral_deviation_m", "mean_lateral_deviation_m",
    "final_position_gap_m", "min_ttc_rule_s", "min_ttc_ml_s", "ttc_delta_s",
    "max_jerk_rule", "max_jerk_ml", "jerk_delta", "mean_abs_accel_rule",
    "mean_abs_accel_ml", "decision_flip_count", "first_flip_time_s", "flip_kind",
    "completion_progress_rule_m", "completion_progress_ml_m", "hard_brake_rule",
    "hard_brake_ml", "collision_rule", "collision_ml", "ade_rule_vs_human_m",
    "ade_ml_vs_human_m", "divergence_score",
]


def min_ttc_at_step(ego_state: np.ndarray, agent_states: np.ndarray) -> float:
    """Minimum TTC of one ego state against a set of agent states (see module
    docstring). Returns +inf when no agent is on a collision course."""
    if len(agent_states) == 0:
        return float("inf")
    p_e = ego_state[1:3]
    v_e = ego_state[4] * np.array([math.cos(ego_state[3]), math.sin(ego_state[3])])
    d = agent_states[:, :2] - p_e
    dist = np.linalg.norm(d, axis=1)
    v_a = agent_states[:, 3:4] * np.column_stack([np.cos(agent_states[:, 2]), np.sin(agent_states[:, 2])])
    v_rel = v_a - v_e
    closing = -(d * v_rel).sum(axis=1) / np.maximum(dist, 1e-9)
    ttc = np.full(len(dist), np.inf)
    ok = closing > TTC_CLOSING_EPS
    ttc[ok] = dist[ok] / closing[ok]
    return float(ttc.min())


def _ttc_and_collisions(states: np.ndarray, tracks: list[tuple[str, np.ndarray]],
                        times: np.ndarray, agents_grid: np.ndarray | None) -> tuple[float, bool]:
    ttc_min = float("inf")
    collision = False
    if agents_grid is None:
        agents_grid = interpolate_agent_grid(tracks, times)[1]
    if agents_grid.shape[0] == 0:
        return ttc_min, collision
    for i, st in enumerate(states):
        ag = agents_grid[:, i, :]
        d = np.linalg.norm(ag[:, :2] - st[1:3], axis=1)
        if d.min() < COLLISION_RADIUS_M:
            collision = True
        ttc_min = min(ttc_min, min_ttc_at_step(st, ag))
    return ttc_min, collision


def _completion_progress(states: np.ndarray, cl: np.ndarray, cum: np.ndarray) -> float:
    if len(cl) >= 2:
        s, _ = project_to_centerline(states[-1, 1:3][None, :], cl, cum)
        return float(s[0])
    d = np.diff(states[:, 1:3], axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def compute_metrics_row(scenario: dict, rule: dict, ml: dict) -> dict:
    R = np.asarray(rule["states"], dtype=float)
    M = np.asarray(ml["states"], dtype=float)
    rdec = [int(d) for d in rule["decisions"]]
    mdec = [int(d) for d in ml["decisions"]]
    n = min(len(R), len(M))

    th = R[:, 3]
    nrm = np.column_stack([-np.sin(th), np.cos(th)])
    e = M[:n, 1:3] - R[:n, 1:3]
    cross = (e * nrm).sum(axis=1)
    max_lateral = float(np.abs(cross).max())
    mean_lateral = float(np.abs(cross).mean())
    final_gap = float(np.linalg.norm(M[n - 1, 1:3] - R[n - 1, 1:3]))

    cl = np.asarray(scenario.get("centerline") or [], dtype=float).reshape(-1, 2)
    cum = arc_lengths(cl)
    tracks = []
    for a in scenario.get("agents", []):
        tr = states_to_array(a["track"])
        if len(tr) == 0:
            continue
        tr[:, 3] = np.unwrap(tr[:, 3])
        tracks.append((a["type"], tr))
    times = R[:n, 0]
    agents_grid = interpolate_agent_grid(tracks, times)[1]

    ttc_rule, collision_rule = _ttc_and_collisions(R[:n], tracks, times, agents_grid)
    ttc_ml, collision_ml = _ttc_and_collisions(M[:n], tracks, times, agents_grid)
    ttc_delta = (ttc_ml - ttc_rule) if math.isfinite(ttc_rule) and math.isfinite(ttc_ml) else float("nan")

    def max_jerk(X: np.ndarray) -> float:
        if len(X) < 2:
            return 0.0
        return float(np.abs(np.diff(X[:n, 5]) / DT).max())

    jerk_rule, jerk_ml = max_jerk(R), max_jerk(M)
    acc_rule = float(np.abs(R[:n, 5]).mean())
    acc_ml = float(np.abs(M[:n, 5]).mean())

    flips = [i for i in range(n) if rdec[i] != mdec[i]]
    flip_count = len(flips)
    if flips:
        i = flips[0]
        first_flip = float(R[i, 0])
        flip_kind = f"rule_{DECISION_NAMES[rdec[i]]}_ml_{DECISION_NAMES[mdec[i]]}"
    else:
        first_flip = None
        flip_kind = ""

    prog_rule = _completion_progress(R[:n], cl, cum)
    prog_ml = _completion_progress(M[:n], cl, cum)

    hard_brake_rule = bool((R[:n, 5] < -HARD_BRAKE_DECEL).any())
    hard_brake_ml = bool((M[:n, 5] < -HARD_BRAKE_DECEL).any())

    human = states_to_array(scenario.get("logged_ego") or [])
    n_h = min(n, len(human))
    ade_rule = ade_ml = float("nan")
    if n_h:
        ade_rule = float(np.linalg.norm(R[:n_h, 1:3] - human[:n_h, 1:3], axis=1).mean())
        ade_ml = float(np.linalg.norm(M[:n_h, 1:3] - human[:n_h, 1:3], axis=1).mean())

    score = (
        SCORE_WEIGHTS["lateral"] * math.tanh(max_lateral / 2.0)
        + SCORE_WEIGHTS["position"] * math.tanh(final_gap / 3.0)
        + SCORE_WEIGHTS["ttc"] * (math.tanh(abs(ttc_delta) / 2.0) if math.isfinite(ttc_delta) else 0.0)
        + SCORE_WEIGHTS["jerk"] * math.tanh(abs(jerk_ml - jerk_rule) / 5.0)
        + SCORE_WEIGHTS["accel"] * math.tanh(abs(acc_ml - acc_rule) / 1.0)
        + SCORE_WEIGHTS["decision"] * (flip_count / n)
        + SCORE_WEIGHTS["ade"] * (math.tanh(abs(ade_ml - ade_rule) / 2.0) if n_h else 0.0)
    )

    return {
        "scenario_id": scenario["id"],
        "tag": scenario["tag"],
        "max_lateral_deviation_m": max_lateral,
        "mean_lateral_deviation_m": mean_lateral,
        "final_position_gap_m": final_gap,
        "min_ttc_rule_s": ttc_rule if math.isfinite(ttc_rule) else float("nan"),
        "min_ttc_ml_s": ttc_ml if math.isfinite(ttc_ml) else float("nan"),
        "ttc_delta_s": ttc_delta,
        "max_jerk_rule": jerk_rule,
        "max_jerk_ml": jerk_ml,
        "jerk_delta": jerk_ml - jerk_rule,
        "mean_abs_accel_rule": acc_rule,
        "mean_abs_accel_ml": acc_ml,
        "decision_flip_count": flip_count,
        "first_flip_time_s": first_flip,
        "flip_kind": flip_kind,
        "completion_progress_rule_m": prog_rule,
        "completion_progress_ml_m": prog_ml,
        "hard_brake_rule": hard_brake_rule,
        "hard_brake_ml": hard_brake_ml,
        "collision_rule": collision_rule,
        "collision_ml": collision_ml,
        "ade_rule_vs_human_m": ade_rule,
        "ade_ml_vs_human_m": ade_ml,
        "divergence_score": float(score),
    }


def compute_metrics_rows(scenarios: list[dict], pairs: list[tuple[dict, dict]]) -> list[dict]:
    """Map over scenarios: each row depends only on its own scenario + pair,
    so the result is independent of input order."""
    rows = []
    for scenario, (rule, ml) in zip(scenarios, pairs):
        rows.append(compute_metrics_row(scenario, rule, ml))
    return rows
