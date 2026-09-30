"""Small numpy geometry helpers shared by the policy rollout and the divergence
metrics. All functions are pure and deterministic."""

from __future__ import annotations

import numpy as np


def arc_lengths(cl: np.ndarray) -> np.ndarray:
    cl = np.asarray(cl, dtype=float)
    if len(cl) < 2:
        return np.zeros(len(cl))
    d = np.diff(cl, axis=0)
    return np.concatenate([np.zeros(1), np.cumsum(np.linalg.norm(d, axis=1))])


def project_to_centerline(pts: np.ndarray, cl: np.ndarray, cum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=float)
    if len(cl) < 2:
        if len(cl) == 1:
            d = np.linalg.norm(pts - cl[0], axis=1)
            return np.zeros(len(pts)), d
        return np.zeros(len(pts)), np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    best_d = np.full(len(pts), np.inf)
    for i in range(len(cl) - 1):
        a = cl[i]
        ab = cl[i + 1] - a
        t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(pts - proj, axis=1)
        s = cum[i] + t * np.linalg.norm(ab)
        better = d < best_d
        best_s = np.where(better, s, best_s)
        best_d = np.where(better, d, best_d)
    return best_s, best_d


def point_at_arc(cl: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    if len(cl) == 0:
        return np.zeros(2)
    if len(cl) == 1 or cum[-1] <= 1e-9:
        return cl[0].copy()
    s = float(np.clip(s, 0.0, cum[-1]))
    for i in range(len(cl) - 1):
        if cum[i + 1] >= s:
            seg = cum[i + 1] - cum[i]
            t = (s - cum[i]) / seg if seg > 1e-12 else 0.0
            return cl[i] + t * (cl[i + 1] - cl[i])
    return cl[-1].copy()


def states_to_array(states: list[dict] | np.ndarray) -> np.ndarray:
    if states is None or len(states) == 0:
        return np.zeros((0, 6))
    if isinstance(states, np.ndarray):
        return states.astype(float)
    if isinstance(states[0], dict):
        return np.asarray(
            [[s["t"], s["x"], s["y"], s["heading"], s["v"], s["a"]] for s in states], dtype=float
        )
    return np.asarray(states, dtype=float)


def agent_state_at(track: np.ndarray, t: float) -> np.ndarray | None:
    if track is None or len(track) == 0:
        return None
    ts = track[:, 0]
    if t <= ts[0]:
        return track[0, 1:]
    if t >= ts[-1]:
        return track[-1, 1:]
    i = int(np.searchsorted(ts, t)) - 1
    i = min(i, len(track) - 2)
    t0, t1 = ts[i], ts[i + 1]
    f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return track[i, 1:] + f * (track[i + 1, 1:] - track[i, 1:])


def agents_at_t(tracks: list[tuple[str, np.ndarray]], t: float) -> np.ndarray:
    out = []
    for _typ, tr in tracks:
        s = agent_state_at(tr, t)
        if s is not None:
            out.append(s)
    if not out:
        return np.zeros((0, 5))
    return np.asarray(out, dtype=float)
