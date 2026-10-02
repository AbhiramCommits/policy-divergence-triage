#!/usr/bin/env python3
"""Generate the procedural fixture set used by the C++ unit tests and CI.

Writes tests/fixtures/scenarios.jsonl with 20 deterministic scenarios. Pure
stdlib, no network, no randomness: re-running always produces the same file.

Fixture ids:
  fixture_straight_{00..03}          free-flow straight, varying speed limits
  fixture_lead_{00..02}              lead vehicle braking hard (IDM clamp)
  fixture_ped_00                     pedestrian crossing, ego yields and stops
  fixture_ped_01                     cyclist crossing, ego yields (no full stop)
  fixture_ped_02                     crossing far away, ego asserts
  fixture_intersection_{00,01}       crossing vehicle at intersection (yield / assert)
  fixture_turn_left_00               left turn along a quarter-circle centerline
  fixture_turn_right_00              right turn along a quarter-circle centerline
  fixture_lanechange_{00,01}         ego starts offset, converges to centerline
  fixture_mixed_{00..03}             slow leader / s-curve / accel / sidewalk ped
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "scenarios.jsonl"

DT = 0.1
HORIZON = 8.0
N_STEPS = round(HORIZON / DT)
LOG_HORIZON = 11.0
LOG_STEPS = round(LOG_HORIZON / DT)


def state(t, x, y, heading, v, a=0.0):
    return {
        "t": round(t, 6),
        "x": round(x, 6),
        "y": round(y, 6),
        "heading": round(heading, 6),
        "v": round(v, 6),
        "a": round(a, 6),
    }


def track_of(fn, t0=0.0, t1=HORIZON, dt=DT):
    n = round((t1 - t0) / dt)
    return [state(t0 + i * dt, *fn(t0 + i * dt)) for i in range(n + 1)]


def integrate(start, n, dt, step_fn):
    t = 0.0
    x, y, h, v = start
    pts = []
    for _ in range(n + 1):
        a, h_dot = step_fn(t, x, y, h, v)
        pts.append(state(t, x, y, h, v, a))
        v2 = max(0.0, v + a * dt)
        h += h_dot * dt
        x += v2 * math.cos(h) * dt
        y += v2 * math.sin(h) * dt
        t += dt
        v = v2
    return pts


def straight_cl(x0=0.0, x1=60.0, y=0.0, step=5.0):
    n = round((x1 - x0) / step)
    return [[round(x0 + i * step, 6), y] for i in range(n + 1)]


def scenario(sid, tag, cl, v0, h0, limit, agents, logged):
    return {
        "id": sid,
        "tag": tag,
        "ego_init": state(0.0, 0.0, 0.0, h0, v0, 0.0),
        "agents": agents,
        "centerline": cl,
        "speed_limit": limit,
        "logged_ego": logged,
    }


def agent(aid, typ, track):
    return {"id": aid, "type": typ, "track": track}


def straight_free(i):
    limit = [11.18, 13.41, 8.94, 15.65][i]

    def step_fn(t, x, y, h, v):
        if v < limit:
            return 0.5, 0.0
        if v > limit:
            return -0.8, 0.0
        return 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, step_fn)
    return scenario(f"fixture_straight_{i:02d}", "other", straight_cl(), 10.0, 0.0, limit, [], logged)


def lead_braking(i):
    d0 = [18.0, 22.0, 26.0][i]

    def leader_fn(t):
        if t < 2.0:
            v, x, a = 10.0, d0 + 10.0 * t, 0.0
        else:
            v = max(0.0, 10.0 - 5.0 * (t - 2.0))
            x = d0 + 20.0 + 10.0 * (t - 2.0) - 2.5 * (t - 2.0) ** 2
            a = -5.0 if v > 0.0 else 0.0
        return x, 0.0, 0.0, v, a

    leader = track_of(leader_fn)

    def human_fn(t):
        if t < 3.0:
            v, x, a = 10.0, 10.0 * t, 0.0
        else:
            v = max(0.0, 10.0 - 4.0 * (t - 3.0))
            x = 30.0 + 10.0 * (t - 3.0) - 2.0 * (t - 3.0) ** 2
            a = -4.0 if v > 0.0 else 0.0
        return x, 0.0, 0.0, v, a

    logged = track_of(human_fn, t1=LOG_HORIZON)
    return scenario(
        f"fixture_lead_{i:02d}",
        "lead_vehicle_braking",
        straight_cl(),
        10.0,
        0.0,
        11.18,
        [agent(1, "vehicle", leader)],
        logged,
    )


def ped_yield():
    def ped_fn(t):
        if t < 2.0:
            return 30.0, 4.5 - 1.5 * t, -math.pi / 2, 1.5, 0.0
        return 30.0, 1.5 - 0.3 * (t - 2.0), -math.pi / 2, 0.3, 0.0

    ped = track_of(ped_fn, t1=12.0)

    def human_fn(t, x, y, h, v):
        if t < 7.0:
            return -1.8 if v > 0.0 else 0.0, 0.0
        return 1.5 if v < 11.18 else 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, human_fn)
    return scenario(
        "fixture_ped_00",
        "ped_or_cyclist_interaction",
        straight_cl(),
        10.0,
        0.0,
        11.18,
        [agent(2, "pedestrian", ped)],
        logged,
    )


def cyclist_yield():
    def cyc_fn(t):
        return 30.0, 4.0 - 1.0 * t, -math.pi / 2, 1.0, 0.0

    cyc = track_of(cyc_fn)

    def human_fn(t, x, y, h, v):
        return (-1.5 if t < 5.0 else 1.0), 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, human_fn)
    return scenario(
        "fixture_ped_01",
        "ped_or_cyclist_interaction",
        straight_cl(),
        10.0,
        0.0,
        11.18,
        [agent(2, "cyclist", cyc)],
        logged,
    )


def ped_assert():
    def ped_fn(t):
        return 30.0, 9.0 - 1.5 * t, -math.pi / 2, 1.5, 0.0

    ped = track_of(ped_fn)

    def human_fn(t, x, y, h, v):
        return 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, human_fn)
    return scenario(
        "fixture_ped_02",
        "ped_or_cyclist_interaction",
        straight_cl(),
        10.0,
        0.0,
        11.18,
        [agent(2, "pedestrian", ped)],
        logged,
    )


def crossing_vehicle(i):
    y0 = -20.0 if i == 0 else -48.0
    vc = 8.0 if i == 0 else 9.0

    def veh_fn(t):
        return 40.0, y0 + vc * t, math.pi / 2, vc, 0.0

    veh = track_of(veh_fn)

    def human_fn(t, x, y, h, v):
        if i == 0:
            return (-1.2 if t < 3.0 else 1.0), 0.0
        return 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, human_fn)
    return scenario(
        f"fixture_intersection_{i:02d}",
        "straight_through_intersection",
        straight_cl(),
        10.0,
        0.0,
        11.18,
        [agent(3, "vehicle", veh)],
        logged,
    )


def left_turn():
    r = 40.0
    cl = [[round(r * math.sin(th), 6), round(r * (1.0 - math.cos(th)), 6)] for th in [i * math.pi / 20 for i in range(11)]]

    def step_fn(t, x, y, h, v):
        a = 0.5 if v < 11.18 else 0.0
        h_dot = v / r if t < 6.28 else 0.0
        return a, h_dot

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, step_fn)
    return scenario("fixture_turn_left_00", "left_turn", cl, 10.0, 0.0, 11.18, [], logged)


def right_turn():
    r = 40.0
    cl = [
        [round(r * math.sin(th), 6), round(-r * (1.0 - math.cos(th)), 6)]
        for th in [i * math.pi / 20 for i in range(11)]
    ]

    def step_fn(t, x, y, h, v):
        a = 0.5 if v < 11.18 else 0.0
        h_dot = -v / r if t < 6.28 else 0.0
        return a, h_dot

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, step_fn)
    return scenario("fixture_turn_right_00", "right_turn", cl, 10.0, 0.0, 11.18, [], logged)


def lane_change(i):
    sgn = 1.0 if i == 0 else -1.0

    def step_fn(t, x, y, h, v):
        if t < 1.0:
            return 0.0, sgn * 0.25
        if t < 2.0:
            return 0.0, -sgn * 0.25
        return 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, step_fn)
    return scenario(f"fixture_lanechange_{i:02d}", "lane_change", straight_cl(), 10.0, 0.0, 11.18, [], logged)


def mixed_slow_leader():
    def slow_fn(t):
        return 15.0 + 2.0 * t, 0.0, 0.0, 2.0, 0.0

    slow = track_of(slow_fn)

    def human_fn(t, x, y, h, v):
        if t < 1.0:
            return 0.0, 0.0
        if t < 3.5:
            return -2.0, 0.0
        return 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 7.0), LOG_STEPS, DT, human_fn)
    return scenario(
        "fixture_mixed_00",
        "other",
        straight_cl(),
        7.0,
        0.0,
        11.18,
        [agent(1, "vehicle", slow)],
        logged,
    )


def mixed_s_curve():
    def cl_y(x):
        return 2.0 * math.sin(x / 12.0)

    cl = [[round(x, 6), round(cl_y(x), 6)] for x in [i * 2.5 for i in range(25)]]

    def step_fn(t, x, y, h, v):
        h_target = math.atan(math.cos(x / 12.0) / 6.0)
        h_dot = (h_target - h) * 2.0
        a = 0.5 if v < 11.18 else 0.0
        return a, h_dot

    logged = integrate((0.0, 0.0, 0.0, 8.0), LOG_STEPS, DT, step_fn)
    return scenario("fixture_mixed_01", "other", cl, 8.0, 0.0, 11.18, [], logged)


def mixed_accel():
    def step_fn(t, x, y, h, v):
        return (1.0 if v < 8.94 else 0.0), 0.0

    logged = integrate((0.0, 0.0, 0.0, 2.0), LOG_STEPS, DT, step_fn)
    return scenario("fixture_mixed_02", "other", straight_cl(), 2.0, 0.0, 8.94, [], logged)


def mixed_sidewalk_ped():
    def ped_fn(t):
        return 20.0 + 1.2 * t, 3.5, 0.0, 1.2, 0.0

    ped = track_of(ped_fn)

    def human_fn(t, x, y, h, v):
        return 0.0, 0.0

    logged = integrate((0.0, 0.0, 0.0, 10.0), LOG_STEPS, DT, human_fn)
    return scenario(
        "fixture_mixed_03",
        "other",
        straight_cl(),
        10.0,
        0.0,
        11.18,
        [agent(2, "pedestrian", ped)],
        logged,
    )


def main():
    scenarios = [
        *[straight_free(i) for i in range(4)],
        *[lead_braking(i) for i in range(3)],
        ped_yield(),
        cyclist_yield(),
        ped_assert(),
        crossing_vehicle(0),
        crossing_vehicle(1),
        left_turn(),
        right_turn(),
        lane_change(0),
        lane_change(1),
        mixed_slow_leader(),
        mixed_s_curve(),
        mixed_accel(),
        mixed_sidewalk_ped(),
    ]
    assert len(scenarios) == 20
    assert len({s["id"] for s in scenarios}) == 20
    with open(OUT, "w") as f:
        f.writelines(json.dumps(s, separators=(",", ":")) + "\n" for s in scenarios)
    print(f"wrote {len(scenarios)} scenarios to {OUT}")


if __name__ == "__main__":
    main()
