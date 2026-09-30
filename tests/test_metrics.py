import math

import numpy as np
import pytest

from pdt.metrics import compute_metrics_row, min_ttc_at_step


def make_states(v=10.0, n=81, dt=0.1, heading=0.0):
    states = np.zeros((n, 6))
    for i in range(n):
        states[i] = [i * dt, i * dt * v * math.cos(heading), i * dt * v * math.sin(heading), heading, v, 0.0]
    return states


def make_scenario(agents=None, centerline=((0.0, 0.0), (60.0, 0.0)), speed_limit=11.18, human=None):
    n = 81
    if human is None:
        human = make_states(10.0)
    return {
        "id": "test_scenario",
        "tag": "other",
        "ego_init": {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "v": 10.0, "a": 0.0},
        "agents": agents or [],
        "centerline": [list(p) for p in centerline],
        "speed_limit": speed_limit,
        "logged_ego": human.tolist(),
    }


def make_agent(track):
    return {"id": 1, "type": "vehicle", "track": [list(s) for s in track]}


def make_pair(states, decisions=None):
    return {"states": states, "decisions": decisions if decisions is not None else [0] * len(states)}


def test_zero_divergence_synthetic_pair_scores_zero():
    scenario = make_scenario()
    states = make_states(10.0)
    row = compute_metrics_row(scenario, make_pair(states, [0] * 81), make_pair(states.copy(), [0] * 81))
    assert row["divergence_score"] == 0.0
    assert row["max_lateral_deviation_m"] == 0.0
    assert row["mean_lateral_deviation_m"] == 0.0
    assert row["final_position_gap_m"] == 0.0
    assert row["decision_flip_count"] == 0
    assert row["jerk_delta"] == 0.0
    assert row["ade_rule_vs_human_m"] == row["ade_ml_vs_human_m"]
    assert row["flip_kind"] == ""


def make_agent_track_straight(x0=30.0, v=5.0, n=81):
    track = np.zeros((n, 6))
    for i in range(n):
        track[i] = [i * 0.1, x0 + i * 0.1 * v, 0.0, 0.0, v, 0.0]
    return track


def test_metrics_order_independent():
    agent_a = make_agent(make_agent_track_straight(40.0, 6.0))
    scenario_a = make_scenario(agents=[agent_a], centerline=((0.0, 0.0), (60.0, 0.0)))
    states_a = make_states(10.0)
    ml_a = make_states(10.0)
    ml_a[:, 2] += np.linspace(0.0, 4.0, 81)
    row_a = compute_metrics_row(scenario_a, make_pair(states_a), make_pair(ml_a))

    agent_b = make_agent(make_agent_track_straight(50.0, 0.0))
    scenario_b = make_scenario(agents=[agent_b])
    states_b = make_states(8.0)
    ml_b = make_states(8.0)
    ml_b[:, 5] = -3.0
    row_b = compute_metrics_row(scenario_b, make_pair(states_b), make_pair(ml_b, [1] * 81))

    again_a = compute_metrics_row(scenario_a, make_pair(states_a), make_pair(ml_a))
    again_b = compute_metrics_row(scenario_b, make_pair(states_b), make_pair(ml_b, [1] * 81))

    assert row_a == again_a
    assert row_b == again_b
    assert row_a["scenario_id"] == scenario_a["id"]
    assert row_b["scenario_id"] == scenario_b["id"]


def test_ttc_hand_computed_two_agent_fixture():
    ego = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    stationary = np.array([[20.0, 0.0, 0.0, 0.0, 0.0]])
    assert min_ttc_at_step(ego, stationary) == pytest.approx(2.0)

    receding = np.array([[40.0, 0.0, 0.0, 12.0, 0.0]])
    assert math.isinf(min_ttc_at_step(ego, receding))

    approaching_slow = np.array([[50.0, 0.0, math.pi, 5.0, 0.0]])
    assert min_ttc_at_step(ego, approaching_slow) == pytest.approx(50.0 / 15.0)

    both = np.vstack([stationary, receding])
    assert min_ttc_at_step(ego, both) == pytest.approx(2.0)


def test_ttc_inside_full_metrics_row():
    n = 81
    agent_track = np.zeros((n, 6))
    for i in range(n):
        agent_track[i] = [i * 0.1, 20.0, 0.0, 0.0, 0.0, 0.0]
    scenario = make_scenario(agents=[make_agent(agent_track)])
    states = make_states(10.0)
    row = compute_metrics_row(scenario, make_pair(states), make_pair(states.copy()))
    # ego closes 1 m per step: ttc_i = (20 - i) / 10, min over the horizon is
    # 0.1 s at step 19 (1 m gap); at step 20 the gap is 0 so closing speed is 0.
    assert row["min_ttc_rule_s"] == pytest.approx(0.1)
    assert row["min_ttc_ml_s"] == pytest.approx(0.1)
    assert row["ttc_delta_s"] == pytest.approx(0.0)
    assert row["collision_rule"] is True
