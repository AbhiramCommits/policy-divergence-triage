import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pdt.policy import (
    FOLLOW,
    POLICY_CFG,
    STOP,
    YIELD,
    MLPolicy,
    classify_decisions,
    rollout,
)


def make_scenario():
    n = 81
    human = np.zeros((n, 6))
    for i in range(n):
        human[i] = [i * 0.1, i * 0.1 * 10.0, 0.0, 0.0, 10.0, 0.0]
    return {
        "id": "test_policy_scenario",
        "tag": "other",
        "ego_init": {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "v": 10.0, "a": 0.0},
        "agents": [],
        "centerline": [[x, 0.0] for x in range(0, 61, 5)],
        "speed_limit": 11.18,
        "logged_ego": human.tolist(),
    }


def build_model():
    torch.manual_seed(0)
    model = MLPolicy(POLICY_CFG["input_dim"], POLICY_CFG["hidden"], 2 * POLICY_CFG["out_steps"])
    return model


def test_rollout_is_deterministic():
    scenario = make_scenario()
    model = build_model()
    s1, d1 = rollout(scenario, model, POLICY_CFG, seed=0)
    s2, d2 = rollout(scenario, model, POLICY_CFG, seed=0)
    assert s1.shape == (81, 6)
    assert len(d1) == 81
    np.testing.assert_array_equal(s1, s2)
    assert d1 == d2


def test_decision_classifier_yield_and_stop():
    scenario = make_scenario()
    from pdt.geom import arc_lengths

    scn_arrays = {"cl": np.array(scenario["centerline"]), "cum": np.zeros(len(scenario["centerline"])), "tracks": []}
    scn_arrays["cum"] = arc_lengths(scn_arrays["cl"])

    states = np.zeros((3, 6))
    states[0] = [0.0, 0.0, 0.0, 0.0, 10.0, -3.0]
    states[1] = [0.1, 1.0, 0.0, 0.0, 9.5, -3.0]
    states[2] = [0.2, 1.9, 0.0, 0.0, 0.0, 0.0]
    decs = classify_decisions(states, scn_arrays)
    assert decs == [YIELD, YIELD, STOP]


def test_decision_classifier_follow_when_braking_behind_leader():
    scenario = make_scenario()
    n = 81
    leader = np.zeros((n, 6))
    for i in range(n):
        leader[i] = [i * 0.1, 22.0 + i * 0.1 * 9.0, 0.0, 0.0, 9.0, 0.0]
    scenario["agents"] = [{"id": 1, "type": "vehicle", "track": leader.tolist()}]
    from pdt.policy import scenario_arrays

    scn = scenario_arrays(scenario)
    states = np.zeros((3, 6))
    states[0] = [0.0, 0.0, 0.0, 0.0, 10.0, -3.0]
    states[1] = [0.1, 1.0, 0.0, 0.0, 9.7, -3.0]
    states[2] = [0.2, 2.0, 0.0, 0.0, 9.4, -3.0]
    decs = classify_decisions(states, scn)
    assert decs == [FOLLOW, FOLLOW, FOLLOW]
