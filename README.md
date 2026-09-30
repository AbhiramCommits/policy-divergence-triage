# policy-divergence-triage

Replay real logged driving scenarios (Argoverse 2 Motion Forecasting) through
both a learned ML policy and a deterministic rule-based C++ planner, then mine,
cluster, and triage where they diverge.

Current state: the C++ core (`pdt_core`), the deterministic rule planner, the
JSONL scenario format, the AV2 data pipeline, the learned ML policy, the
shadow-mode replay harness, the divergence metrics, the DuckDB query layer,
the cluster analysis, and the human review CLI are implemented. The regression
gate is still a stub.

## Repository layout

```
cpp/include/pdt/   types.hpp, planner.hpp, scenario.hpp
cpp/src/           planner.cpp (RulePlanner), scenario.cpp (JSONL I/O)
cpp/bindings/      pybind11 module `pdt_core`
cpp/tests/         GoogleTest suite (via FetchContent)
python/pdt/        policy.py (MLP + rollout), train_policy.py, harness.py (pdt-shadow),
                   metrics.py, query.py (pdt-query), cluster.py (pdt-cluster),
                   review.py (pdt-review), geom.py; gate.py stub
scenarios/         fetch_av2.py, convert_av2.py, manifest.txt
labels/            cluster_labels.yaml (human-editable, version-controlled)
tests/             pytest suite + procedurally generated 20-scenario fixture for CI
CMakeLists.txt     builds pdt_core lib, pdt_core extension, pdt_tests
pyproject.toml     scikit-build-core packaging (pip install -e .)
Dockerfile         Linux build + test image (Python 3.11)
README.md / NOTICE attribution for Argoverse 2 (CC BY-NC-SA 4.0)
```

## Stack

C++17, CMake 3.20+, pybind11, Python 3.11, `av2` (Argoverse 2 API), numpy,
pandas, pyarrow (Parquet), duckdb, scikit-learn, PyTorch, pytest, Docker, Linux.

`pip install -e ".[pipeline]"` installs the data/ML libraries; the core build
does not require them.

## Build and test

Prerequisites: a C++17 toolchain, CMake >= 3.20, Python 3.11 with dev headers,
network access (pybind11/GoogleTest/nlohmann-json are fetched by CMake).

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
pip install -e .
```

Or in Docker:

```sh
docker build -t pdt .
```

## Core C++ (`pdt_core`)

Data model (`cpp/include/pdt/types.hpp`):

- `State { t, x, y, heading, v, a }`
- `Agent { id, type, track }`, `type` in `{vehicle, pedestrian, cyclist}`
- `Scenario { id, tag, ego_init, agents, centerline, speed_limit, logged_ego }`;
  `logged_ego` is the logged human-driver trajectory (ground truth for training
  and "vs human" metrics).
- `Trajectory { scenario_id, source, states, decisions }` — `decisions` is
  parallel to `states` so decision flips are observable downstream.
- `Decision { FOLLOW, YIELD, ASSERT, STOP }`

### RulePlanner

Fixed 0.1 s step, 8.0 s horizon, fully deterministic (no randomness, no
uninitialized state). Each step emits a decision alongside the state:

- **Lateral**: pure-pursuit centerline tracking, lookahead
  `clamp(1.0 * v, 3.0, 15.0)` m, yaw rate `2 * v * sin(alpha) / lookahead`.
- **Longitudinal**: IDM car-following on the nearest in-lane leader, desired
  speed = `speed_limit`, time headway 1.6 s, minimum gap 2.0 m, max accel
  1.5 m/s^2, comfort decel 2.0 m/s^2. Commanded accel is clamped to
  `[-comfort_decel, max_accel]` and never leaves those bounds.
- **Yield logic**: for each crossing agent, predict the agent forward at
  constant velocity and find its conflict point on the ego path. Compute
  `t_ego` and `t_agent` (times to the conflict point). If
  `|t_ego - t_agent| < 2.0 s` the planner **yields** (decelerates to stop
  before the conflict point, holding the yield while the conflict is pending);
  otherwise it **asserts** (holds speed). While stopped for a pending conflict
  the decision is `STOP`; otherwise `FOLLOW`.
- `ASSERT` means a crossing conflict exists but the margin is large enough that
  the planner holds its current speed instead of yielding.

`PlannerConfig` carries every threshold; construct it from a Python dict:

```python
import pdt_core
planner = pdt_core.RulePlanner(pdt_core.PlannerConfig({
    "yield_time_margin": 1.5, "speed_limit...": ...}))
traj = planner.plan(scenario)
```

| Config field             | Default | Meaning                                        |
|--------------------------|---------|------------------------------------------------|
| `dt`                     | 0.1     | step (s)                                       |
| `horizon`                | 8.0     | horizon (s)                                    |
| `lookahead_gain/min/max` | 1.0/3.0/15.0 | pure-pursuit lookahead              |
| `idm_time_headway`       | 1.6     | IDM headway (s)                                |
| `idm_min_gap`            | 2.0     | IDM min gap (m)                                |
| `idm_max_accel`          | 1.5     | IDM max accel (m/s^2)                          |
| `idm_comfort_decel`      | 2.0     | comfort decel (m/s^2)                          |
| `idm_delta`              | 4.0     | IDM exponent                                   |
| `yield_time_margin`      | 2.0     | yield if \|t_ego - t_agent\| below this (s)    |
| `yield_stop_buffer`      | 2.0     | stop this far before the conflict point (m)    |
| `yield_stop_eps`         | 0.05    | min remaining distance floor for stop decel    |
| `conflict_radius`        | 2.5     | max lateral distance for a crossing (m)        |
| `agent_predict_horizon`  | 8.0     | constant-velocity prediction window (s)        |
| `min_crossing_speed`     | 0.2     | ignore near-stationary agents (m/s)            |
| `min_cross_angle`        | 0.35    | min path-crossing angle (rad)                  |
| `ttc_min_speed`          | 1.0     | floor on ego speed in t_ego (m/s)              |
| `stop_speed`             | 0.05    | below this the ego counts as stopped (m/s)     |
| `vehicle_length`         | 4.5     | subtracted from leader gap (m)                 |
| `lane_half_width`        | 2.5     | lateral window for "in-lane leader" (m)        |

### Scenario format

`scenarios/logs.jsonl`: one JSON object per line, matching `Scenario`:

```json
{"id":"00a0ffd7-...","tag":"left_turn","ego_init":{"t":0,"x":0,"y":0,"heading":0,"v":10,"a":0},
 "agents":[{"id":1,"type":"pedestrian","track":[{"t":0,"x":30,"y":4.5,"heading":-1.57,"v":1.5,"a":0},...]}],
 "centerline":[[0,0],[5,0],...],"speed_limit":11.18,"logged_ego":[{...},...]}
```

## Data pipeline (Argoverse 2)

AV2 data is large; the raw data and `scenarios/logs.jsonl` are gitignored.
Only the scripts and `scenarios/manifest.txt` are committed.

1. `scenarios/fetch_av2.py` — downloads a fixed, seeded subset of 2,000
   scenario folders from the public bucket
   (`s5cmd --no-sign-request` against
   `s3://argoverse/datasets/av2/motion-forecasting/val/`). It lists the bucket
   (metadata only), selects 2,000 folders with a fixed seed, records them in
   `scenarios/manifest.txt`, and copies only those folders to
   `scenarios/av2_raw/val/<id>/`.
2. `scenarios/convert_av2.py` — converts each scenario with the `av2` API:
   - ego = the AV (focal) track; agents = other vehicles/pedestrians/cyclists
     (`ObjectType` -> `{vehicle, pedestrian, cyclist}`; unknown types dropped);
   - all geometry localized to the ego frame (ego start at origin, heading 0);
   - centerline = the lane centerline(s) the logged ego actually traversed
     (lanes within 2.5 m of the ego track, stitched in traversal order);
   - the logged ego track is kept as `logged_ego` (human ground truth);
   - full logged tracks are kept for all agents (observed + future windows are
     both real logged data, so the replay stays on ground truth);
    - writes `scenarios/logs.jsonl` and prints the tag distribution.

Full pipeline:

```sh
python scenarios/fetch_av2.py                 # ~0.5 GB into scenarios/av2_raw/
python scenarios/convert_av2.py               # -> scenarios/logs.jsonl
python -m pdt.train_policy                    # 70% train -> artifacts/policy.pt
pdt-shadow --scenarios scenarios/logs.jsonl   # held-out 30% -> parquet artifacts
pdt-query --query per_tag                     # per-tag divergence table
pdt-cluster --divergence artifacts/divergence.parquet
pdt-review                                     # label clusters -> labels/cluster_labels.yaml
```

### Speed limits

Argoverse 2 maps carry no posted speed limits. We apply a documented
per-lane-type default taken from the ego start lane's `lane_type`:

| lane_type | speed limit (m/s) | note            |
|-----------|-------------------|-----------------|
| VEHICLE   | 11.18             | 25 mph, urban   |
| BUS       | 11.18             | 25 mph          |
| BIKE      | 6.71              | 15 mph          |
| fallback  | 11.18             | 25 mph          |

### Tag heuristics (priority order, thresholds in `convert_av2.py`)

1. `left_turn` / `right_turn`: net unwrapped heading change over the ego track
   beyond +/- 35 deg.
2. `lane_change`: the signed lateral offset from the nearest traversed lane
   centerline crosses beyond +/- 1.5 m on both sides (a >= 3 m lateral shift)
   while heading change stays below 20 deg.
3. `ped_or_cyclist_interaction`: a pedestrian/cyclist track comes within 6.0 m
   of the ego track.
4. `lead_vehicle_braking`: a vehicle within 40 m ahead slows by more than
   3.0 m/s within a 2.0 s window.
5. `straight_through_intersection`: the ego traverses a map lane segment
   flagged `is_intersection`.
6. `other`.

## Tests

`cpp/tests/` (GoogleTest, run by ctest) proves, against the committed
20-scenario fixture in `tests/fixtures/scenarios.jsonl`:

- the planner is deterministic — planning the same scenario twice produces
  byte-identical trajectories (every state field and decision equal);
- commanded accel never leaves `[-comfort_decel, max_accel]` on any fixture;
- a yield always produces a stop before the conflict point
  (`fixture_ped_00`: pedestrian crossing; the ego emits YIELD, comes to a stop
  while the crossing is pending, and never crosses the conflict point while
  yielding/stopped);
- scenario JSONL round-trips through `load_scenarios`/`save_scenarios`.

`tests/test_metrics.py` and `tests/test_policy.py` (pytest) cover: metrics are
order-independent, a zero-divergence synthetic pair scores exactly 0.0, TTC
math matches hand-computed values on a two-agent fixture, ML rollout
determinism, and the post-hoc decision classifier.

Fixtures are procedurally generated by `tests/fixtures/generate_fixtures.py`
(pure stdlib, deterministic); CI never downloads AV2.

## ML policy (`python/pdt/policy.py`)

Small MLP: input = flattened ego state + the 6 nearest agents' relative
pose/velocity + 10 centerline points ahead of the ego (ego frame); hidden
256-256 ReLU; output = 80 values = 40 steps x (accel, steer_rate). Inference is
deterministic (`model.eval()`, `torch.no_grad()`, fixed seed).

- **Training** (`python/pdt/train_policy.py`): behavior-clones the real logged
  human ego trajectories (`logged_ego`) from the AV2 scenarios — never a tuned
  copy of the rule planner or hand-written expert behavior. Split by scenario
  ID (sha256, 70% train / 30% held-out); all downstream analysis runs on the
  held-out 30% only. Saves `artifacts/policy.pt` and logs train/val loss plus
  open-loop ADE/FDE (`artifacts/train_log.jsonl`). Whatever the policy does
  better or worse than the planner is the finding; neither system is tuned
  toward an expected result.
- **Rollout**: closed-loop, 0.1 s step, 8 s horizon, re-query the model every
  step, apply only the first action (receding horizon), integrate with a
  kinematic bicycle model (`wheelbase 2.8 m`, steer-angle/rate limits). Same
  `Trajectory` shape as the C++ side.
- **Decisions**: the same enum is derived post-hoc from the ML rollout by a
  kinematic classifier on the speed profile + gap to the nearest in-lane leader
  (`classify_decisions` in `policy.py`), so ML decisions are comparable to the
  planner's.

### Known limitation

Other agents replay their logged tracks (non-reactive) for both systems. The
rule planner and the ML policy do not simulate agent responses; both see the
same frozen logged world, so divergences come from the policies themselves,
not from differing agent behavior.

## Shadow harness (`pdt-shadow`)

`python/pdt/harness.py` loads scenarios via `pdt_core.load_scenarios`, then for
each scenario runs the rule planner (pybind11) and the ML policy on identical
initial conditions, in the same process. Scenarios are processed with
multiprocessing over a fixed chunk order so output row order is stable. Writes:

- `artifacts/trajectories.parquet` — paired trajectories, long format
  (`scenario_id, source, step, t, x, y, heading, v, a, decision, decision_name`);
- `artifacts/divergence.parquet` — one divergence row per scenario.

```sh
python -m pdt.train_policy --scenarios scenarios/logs.jsonl
pdt-shadow --scenarios scenarios/logs.jsonl --checkpoint artifacts/policy.pt
```

## Divergence metrics (`python/pdt/metrics.py`)

One row per scenario in `artifacts/divergence.parquet`:

`scenario_id, tag, max_lateral_deviation_m, mean_lateral_deviation_m,
final_position_gap_m, min_ttc_rule_s, min_ttc_ml_s, ttc_delta_s, max_jerk_rule,
max_jerk_ml, jerk_delta, mean_abs_accel_rule, mean_abs_accel_ml,
decision_flip_count, first_flip_time_s, flip_kind, completion_progress_rule_m,
completion_progress_ml_m, hard_brake_rule, hard_brake_ml, collision_rule,
collision_ml, ade_rule_vs_human_m, ade_ml_vs_human_m, divergence_score`

Every formula (lateral deviation, TTC, jerk, ADE vs the logged human, hard
brake at decel > 4.0 m/s^2, collision at < 2.0 m, the weighted normalized
`divergence_score`) is documented in the `python/pdt/metrics.py` module
docstring.

## DuckDB queries (`pdt-query`)

Runs SQL directly over the Parquet files (`divergence`, `trajectories` views).
Four canned queries: top-50 divergences by score (`top50`), divergence rate per
tag (`per_tag`), decision-flip breakdown (`flips`), hard-brake and collision
comparison (`hard_brake`). Arbitrary SQL via `--sql`.

```sh
pdt-query --divergence artifacts/divergence.parquet --query per_tag
pdt-query --divergence artifacts/divergence.parquet --sql "SELECT ..."
```

## Cluster analysis (`pdt-cluster`)

`python/pdt/cluster.py` takes `artifacts/divergence.parquet`, filters rows
strictly above a configurable `divergence_score` threshold (default 0.5), and
clusters the divergence signatures. Feature vector: 9 normalized numeric
metric columns (lateral/final-gap/TTC-delta/jerk-delta/accel-delta/flip-count/
progress-delta/ADE-delta; NaN TTC deltas imputed as 0), a one-hot of the first
decision flip kind (12 fixed rule/ML pairs), and a one-hot of the scenario tag
(7 fixed tags). Numeric columns are standardized (StandardScaler); one-hots are
appended unscaled. Both KMeans (k swept 3..12, chosen by silhouette) and DBSCAN
(eps swept) run; the full sweep is reported and KMeans is persisted.

The fitted scaler + model persist to `artifacts/cluster_model.joblib`, so new
runs **assign** to existing clusters rather than re-fitting — cluster ids are
only meaningful relative to the persisted model (required for the regression
gate). Outputs: `artifacts/clusters.parquet` (scenario_id, cluster_id,
distance_to_centroid) and `artifacts/cluster_summary.json` (per cluster: size,
centroid in original metric units, 5 medoid exemplar scenario_ids, and an
auto-generated human-readable signature, e.g. "ML ASSERT where rule FOLLOW;
ML min TTC 0.6 s lower; ML progress +43.3 m").

```sh
pdt-cluster --divergence artifacts/divergence.parquet --threshold 0.5
```

## Human triage (`pdt-review`)

`python/pdt/review.py` walks clusters in descending size; per cluster it
prints the signature, centroid metrics, and an ASCII side-by-side plot of the
rule vs ML trajectory (x-y path and speed profile) for each of the 5 medoid
exemplars, then prompts for `label` (desirable / undesirable / mixed),
`rationale` (required), and an optional `ported_rule_hint`. Labels persist to
`labels/cluster_labels.yaml` (human-editable, version-controlled); re-running
shows existing labels and only prompts for unlabeled clusters. A matplotlib
PNG per cluster is exported to `artifacts/plots/`.

```sh
pdt-review --non-interactive --labels labels/cluster_labels.yaml   # CI / plots only
pdt-review                                                          # interactive triage
```

The committed `labels/cluster_labels.yaml` contains proposed labels for every
cluster from the held-out run; review and edit them before they are consumed
by the regression gate.

## Not yet implemented

The regression gate (`python/pdt/gate.py` stub).

## License and attribution

The Argoverse 2 Motion Forecasting Dataset is licensed CC BY-NC-SA 4.0
(non-commercial). See `NOTICE` for the required attribution. Dataset files are
never stored in this repository.
