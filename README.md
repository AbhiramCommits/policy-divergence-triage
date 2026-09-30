# policy-divergence-triage

Replay real logged driving scenarios (Argoverse 2 Motion Forecasting) through
both a learned ML policy and a deterministic rule-based C++ planner, then mine,
cluster, and triage where they diverge.

Current state: the C++ core (`pdt_core`), the deterministic rule planner, the
JSONL scenario format, the AV2 data pipeline, and the GoogleTest suite are
implemented. The ML policy and the divergence metrics/clustering/review/gate
tooling are stubs and intentionally not implemented yet.

## Repository layout

```
cpp/include/pdt/   types.hpp, planner.hpp, scenario.hpp
cpp/src/           planner.cpp (RulePlanner), scenario.cpp (JSONL I/O)
cpp/bindings/      pybind11 module `pdt_core`
cpp/tests/         GoogleTest suite (via FetchContent)
python/pdt/        harness, metrics, clustering, review CLI, gate (stubs)
scenarios/         fetch_av2.py, convert_av2.py, manifest.txt
tests/fixtures/    procedurally generated 20-scenario fixture for CI (no AV2 downloads)
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
2. `lane_change`: lateral offset from the ego lane centerline varies by more
   than 3.0 m while heading change stays below 20 deg.
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

Fixtures are procedurally generated by `tests/fixtures/generate_fixtures.py`
(pure stdlib, deterministic); CI never downloads AV2.

## Not yet implemented

ML policy, divergence metrics, clustering, review CLI, and the regression gate
(see `python/pdt/*.py` stubs).

## License and attribution

The Argoverse 2 Motion Forecasting Dataset is licensed CC BY-NC-SA 4.0
(non-commercial). See `NOTICE` for the required attribution. Dataset files are
never stored in this repository.
