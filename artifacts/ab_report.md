# A/B report: `early_braking, intersection_caution`

Target cluster: **2** (desirable).

## Headline

- target-cluster undesirable rate: 0.246 -> 0.240 (drop of **0.6 pp** over the scenarios still assigned to the cluster)
- fixed-population rate (before-cluster scenario ids): 0.246 -> 0.214 (drop of **3.2 pp**)
- target-cluster rule collisions: 291 -> 236 (fixed population: 251)

## Global safety metrics (rule planner, held-out split)

| metric | before | after |
|---|---|---|
| collision count | 1128 | 1029 |
| hard-brake count | 0 | 0 |
| p95 max jerk (m/s^3) | 34.80 | 34.55 |
| mean progress (m) | 59.63 | 56.68 |
| mean min TTC (s) | 1.33 | 1.69 |

## Per-cluster undesirable rate

| cluster | label | size before/after | rate before | rate after | delta |
|---|---|---|---|---|---|
| 0 | mixed | 137/130 | 0.409 | 0.392 | -0.016 |
| 1 | mixed | 430/609 | 0.402 | 0.292 | -0.110 |
| 2 *target* | desirable | 1184/984 | 0.246 | 0.240 | -0.006 |
| 3 | undesirable | 917/979 | 0.663 | 0.576 | -0.087 |
| 4 | desirable | 1/1 | 0.000 | 0.000 | +0.000 |

## Override stats

- early_braking: activations 33935, vetoes 630, veto reasons {'veto_jerk': 391, 'veto_pedestrian_buffer': 239}
- intersection_caution: activations 2878, vetoes 19, veto reasons {'veto_pedestrian_buffer': 19}

## No-regression check (non-target clusters)

| cluster | label | rate before | rate after | delta |
|---|---|---|---|---|
| 1 | mixed | 0.402 | 0.292 | -0.110 |
| 3 | undesirable | 0.663 | 0.576 | -0.087 |
| 0 | mixed | 0.409 | 0.392 | -0.016 |
| 4 | desirable | 0.000 | 0.000 | +0.000 |
