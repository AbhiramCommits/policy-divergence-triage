# A/B report: `early_braking` override

Target cluster: **2** (desirable).

## Headline

- target-cluster undesirable rate: 0.340 -> 0.333 (drop of **0.7 pp** over the scenarios still assigned to the cluster)
- fixed-population rate (before-cluster scenario ids): 0.340 -> 0.289 (drop of **5.2 pp**)
- target-cluster rule collisions: 16 -> 12 (fixed population: 13)

## Global safety metrics (rule planner, held-out split)

| metric | before | after |
|---|---|---|
| collision count | 210 | 196 |
| hard-brake count | 0 | 0 |
| p95 max jerk (m/s^3) | 34.45 | 33.82 |
| mean progress (m) | 56.90 | 53.80 |
| mean min TTC (s) | 1.22 | 1.44 |

## Per-cluster undesirable rate

| cluster | label | size before/after | rate before | rate after | delta |
|---|---|---|---|---|---|
| 0 | mixed | 106/92 | 0.245 | 0.239 | -0.006 |
| 1 | mixed | 149/172 | 0.329 | 0.291 | -0.038 |
| 2 *target* | desirable | 47/36 | 0.340 | 0.333 | -0.007 |
| 3 | desirable | 63/56 | 0.444 | 0.411 | -0.034 |
| 4 | undesirable | 35/53 | 0.429 | 0.377 | -0.051 |
| 5 | undesirable | 2/2 | 0.000 | 0.000 | +0.000 |
| 6 | mixed | 103/110 | 0.680 | 0.582 | -0.098 |
| 7 | mixed | 6/6 | 0.500 | 0.500 | +0.000 |
| 8 | desirable | 1/1 | 1.000 | 1.000 | +0.000 |
| 9 | desirable | 21/16 | 0.095 | 0.062 | -0.033 |

## Override stats

- activations: 7301
- vetoes: 76
- veto reasons: {'veto_jerk': 75, 'veto_pedestrian_buffer': 1}

## No-regression check (non-target clusters)

| cluster | label | rate before | rate after | delta |
|---|---|---|---|---|
| 6 | mixed | 0.680 | 0.582 | -0.098 |
| 4 | undesirable | 0.429 | 0.377 | -0.051 |
| 1 | mixed | 0.329 | 0.291 | -0.038 |
| 3 | desirable | 0.444 | 0.411 | -0.034 |
| 9 | desirable | 0.095 | 0.062 | -0.033 |
| 0 | mixed | 0.245 | 0.239 | -0.006 |
| 5 | undesirable | 0.000 | 0.000 | +0.000 |
| 7 | mixed | 0.500 | 0.500 | +0.000 |
| 8 | desirable | 1.000 | 1.000 | +0.000 |
