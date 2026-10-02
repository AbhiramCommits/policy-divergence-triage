# Design notes

## Why shadow-mode replay instead of online A/B

The ML policy is behavior-cloned from logged human driving, and the rule
planner is deterministic C++. The question we answer is "where do they diverge,
and which divergences are worth porting" — not "which policy wins at a specific
deployment metric". Two options for answering that:

- **Online A/B** (deploy both, split traffic, watch outcomes) requires a
  vehicle fleet, an ODD (operational design domain) carve-out, and a long time
  horizon to accumulate rare events. It also mixes policy differences with
  traffic/environment variance, and a bad ML behavior can only be discovered
  by exposing real users to it.
- **Shadow-mode replay** runs both policies on identical, replayed initial
  conditions with the same logged world. Every difference is attributable to
  the policies, every scenario is reproducible, and "before/after" for a rule
  change is a config flag on the same frozen data rather than a deployment.

We choose shadow mode because this repo's job is *diffing*, *clustering*, and
*triaging* behavior at scale — 2,927 held-out scenarios per run (the shadow
replay itself is 2 min 14 s on 8 cores; a full from-scratch container pipeline
was measured at 76 minutes at the previous 2,000-scenario manifest) — before
anything touches a vehicle. The known cost
(documented in README "Limitations") is that agents are non-reactive in both
systems, so interaction effects are not modeled.

## Why these divergence metrics

Each metric in `python/pdt/metrics.py` maps to a failure mode we can act on:

- **Lateral deviation / final position gap** — do the policies drive different
  paths (lane choice, corner cutting, off-route behavior)?
- **min TTC** (distance / closing speed vs every agent, min over the horizon) —
  the safety-relevant one; a rule change that improves progress but crushes
  TTC should be caught by the gate.
- **Jerk** — passenger comfort and a proxy for "smoothness of control", which
  distinguishes the ML's smoothed behavior from the planner's bang-bang
  acceleration.
- **Decision flips** (FOLLOW/YIELD/ASSERT/STOP at matched timesteps) — a
  human-interpretable, decision-level diff between the two systems.
- **Hard brake** (a < -4 m/s^2) and **collision** (< 2 m from any agent) —
  binary safety flags that define the per-cluster "undesirable rate" the A/B
  study and the gate optimize against.
- **ADE vs the logged human** — grounds both policies in the only ground-truth
  reference we have.
- **divergence_score** — a documented weighted, tanh-normalized combination
  (lateral 0.20, position 0.15, TTC 0.10, jerk 0.10, accel 0.10, decision
  0.20, ADE 0.15) used only to *rank and filter* scenarios for clustering; the
  gate never uses it.

## Why guarded overrides instead of retraining or deploying the net

Three candidate ways to act on a desirable divergence cluster:

1. **Retrain the ML policy toward it** — circular: the ML policy is the
   *instrument* that found the behavior; tuning it to the cluster's signature
   overfits the instrument to the finding and removes the independent
   comparison.
2. **Deploy the neural net into the production planner** — gives up the
   production planner's determinism, certifiability, and explicit guards for
   an improvement observed on 1,184 scenarios.
3. **Port the behavior as a guarded override** — encode what the cluster shows
   as an explicit, readable rule (e.g. `EarlyBrakingOverride`: brake early and
   wait behind a hard-braking or slow forward vehicle), active only when its
   preconditions hold, and vetoed by non-negotiable guards: predicted min TTC
   below `override_ttc_floor` (1.0 s), predicted jerk above
   `override_jerk_max` (20 m/s^3), or a pedestrian inside a 3 m buffer. The
   override is toggleable by name in `PlannerConfig`, so before/after is a
   config flag, not a code revert — which is exactly what the A/B study and
   the regression gate need.

We chose (3). The guard set is deliberately narrow and safety-shaped: an
override may be *more conservative* or *more aggressive* than the base
planner, but it may never violate those three floors, and every activation and
veto is logged with its reason (`RulePlanner.events()`), so the gate can audit
what the override actually did.

## What the gate protects against

`pdt-gate` compares the override-ON run against the override-OFF baseline on
the same held-out scenarios, assigned through the *persisted* cluster model
(never re-fitted, so cluster identities are stable across runs). It fails when
any of:

- collision count increases at all (hard floor — no tolerance),
- hard-brake count increases beyond `hard_brake_increase_tolerance`,
- p95 max jerk regresses beyond `jerk_p95_rel_tolerance`,
- any *non-target* cluster's undesirable rate rises beyond
  `non_target_undesirable_rate_rise_tolerance` (the fix must not leak damage
  into other clusters),
- a *new* cluster appears with size above `new_cluster_min_size` — unmodeled
  behavior, i.e. the override pushed scenarios into a divergence signature
  that did not exist before.

The headline metric is the fixed-population undesirable-rate drop in the
*target* cluster (the baseline cluster's scenario ids re-measured after the
change), so cluster re-assignment cannot game the headline.

## What would need to change for real vehicle logs

- **Agent reactivity**: real logs give you one ego's observations, not other
  agents' full future tracks. Replay would need a *prediction* layer for other
  agents (or the metric horizon shortened to what was observable at decision
  time).
- **Perception**: `Scenario.centerline` comes from HD map + ego localization
  here. With perception-only input, the planner's path model, the ML policy's
  centerline features, and the TTC/collision metrics all change definition.
- **Fleet calibration**: AV2 has no speed limits (we use documented
  per-lane-type defaults) and 10 Hz logged states. Real logs need per-road
  limits, actuation latency, and sensor frequency handling.
- **Behavior cloning covariate shift** (README "Limitations") becomes the
  dominant issue: the policy must be trained/evaluated with closed-loop DAgger
  or offline RL, not pure BC, before shadow-mode numbers are meaningful.
- **The gate's thresholds**: tolerance values in `gate_config.yaml` are
  dataset-scale; at fleet scale they should be set from a statistical
  argument over deployment volume, not hand-picked.
