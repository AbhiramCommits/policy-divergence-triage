#!/usr/bin/env python3
"""Behavior-clone the REAL logged human ego trajectories (`logged_ego`) from
the Argoverse 2 scenarios. The policy is never shown the rule planner or any
hand-written expert behavior: the reference is the human driver, and whatever
the trained policy does better or worse than the planner is the finding.

Split by scenario ID (sha256 of the id, 70% train / 30% held-out). All
downstream analysis runs on the held-out 30% only.

Outputs:
  artifacts/policy.pt     model checkpoint + feature config
  artifacts/train_log.jsonl  per-epoch train/val loss and open-loop ADE/FDE
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pdt.policy import (
    POLICY_CFG,
    MLPolicy,
    build_features,
    is_train_id,
    open_loop_ade_fde,
    scenario_arrays,
)

DT = POLICY_CFG["dt"]


def build_sample(scn: dict, t: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    logged = scn["logged_ego"]
    ego = logged[t]
    state = np.array([ego[0], ego[1], ego[2], ego[3], ego[4], ego[5], 0.0], dtype=float)
    x = build_features(scn, state, cfg)

    L = cfg["wheelbase"]
    y = np.zeros(2 * cfg["out_steps"], dtype=np.float32)
    for k in range(cfg["out_steps"]):
        v_k = max(logged[t + k, 4], 1e-3)
        v_k1 = max(logged[t + k + 1, 4], 1e-3)
        a = (v_k1 - v_k) / DT
        a = float(np.clip(a, cfg["accel_min"], cfg["accel_max"]))
        hdot_k = (logged[t + k + 1, 3] - logged[t + k, 3]) / DT
        hdot_k1 = (logged[t + k + 2, 3] - logged[t + k + 1, 3]) / DT
        d_k = math.atan(L * hdot_k / max(v_k, 1.0))
        d_k1 = math.atan(L * hdot_k1 / max(v_k1, 1.0))
        sr = float(np.clip((d_k1 - d_k) / DT, -cfg["steer_rate_max"], cfg["steer_rate_max"]))
        y[2 * k] = a / cfg["acc_scale"]
        y[2 * k + 1] = sr / cfg["sr_scale"]
    return x, y


def load_scenario_dicts(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() + 1) // 2))

    scenarios = load_scenario_dicts(args.scenarios)
    print(f"loaded {len(scenarios)} scenarios from {args.scenarios}")

    train_scns: list[dict] = []
    val_scns: list[dict] = []
    for sc in scenarios:
        scn = scenario_arrays(sc)
        (train_scns if is_train_id(sc["id"]) else val_scns).append(scn)
    print(f"split by scenario id: {len(train_scns)} train / {len(val_scns)} held-out")

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    for scn in train_scns:
        n_states = len(scn["logged_ego"])
        for t in range(n_states - 2 * POLICY_CFG["out_steps"] - 1):
            x, y = build_sample(scn, t, POLICY_CFG)
            X.append(x)
            Y.append(y)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32))
    print(f"train samples: {len(Xt)}")

    val_samples = []
    for scn in val_scns:
        n_states = len(scn["logged_ego"])
        for t in range(n_states - 2 * POLICY_CFG["out_steps"] - 1):
            val_samples.append(build_sample(scn, t, POLICY_CFG))
    if val_samples:
        Xv = torch.from_numpy(np.asarray([x for x, _ in val_samples], dtype=np.float32))
        Yv = torch.from_numpy(np.asarray([y for _, y in val_samples], dtype=np.float32))
    else:
        Xv, Yv = None, None

    model = MLPolicy(POLICY_CFG["input_dim"], POLICY_CFG["hidden"], 2 * POLICY_CFG["out_steps"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    gen = torch.Generator().manual_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(Xt, Yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n_batch = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n_batch += len(xb)
        train_loss = total / n_batch

        model.eval()
        val_loss = float("nan")
        if Xv is not None:
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xv), Yv).item())

        ades, fdes = [], []
        with torch.no_grad():
            for scn in val_scns:
                if len(scn["logged_ego"]) < POLICY_CFG["out_steps"] + 1:
                    continue
                ade, fde = open_loop_ade_fde(scn, model, POLICY_CFG)
                if math.isfinite(ade):
                    ades.append(ade)
                    fdes.append(fde)
        ade = float(np.mean(ades)) if ades else float("nan")
        fde = float(np.mean(fdes)) if fdes else float("nan")

        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ade_m": ade, "val_fde_m": fde}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:2d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"open-loop ADE={ade:.3f} m  FDE={fde:.3f} m")

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "state_dict": model.state_dict(),
        "cfg": POLICY_CFG,
        "seed": args.seed,
        "input_dim": POLICY_CFG["input_dim"],
        "out_dim": 2 * POLICY_CFG["out_steps"],
        "train_scenarios": len(train_scns),
        "val_scenarios": len(val_scns),
    }
    torch.save(ckpt, args.out)
    print(f"saved checkpoint to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=Path("scenarios/logs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/policy.pt"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/train_log.jsonl"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
