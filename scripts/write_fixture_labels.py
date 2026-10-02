#!/usr/bin/env python3
"""Write an all-desirable labels file for the fixture pipeline (CI only).

Labels every cluster in artifacts/cluster_summary.json as desirable so that
pdt-ab can pick a target cluster and pdt-gate can run on the 20-scenario
fixture subset without human review. Written to a separate path so the
human-edited labels/cluster_labels.yaml is never clobbered.
"""

import json
from pathlib import Path

from pdt.review import save_labels

SUMMARY = Path("artifacts/fixture-pipeline/cluster_summary.json")
OUT = Path("labels/fixture_labels.yaml")


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    labels = {"version": 1, "clusters": {}}
    for c in summary["clusters"]:
        labels["clusters"][c["cluster_id"]] = {
            "label": "desirable",
            "rationale": "fixture pipeline auto-label (no human review)",
            "ported_rule_hint": None,
        }
    save_labels(OUT, labels)
    print(f"wrote {len(labels['clusters'])} all-desirable labels to {OUT}")


if __name__ == "__main__":
    main()
