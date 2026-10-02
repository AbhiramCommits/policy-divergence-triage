#!/usr/bin/env python3
"""DuckDB query module (CLI: pdt-query): run SQL directly over the Parquet
artifacts. Ships four canned queries and accepts arbitrary SQL via --sql."""

from __future__ import annotations

import argparse

import duckdb
import pandas as pd

CANNED_QUERIES: dict[str, str] = {
    "top50": """
        SELECT * FROM divergence
        ORDER BY divergence_score DESC, scenario_id
        LIMIT 50
    """,
    "per_tag": """
        SELECT tag,
               COUNT(*) AS n,
               ROUND(AVG(divergence_score), 4) AS mean_score,
               ROUND(MAX(divergence_score), 4) AS max_score,
               ROUND(AVG(decision_flip_count), 2) AS avg_flips
        FROM divergence
        GROUP BY tag
        ORDER BY mean_score DESC, tag
    """,
    "flips": """
        SELECT flip_kind, COUNT(*) AS n
        FROM divergence
        WHERE flip_kind <> ''
        GROUP BY flip_kind
        ORDER BY n DESC, flip_kind
    """,
    "hard_brake": """
        SELECT SUM(hard_brake_rule) AS rule_hard_brakes,
               SUM(hard_brake_ml)  AS ml_hard_brakes,
               SUM(collision_rule) AS rule_collisions,
               SUM(collision_ml)   AS ml_collisions
        FROM divergence
    """,
}


def _escape(path: str) -> str:
    return path.replace("'", "''")


def connect(divergence: str, trajectories: str | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW divergence AS SELECT * FROM read_parquet('{_escape(divergence)}')")
    if trajectories:
        con.execute(f"CREATE OR REPLACE VIEW trajectories AS SELECT * FROM read_parquet('{_escape(trajectories)}')")
    return con


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divergence", type=str, default="artifacts/divergence.parquet")
    parser.add_argument("--trajectories", type=str, default="artifacts/trajectories.parquet")
    parser.add_argument("--query", type=str, choices=sorted(CANNED_QUERIES))
    parser.add_argument("--sql", type=str, help="arbitrary SQL; tables: divergence, trajectories")
    args = parser.parse_args()

    con = connect(args.divergence, args.trajectories)
    if args.sql:
        print(run_query(con, args.sql).to_string(index=False))
    elif args.query:
        print(run_query(con, CANNED_QUERIES[args.query]).to_string(index=False))
    else:
        for name in sorted(CANNED_QUERIES):
            print(f"=== {name} ===")
            print(run_query(con, CANNED_QUERIES[name]).to_string(index=False))
            print()


if __name__ == "__main__":
    main()
