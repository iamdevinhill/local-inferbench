"""SQLite storage layer for benchmark results.

Stores runs, generation results, and metric summaries in
~/.local-inferbench/results.db (configurable).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DB_PATH = Path.home() / ".local-inferbench" / "results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    adapter_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_metadata TEXT,
    profile TEXT NOT NULL,
    config TEXT,
    hardware_summary TEXT
);

CREATE TABLE IF NOT EXISTS generation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_category TEXT NOT NULL,
    result TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE,
    metrics TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quality_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
"""


class Storage:
    """SQLite-backed storage for benchmark runs and results."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)

    def save_run(
        self,
        adapter_name: str,
        model_id: str,
        profile: str,
        config: dict[str, Any],
        generation_results: list[dict[str, Any]],
        metrics: dict[str, Any],
        model_metadata: dict[str, Any] | None = None,
        hardware_summary: dict[str, Any] | None = None,
        scoring_summary: dict[str, Any] | None = None,
    ) -> int:
        """Save a complete benchmark run. Returns the run ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO runs (timestamp, adapter_name, model_id, model_metadata, profile, config, hardware_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                adapter_name,
                model_id,
                json.dumps(model_metadata) if model_metadata else None,
                profile,
                json.dumps(config),
                json.dumps(hardware_summary) if hardware_summary else None,
            ),
        )
        run_id = cursor.lastrowid
        assert run_id is not None

        for gr in generation_results:
            self._conn.execute(
                """INSERT INTO generation_results (run_id, prompt_text, prompt_category, result)
                   VALUES (?, ?, ?, ?)""",
                (
                    run_id,
                    gr["prompt_text"],
                    gr["prompt_category"],
                    json.dumps(gr["result"]),
                ),
            )

        self._conn.execute(
            "INSERT INTO metrics_summary (run_id, metrics) VALUES (?, ?)",
            (run_id, json.dumps(metrics)),
        )

        if scoring_summary:
            self._conn.execute(
                "INSERT INTO quality_scores (run_id, summary) VALUES (?, ?)",
                (run_id, json.dumps(scoring_summary)),
            )

        self._conn.commit()
        return run_id

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Retrieve a single run by ID."""
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None

        run = dict(row)
        run["model_metadata"] = json.loads(run["model_metadata"]) if run["model_metadata"] else None
        run["config"] = json.loads(run["config"]) if run["config"] else None
        run["hardware_summary"] = json.loads(run["hardware_summary"]) if run["hardware_summary"] else None

        gen_rows = self._conn.execute(
            "SELECT * FROM generation_results WHERE run_id = ?", (run_id,)
        ).fetchall()
        run["generation_results"] = [
            {
                "prompt_text": r["prompt_text"],
                "prompt_category": r["prompt_category"],
                "result": json.loads(r["result"]),
            }
            for r in gen_rows
        ]

        metrics_row = self._conn.execute(
            "SELECT metrics FROM metrics_summary WHERE run_id = ?", (run_id,)
        ).fetchone()
        run["metrics"] = json.loads(metrics_row["metrics"]) if metrics_row else None

        scores_row = self._conn.execute(
            "SELECT summary FROM quality_scores WHERE run_id = ?", (run_id,)
        ).fetchone()
        run["scoring_summary"] = json.loads(scores_row["summary"]) if scores_row else None

        return run

    def list_runs(
        self,
        model: str | None = None,
        adapter: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List runs with optional filters."""
        query = "SELECT id, timestamp, adapter_name, model_id, profile FROM runs WHERE 1=1"
        params: list[Any] = []
        if model:
            query += " AND model_id LIKE ?"
            params.append(f"%{model}%")
        if adapter:
            query += " AND adapter_name = ?"
            params.append(adapter)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_comparison(self, run_ids: list[int]) -> list[dict[str, Any]]:
        """Fetch multiple runs for side-by-side comparison."""
        runs = []
        for rid in run_ids:
            run = self.get_run(rid)
            if run:
                runs.append(run)
        return runs

    def delete_run(self, run_id: int) -> bool:
        """Delete a run and its associated data. Returns True if found."""
        cursor = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()
