"""Durable SQLite-backed operational store for AeroVigil.

Single process-wide persistence layer for the state that previously lived
only in memory (ring buffers, twin registries, fleet aggregates) plus the
records that were already file/DB backed:

* ``telemetry``    – normalized gateway readings (append-only).
* ``assets``       – one row per turbine: latest telemetry, fleet-health
                     aggregate and advisory outcome (upserted on every stream).
* ``twin_states``  – digital-twin state snapshots (append-only history).
* ``reports``      – generated fleet/advisory reports (markdown body + JSON
                     metadata), keyed by kind so the API can serve the latest.
* ``imports``      – offline file imports (USB / cloud / API) with provenance.
* ``jobs``         – framework job queue/status/logs (managed by
                     :mod:`src.jobs.manager`; the store reports its stats).

SQLite is deliberately used with WAL mode and per-call connections so the
store is safe from FastAPI's threadpool and from multiple uvicorn workers
sharing one database file (each connection serializes on the file lock).
The database file lives in ``artifacts/aerovigil.sqlite3`` (gitignored) and
can be pointed elsewhere with the ``AEROVIGIL_STORE_DB`` environment
variable or the ``AV_STORE_DB`` env var (kept short for containers).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _REPO_ROOT / "artifacts" / "aerovigil.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    gateway_id TEXT NOT NULL,
    turbine_id TEXT,
    signal     TEXT NOT NULL,
    value      REAL NOT NULL,
    unit       TEXT,
    quality    TEXT,
    ts         TEXT NOT NULL,
    received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry (ts DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_turbine ON telemetry (turbine_id, ts DESC);

CREATE TABLE IF NOT EXISTS assets (
    turbine_id    TEXT PRIMARY KEY,
    gateway_id    TEXT,
    farm          TEXT,
    model_key     TEXT,
    status        TEXT,
    health_score  REAL,
    availability  REAL,
    predicted_rul_days REAL,
    epistemic_std      REAL,
    aleatoric_std      REAL,
    inspection_window_days REAL,
    last_seen     TEXT,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS twin_states (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id  TEXT NOT NULL,
    payload   TEXT NOT NULL,
    ts        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_twin_states_asset ON twin_states (asset_id, ts DESC);

CREATE TABLE IF NOT EXISTS reports (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind   TEXT NOT NULL,
    title  TEXT,
    body   TEXT NOT NULL,
    meta   TEXT,
    ts     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_kind ON reports (kind, ts DESC);

CREATE TABLE IF NOT EXISTS imports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT NOT NULL,
    content_type TEXT,
    size_bytes   INTEGER NOT NULL,
    source       TEXT NOT NULL,
    ts           REAL NOT NULL
);
"""


def _default_db_path() -> Path:
    """Resolve the store database path from the environment, else default."""
    for key in ("AEROVIGIL_STORE_DB", "AV_STORE_DB"):
        override = os.environ.get(key)
        if override:
            return Path(override)
    return _DEFAULT_DB


class Store:
    """SQLite-backed operational store (see module docstring for schema)."""

    def __init__(self, db_path: (Path | str | None) = None) -> None:
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------- helpers
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ----------------------------------------------------------- telemetry
    def record_telemetry(self, readings: list[dict[str, Any]]) -> int:
        """Persist a batch of normalized readings; returns the row count."""
        if not readings:
            return 0
        now = time.time()
        rows = [
            (
                r.get("gateway_id", ""),
                r.get("turbine_id"),
                str(r.get("signal", "")),
                float(r.get("value", 0.0)),
                r.get("unit"),
                r.get("quality", "good"),
                str(r.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))),
                now,
            )
            for r in readings
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO telemetry (gateway_id, turbine_id, signal, value, unit, quality, ts, received_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def latest_telemetry(
        self,
        limit: int = 100,
        *,
        turbine_id: (str | None) = None,
        signal: (str | None) = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent readings, newest first."""
        limit = max(1, min(int(limit), 5000))
        sql = "SELECT * FROM telemetry"
        clauses: list[str] = []
        params: list[Any] = []
        if turbine_id:
            clauses.append("turbine_id = ?")
            params.append(turbine_id)
        if signal:
            clauses.append("signal = ?")
            params.append(signal)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def telemetry_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])

    # -------------------------------------------------------------- assets
    def upsert_asset(self, asset: dict[str, Any]) -> None:
        """Insert or update the fleet-health row for one turbine."""
        turbine_id = asset.get("turbine_id")
        if not turbine_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO assets (turbine_id, gateway_id, farm, model_key, status,
                                    health_score, availability, predicted_rul_days,
                                    epistemic_std, aleatoric_std, inspection_window_days,
                                    last_seen, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turbine_id) DO UPDATE SET
                    gateway_id=excluded.gateway_id,
                    farm=excluded.farm,
                    model_key=excluded.model_key,
                    status=excluded.status,
                    health_score=excluded.health_score,
                    availability=excluded.availability,
                    predicted_rul_days=excluded.predicted_rul_days,
                    epistemic_std=excluded.epistemic_std,
                    aleatoric_std=excluded.aleatoric_std,
                    inspection_window_days=excluded.inspection_window_days,
                    last_seen=excluded.last_seen,
                    updated_at=excluded.updated_at
                """,
                (
                    turbine_id,
                    asset.get("gateway_id"),
                    asset.get("farm"),
                    asset.get("model_key"),
                    asset.get("status"),
                    asset.get("health_score"),
                    asset.get("availability"),
                    asset.get("predicted_rul_days"),
                    asset.get("epistemic_std"),
                    asset.get("aleatoric_std"),
                    asset.get("inspection_window_days"),
                    asset.get("last_seen"),
                    time.time(),
                ),
            )

    def get_assets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM assets ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def summarize_fleet(self) -> dict[str, Any]:
        """Fleet-level aggregate over the persisted assets table."""
        rows = self.get_assets()
        ruls = [a["predicted_rul_days"] for a in rows if a["predicted_rul_days"] is not None]
        health = [a["health_score"] for a in rows if a["health_score"] is not None]
        at_risk = sum(1 for r in ruls if r < 104.0)
        return {
            "n_assets": len(rows),
            "at_risk_count": at_risk,
            "mean_rul_days": round(sum(ruls) / len(ruls), 2) if ruls else None,
            "mean_health_score": round(sum(health) / len(health), 2) if health else None,
            "turbines": rows,
        }

    # ---------------------------------------------------------- twin state
    def record_twin_state(self, asset_id: str, payload: dict[str, Any]) -> None:
        """Append a digital-twin state snapshot."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO twin_states (asset_id, payload, ts) VALUES (?, ?, ?)",
                (asset_id, json.dumps(payload, default=str), payload.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )

    def latest_twin_state(self, asset_id: str) -> (dict[str, Any] | None):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM twin_states WHERE asset_id = ? ORDER BY id DESC LIMIT 1",
                (asset_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def twin_history(self, asset_id: str, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 2000))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM twin_states WHERE asset_id = ? ORDER BY id DESC LIMIT ?",
                (asset_id, limit),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # ------------------------------------------------------------- reports
    def record_report(
        self,
        kind: str,
        body: str,
        *,
        title: (str | None) = None,
        meta: (dict[str, Any] | None) = None,
    ) -> None:
        """Persist a generated report (markdown body + JSON metadata)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reports (kind, title, body, meta, ts) VALUES (?, ?, ?, ?, ?)",
                (kind, title, body, json.dumps(meta) if meta else None, time.time()),
            )

    def latest_report(self, kind: str) -> (dict[str, Any] | None):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reports WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)
            ).fetchone()
        if row is None:
            return None
        rec = dict(row)
        rec["meta"] = json.loads(rec["meta"]) if rec["meta"] else None
        return rec

    def list_reports(self, kind: (str | None) = None, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        sql = "SELECT * FROM reports"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            rec["meta"] = json.loads(rec["meta"]) if rec["meta"] else None
            out.append(rec)
        return out

    # ------------------------------------------------------------- imports
    def record_import(
        self,
        filename: str,
        content_type: (str | None),
        size_bytes: int,
        source: str,
    ) -> int:
        """Persist an offline file import; returns the row id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO imports (filename, content_type, size_bytes, source, ts) VALUES (?, ?, ?, ?, ?)",
                (filename, content_type, int(size_bytes), source, time.time()),
            )
            return int(cur.lastrowid)

    def list_imports(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM imports ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        """Row counts for every table (used by ``/api/system/stats``)."""
        counts: dict[str, int] = {}
        with self._connect() as conn:
            for table in ("telemetry", "assets", "twin_states", "reports", "imports"):
                try:
                    counts[table] = int(
                        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
                except sqlite3.Error:  # pragma: no cover - schema drift guard
                    counts[table] = 0
        return {
            "db_path": str(self._db_path),
            "journal_mode": "wal",
            "tables": counts,
        }


_SINGLETON: (Store | None) = None


def get_store() -> Store:
    """Return the process-wide :class:`Store` singleton."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Store()
    return _SINGLETON


def reset_store() -> None:
    """Drop the cached singleton (test isolation; next get_store re-reads env)."""
    global _SINGLETON
    _SINGLETON = None
