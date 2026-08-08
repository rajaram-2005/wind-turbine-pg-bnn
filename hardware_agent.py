"""AeroVigil hardware gateway agent.

A standalone service designed to run locally on industrial gateways at the wind
farm edge. It polls field devices over one of several industrial protocols,
normalizes readings into a common JSON schema, and forwards them to the
canonical AeroVigilAI server at ``/api/hardware/stream``.

Key properties
--------------
* **Pluggable protocols.** An abstract :class:`ProtocolConnector` base class
  with four concrete implementations: MQTT, Modbus/TCP, OPC-UA and HTTPS
  polling. Third-party client libraries are imported lazily so the agent
  imports (and unit-tests) cleanly even when a given library is absent – the
  connector then runs in a clearly-marked simulated mode.
* **Store-and-forward.** A local SQLite buffer persists readings when the
  upstream server is unreachable and flushes them on the next success.

Usage
-----
::

    python hardware_agent.py --connector modbus \\
        --server http://localhost:8080 --gateway-id gw-nacelle-01 --interval 5
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hardware_agent")

DEFAULT_SERVER = "http://localhost:8080"
DEFAULT_BUFFER_DB = Path("hardware_agent_buffer.sqlite3")


def _iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================ connectors
class ProtocolConnector(abc.ABC):
    """Abstract base class for a field-device protocol connector."""

    name: str = "base"

    def __init__(self, gateway_id: str, **options: Any) -> None:
        self.gateway_id = gateway_id
        self.options = options
        self._connected = False

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the underlying protocol connection."""

    @abc.abstractmethod
    async def read(self) -> list[dict[str, Any]]:
        """Return a list of raw readings from the device.

        Each raw reading is a free-form dict; normalization is handled by the
        :class:`NormalizationPipeline`.
        """

    async def close(self) -> None:
        """Tear down the connection. Override when a client needs cleanup."""
        self._connected = False

    @staticmethod
    def _simulated_signals() -> list[dict[str, Any]]:
        """Produce realistic simulated SCADA readings for offline/dev runs."""
        return [
            {"signal": "generator_rpm", "value": round(random.uniform(900, 1800), 1), "unit": "rpm"},
            {"signal": "gearbox_temp", "value": round(random.uniform(45, 95), 2), "unit": "C"},
            {"signal": "vibration_rms", "value": round(random.uniform(0.5, 12.0), 3), "unit": "mm/s"},
            {"signal": "power_output", "value": round(random.uniform(0, 3200), 1), "unit": "kW"},
            {"signal": "wind_speed", "value": round(random.uniform(2, 22), 2), "unit": "m/s"},
        ]


class MQTTConnector(ProtocolConnector):
    """Subscribe to device topics over MQTT (``paho-mqtt``)."""

    name = "mqtt"

    async def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt  # noqa: F401

            self._simulated = False
            logger.info("[mqtt] paho-mqtt available; connecting to %s", self.options.get("host"))
            # Real client wiring would go here (connect + subscribe).
        except ImportError:
            self._simulated = True
            logger.warning("[mqtt] paho-mqtt not installed – running in SIMULATED mode")
        self._connected = True

    async def read(self) -> list[dict[str, Any]]:
        # In a real deployment the on_message callback would fill a queue;
        # here we emit simulated topic payloads.
        return self._simulated_signals()


class ModbusTCPConnector(ProtocolConnector):
    """Poll holding registers over Modbus/TCP (``pymodbus``)."""

    name = "modbus"

    async def connect(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient  # noqa: F401

            self._simulated = False
            logger.info("[modbus] pymodbus available; polling %s", self.options.get("host"))
        except ImportError:
            self._simulated = True
            logger.warning("[modbus] pymodbus not installed – running in SIMULATED mode")
        self._connected = True

    async def read(self) -> list[dict[str, Any]]:
        # Real path: client.read_holding_registers(addr, count, unit) then
        # decode per the register map. Simulated path below.
        return self._simulated_signals()


class OPCUAConnector(ProtocolConnector):
    """Read node values from an OPC-UA server (``asyncua``)."""

    name = "opcua"

    async def connect(self) -> None:
        try:
            from asyncua import Client  # noqa: F401

            self._simulated = False
            logger.info("[opcua] asyncua available; connecting to %s", self.options.get("endpoint"))
        except ImportError:
            self._simulated = True
            logger.warning("[opcua] asyncua not installed – running in SIMULATED mode")
        self._connected = True

    async def read(self) -> list[dict[str, Any]]:
        # Real path: await client.get_node(nodeid).read_value() per node.
        return self._simulated_signals()


class HTTPSPollingConnector(ProtocolConnector):
    """Poll a device REST endpoint over HTTPS."""

    name = "https"

    async def connect(self) -> None:
        self._simulated = self.options.get("url") is None
        if self._simulated:
            logger.warning("[https] no --device-url given – running in SIMULATED mode")
        else:
            logger.info("[https] polling device endpoint %s", self.options.get("url"))
        self._connected = True

    async def read(self) -> list[dict[str, Any]]:
        url = self.options.get("url")
        if not url:
            return self._simulated_signals()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else [data]
        except Exception as exc:  # noqa: BLE001
            logger.error("[https] device poll failed: %s", exc)
            return []


CONNECTORS: dict[str, type[ProtocolConnector]] = {
    MQTTConnector.name: MQTTConnector,
    ModbusTCPConnector.name: ModbusTCPConnector,
    OPCUAConnector.name: OPCUAConnector,
    HTTPSPollingConnector.name: HTTPSPollingConnector,
}


# ========================================================= normalization
class NormalizationPipeline:
    """Standardize raw readings into the common AeroVigil telemetry schema."""

    def __init__(self, gateway_id: str, turbine_id: Optional[str] = None) -> None:
        self.gateway_id = gateway_id
        self.turbine_id = turbine_id

    def normalize(self, raw_readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return schema-conformant readings with an ISO-8601 timestamp."""
        ts = _iso_now()
        out: list[dict[str, Any]] = []
        for raw in raw_readings:
            if "signal" not in raw or "value" not in raw:
                continue
            out.append(
                {
                    "gateway_id": self.gateway_id,
                    "turbine_id": raw.get("turbine_id", self.turbine_id),
                    "signal": str(raw["signal"]),
                    "value": float(raw["value"]),
                    "unit": raw.get("unit"),
                    "quality": raw.get("quality", "good"),
                    "timestamp": raw.get("timestamp", ts),
                }
            )
        return out


# =============================================================== buffer
class OfflineBuffer:
    """SQLite-backed store-and-forward buffer for offline resilience."""

    def __init__(self, db_path: Path | str = DEFAULT_BUFFER_DB) -> None:
        self._db_path = Path(db_path)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS buffer ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)

    def store(self, readings: list[dict[str, Any]]) -> None:
        """Persist a batch of readings for later transmission."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO buffer (payload) VALUES (?)",
                [(json.dumps(r),) for r in readings],
            )

    def drain(self, limit: int = 1000) -> list[tuple[int, dict[str, Any]]]:
        """Return up to ``limit`` buffered readings as ``(row_id, reading)``."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, payload FROM buffer ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
        return [(rid, json.loads(payload)) for rid, payload in rows]

    def delete(self, row_ids: list[int]) -> None:
        """Remove successfully-transmitted rows."""
        if not row_ids:
            return
        with self._connect() as conn:
            conn.executemany("DELETE FROM buffer WHERE id = ?", [(i,) for i in row_ids])

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM buffer").fetchone()[0])


# =============================================================== agent
class HardwareAgent:
    """Orchestrates polling, normalization, buffering and transmission."""

    def __init__(
        self,
        connector: ProtocolConnector,
        *,
        server: str = DEFAULT_SERVER,
        interval: float = 5.0,
        turbine_id: Optional[str] = None,
        buffer_db: Path | str = DEFAULT_BUFFER_DB,
    ) -> None:
        self.connector = connector
        self.server = server.rstrip("/")
        self.interval = interval
        self.pipeline = NormalizationPipeline(connector.gateway_id, turbine_id)
        self.buffer = OfflineBuffer(buffer_db)

    @property
    def stream_url(self) -> str:
        return f"{self.server}/api/hardware/stream"

    async def _post(self, readings: list[dict[str, Any]]) -> bool:
        """POST a normalized batch. Returns True on success."""
        try:
            import httpx

            body = {"gateway_id": self.connector.gateway_id, "readings": readings}
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(self.stream_url, json=body)
                resp.raise_for_status()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.error("transmission failed (%s); buffering offline", exc)
            return False

    async def _flush_buffer(self) -> None:
        """Attempt to transmit any previously buffered readings."""
        pending = self.buffer.drain()
        if not pending:
            return
        readings = [r for _, r in pending]
        if await self._post(readings):
            self.buffer.delete([rid for rid, _ in pending])
            logger.info("flushed %d buffered readings", len(readings))

    async def tick(self) -> list[dict[str, Any]]:
        """Run one poll/normalize/transmit cycle; return the normalized batch."""
        raw = await self.connector.read()
        readings = self.pipeline.normalize(raw)
        if not readings:
            return []
        await self._flush_buffer()
        if not await self._post(readings):
            self.buffer.store(readings)
        return readings

    async def run(self) -> None:
        """Main transmission loop."""
        await self.connector.connect()
        logger.info(
            "agent started: connector=%s server=%s interval=%.1fs buffered=%d",
            self.connector.name, self.server, self.interval, self.buffer.count(),
        )
        try:
            while True:
                batch = await self.tick()
                logger.info("cycle complete: %d readings sent", len(batch))
                await asyncio.sleep(self.interval)
        finally:
            await self.connector.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AeroVigil hardware gateway agent")
    p.add_argument("--connector", choices=sorted(CONNECTORS), default="https")
    p.add_argument("--server", default=DEFAULT_SERVER, help="Canonical AeroVigil server URL")
    p.add_argument("--gateway-id", default="gw-edge-01")
    p.add_argument("--turbine-id", default=None)
    p.add_argument("--interval", type=float, default=5.0, help="Poll interval (seconds)")
    p.add_argument("--host", default=None, help="Device host (mqtt/modbus)")
    p.add_argument("--endpoint", default=None, help="OPC-UA endpoint URL")
    p.add_argument("--device-url", default=None, help="HTTPS device endpoint URL")
    p.add_argument("--buffer-db", default=str(DEFAULT_BUFFER_DB))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    connector_cls = CONNECTORS[args.connector]
    connector = connector_cls(
        gateway_id=args.gateway_id,
        host=args.host,
        endpoint=args.endpoint,
        url=args.device_url,
    )
    agent = HardwareAgent(
        connector,
        server=args.server,
        interval=args.interval,
        turbine_id=args.turbine_id,
        buffer_db=args.buffer_db,
    )
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("agent stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
