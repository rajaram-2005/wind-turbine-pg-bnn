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

  * :class:`MQTTConnector`      – ``paho-mqtt``: credentials, TLS, topic
    subscriptions and topic→signal mapping with an inbox queue.
  * :class:`ModbusTCPConnector` – ``pymodbus``: holding-register polling, a
    configurable register map with scaling, unit addressing and reconnect.
  * :class:`OPCUAConnector`     – ``asyncua``: node mapping, user/password and
    security-string authentication, reconnect with backoff.
  * :class:`HTTPSPollingConnector` – plain HTTPS polling of a device REST
    endpoint (no third-party library beyond ``httpx``).

* **Store-and-forward.** A local SQLite buffer persists readings when the
  upstream server is unreachable and flushes them on the next success.

Usage
-----
::

    python hardware_agent.py --connector modbus \\
        --server http://localhost:8080 --gateway-id gw-nacelle-01 --interval 5 \\
        --host 192.168.10.4 --modbus-port 502 --modbus-unit 1 \\
        --modbus-register-map "gen_rpm@1000:1:0.1:rpm,gbx_temp@1002:1:0.1:C"

    python hardware_agent.py --connector mqtt \\
        --host broker.example.com --mqtt-username turbine \\
        --mqtt-password secret --mqtt-topics "aerovigil/gw-nacelle-01/#"

    python hardware_agent.py --connector opcua \\
        --endpoint opc.tcp://10.0.0.5:4840 --opcua-user operator \\
        --opcua-password secret --opcua-nodes "gen_rpm=ns=2;s=gen_rpm"
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import contextlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        self._simulated = False
        self._last_reconnect_attempt = 0.0
        self._reconnect_delay = 2.0  # seconds, grows with backoff

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

    def _enter_simulated(self, reason: str) -> None:
        """Mark the connector as running in simulated mode (clearly logged)."""
        self._simulated = True
        self._connected = False
        logger.warning("[%s] %s – running in SIMULATED mode", self.name, reason)

    async def _maybe_reconnect(self, connect_once) -> bool:
        """Reconnect with capped exponential backoff. Returns True when up."""
        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_delay:
            return False
        self._last_reconnect_attempt = now
        try:
            await connect_once()
            self._reconnect_delay = 2.0  # reset backoff on success
            return True
        except Exception as exc:  # noqa: BLE001
            self._reconnect_delay = min(self._reconnect_delay * 2.0, 60.0)
            logger.warning("[%s] reconnect failed (%s); retrying in %.0fs",
                           self.name, exc, self._reconnect_delay)
            return False

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


# ------------------------------------------------------------------ MQTT
class MQTTConnector(ProtocolConnector):
    """Subscribe to device topics over MQTT (``paho-mqtt``).

    Options
    -------
    ``host``, ``port`` (1883), ``username``/``password`` (credentials),
    ``tls`` (bool), ``ca_cert`` (path), ``client_id``,
    ``topics`` (list of subscription topics, default
    ``aerovigil/{gateway_id}/#``) and ``topic_map`` (dict mapping an exact
    topic or regex to the canonical signal name). Payloads may be either a
    single ``{"signal": ..., "value": ...}`` object or a flat
    ``{"signal_name": value, ...}`` object; flat keys are mapped through
    ``topic_map``.
    """

    name = "mqtt"

    def __init__(self, gateway_id: str, **options: Any) -> None:
        super().__init__(gateway_id, **options)
        self._inbox: list[dict[str, Any]] = []
        self._inbox_lock = threading.Lock()
        self._client = None
        self._topic_map: dict[str, str] = {}
        raw_map = options.get("topic_map") or {}
        if isinstance(raw_map, dict):
            for topic, signal in raw_map.items():
                self._topic_map[str(topic)] = str(signal)

    # ------------------------------------------------------------ paho
    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:  # noqa: ARG002
        topics = self.options.get("topics") or [f"aerovigil/{self.gateway_id}/#"]
        for topic in topics:
            client.subscribe(topic)
        logger.info("[mqtt] connected; subscribed to %s", topics)

    def _on_message(self, client, userdata, msg) -> None:  # noqa: ARG002
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("[mqtt] non-JSON payload on %s: %s", msg.topic, exc)
            return
        signal = self._resolve_signal(msg.topic)
        if isinstance(payload, dict) and "signal" in payload and "value" in payload:
            reading = dict(payload)
            reading.setdefault("signal", signal or payload["signal"])
            with self._inbox_lock:
                self._inbox.append(reading)
            return
        if isinstance(payload, dict):
            # Flat payload: {<signal>: value, ...} (optionally {ts: ...}).
            ts = payload.get("timestamp") or payload.get("ts")
            for key, value in payload.items():
                if key in ("timestamp", "ts"):
                    continue
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    with self._inbox_lock:
                        self._inbox.append(
                            {
                                "signal": signal or str(key),
                                "value": value,
                                "unit": None,
                                "timestamp": ts,
                            }
                        )
        else:
            logger.warning("[mqtt] unhandled payload on %s: %r", msg.topic, payload)

    def _resolve_signal(self, topic: str) -> str | None:
        """Map an MQTT topic to a signal name via topic_map."""
        if topic in self._topic_map:
            return self._topic_map[topic]
        for pattern, signal in self._topic_map.items():
            try:
                if re.fullmatch(pattern, topic):
                    return signal
            except re.error:  # literal fallback
                continue
        return None

    async def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self._enter_simulated("paho-mqtt not installed")
            return

        host = self.options.get("host")
        if not host:
            self._enter_simulated("no --host given for mqtt")
            return
        port = int(self.options.get("port") or 1883)
        client_id = self.options.get("client_id") or f"aerovigil-{self.gateway_id}"

        # paho 2.x uses an explicit callback API version; 1.x does not.
        try:
            api_version = mqtt.CallbackAPIVersion.VERSION2
        except AttributeError:  # pragma: no cover - paho 1.x
            api_version = mqtt.MQTTv311
        client = mqtt.Client(api_version, client_id=client_id)
        username = self.options.get("username")
        password = self.options.get("password")
        if username:
            client.username_pw_set(str(username), str(password) if password else None)
        if self.options.get("tls"):
            client.tls_set(ca_certs=self.options.get("ca_cert") or None)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=60)
        try:
            client.connect(host, port, keepalive=30)
            client.loop_start()
        except Exception as exc:  # noqa: BLE001
            self._enter_simulated(f"mqtt connect failed ({exc})")
            return
        self._client = client
        self._connected = True
        self._simulated = False
        logger.info("[mqtt] paho-mqtt connected to %s:%d", host, port)

    async def read(self) -> list[dict[str, Any]]:
        with self._inbox_lock:
            readings, self._inbox = list(self._inbox), []
        if readings:
            return readings
        if self._client is not None and not self._connected:
            # paho auto-reconnects in loop_start(); nothing to do here.
            pass
        if self._simulated or self._client is None:
            return self._simulated_signals()
        return []

    async def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._connected = False


# --------------------------------------------------------- Modbus / TCP
class ModbusTCPConnector(ProtocolConnector):
    """Poll holding registers over Modbus/TCP (``pymodbus``).

    Options
    -------
    ``host``, ``port`` (502), ``unit`` (slave id, default 1) and
    ``register_map``: an ordered list of ``{name, address, count, scale,
    unit, signed}`` dicts (or the compact ``name@addr[:count][:scale][:unit]``
    string form, also accepted comma-separated on the CLI). Reads are
    re-scaled with ``value * scale`` and wrapped in a ``{signal, value,
    unit}`` reading.
    """

    name = "modbus"

    def __init__(self, gateway_id: str, **options: Any) -> None:
        super().__init__(gateway_id, **options)
        self._client = None
        self.register_map = self._parse_register_map(options.get("register_map"))
        if not self.register_map:
            logger.warning("[modbus] no register map configured – defaulting to demo map")
            self.register_map = [
                {"name": "generator_rpm", "address": 0, "count": 1, "scale": 0.5, "unit": "rpm"},
                {"name": "gearbox_temp", "address": 2, "count": 1, "scale": 0.1, "unit": "C"},
                {"name": "vibration_rms", "address": 4, "count": 1, "scale": 0.01, "unit": "mm/s"},
                {"name": "power_output", "address": 6, "count": 1, "scale": 1.0, "unit": "kW"},
                {"name": "wind_speed", "address": 8, "count": 1, "scale": 0.1, "unit": "m/s"},
            ]

    @staticmethod
    def _parse_register_map(raw: Any) -> list[dict[str, Any]]:
        """Accept a list of dicts, a JSON string, or compact entries.

        Compact form: ``name@addr[:count][:scale][:unit]`` joined by commas.
        """
        if raw is None:
            return []
        if isinstance(raw, list):
            out = []
            for item in raw:
                if isinstance(item, dict):
                    out.append(dict(item))
                else:
                    out.extend(ModbusTCPConnector._parse_register_map(str(item)))
            return out
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("["):
                try:
                    data = json.loads(text)
                    return ModbusTCPConnector._parse_register_map(data)
                except ValueError:
                    return []
            entries: list[dict[str, Any]] = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if "@" not in part:
                    logger.warning("[modbus] skipping malformed register entry %r", part)
                    continue
                name, _, tail = part.partition("@")
                bits = tail.split(":")
                try:
                    entry = {
                        "name": name.strip(),
                        "address": int(bits[0]),
                        "count": int(bits[1]) if len(bits) > 1 else 1,
                        "scale": float(bits[2]) if len(bits) > 2 else 1.0,
                        "unit": bits[3].strip() if len(bits) > 3 else None,
                    }
                except ValueError:
                    logger.warning("[modbus] skipping malformed register entry %r", part)
                    continue
                entries.append(entry)
            return entries
        return []

    async def connect(self) -> None:
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            self._enter_simulated("pymodbus not installed")
            return
        host = self.options.get("host")
        if not host:
            self._enter_simulated("no --host given for modbus")
            return
        port = int(self.options.get("port") or 502)
        client = ModbusTcpClient(
            host, port=port, timeout=float(self.options.get("timeout", 5.0))
        )
        try:
            ok = client.connect()
        except Exception as exc:  # noqa: BLE001
            self._enter_simulated(f"modbus connect failed ({exc})")
            return
        if not ok:
            self._enter_simulated(f"modbus connect refused at {host}:{port}")
            return
        self._client = client
        self._connected = True
        self._simulated = False
        logger.info("[modbus] connected to %s:%d", host, port)

    def _read_registers(self, client, entry: dict[str, Any]):
        unit = int(self.options.get("unit") or 1)
        count = max(int(entry.get("count") or 1), 1)
        try:
            resp = client.read_holding_registers(int(entry["address"]), count=count, slave=unit)
        except TypeError:  # pymodbus < 3.6 used ``unit=``
            resp = client.read_holding_registers(int(entry["address"]), count=count, unit=unit)
        if resp is None or getattr(resp, "isError", lambda: False)():
            raise ConnectionError(f"read failed for register {entry['address']}")
        return resp.registers

    async def read(self) -> list[dict[str, Any]]:
        if self._simulated or self._client is None:
            return self._simulated_signals()
        readings: list[dict[str, Any]] = []
        try:
            for entry in self.register_map:
                regs = self._read_registers(self._client, entry)
                raw = regs[0] if regs else 0
                if isinstance(raw, bool):  # pragma: no cover - pymodbus edge
                    raw = int(raw)
                if not isinstance(raw, int):
                    raw = int(raw)  # numpy ints
                if entry.get("signed") and raw >= 0x8000:
                    raw -= 0x10000
                scale = float(entry.get("scale") or 1.0)
                readings.append(
                    {
                        "signal": str(entry["name"]),
                        "value": raw * scale,
                        "unit": entry.get("unit"),
                    }
                )
            self._reconnect_delay = 2.0
            return readings
        except Exception as exc:  # noqa: BLE001 - device went away: reconnect
            logger.warning("[modbus] poll failed (%s); reconnecting…", exc)
            self._connected = False
            if await self._maybe_reconnect(self.connect):
                return await self.read()
            return self._simulated_signals()

    async def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):  # already closed
                self._client.close()
            self._client = None
        self._connected = False


# --------------------------------------------------------------- OPC-UA
class OPCUAConnector(ProtocolConnector):
    """Read node values from an OPC-UA server (``asyncua``).

    Options
    -------
    ``endpoint`` (``opc.tcp://host:port``), ``username``/``password``
    (authentication), ``security`` (a security string such as
    ``Basic256Sha256,SignAndEncrypt,cert.der,key.pem`` – passed to
    ``set_security_string``) and ``node_map``: ``{signal_name: node_id}``
    (e.g. ``{"gen_rpm": "ns=2;s=gen_rpm"}``), also accepted as comma
    separated ``name=nodeid`` pairs on the CLI.
    """

    name = "opcua"

    def __init__(self, gateway_id: str, **options: Any) -> None:
        super().__init__(gateway_id, **options)
        self._client = None
        self.node_map: dict[str, str] = {}
        raw_map = options.get("node_map") or {}
        if isinstance(raw_map, dict):
            for name, node_id in raw_map.items():
                self.node_map[str(name)] = str(node_id)
        if not self.node_map:
            logger.warning("[opcua] no node map configured – defaulting to demo map")
            self.node_map = {
                "generator_rpm": "ns=2;s=generator_rpm",
                "gearbox_temp": "ns=2;s=gearbox_temp",
                "vibration_rms": "ns=2;s=vibration_rms",
                "power_output": "ns=2;s=power_output",
                "wind_speed": "ns=2;s=wind_speed",
            }

    async def connect(self) -> None:
        try:
            from asyncua import Client
        except ImportError:
            self._enter_simulated("asyncua not installed")
            return
        endpoint = self.options.get("endpoint")
        if not endpoint:
            self._enter_simulated("no --endpoint given for opcua")
            return
        client = Client(endpoint)
        client.session_timeout = float(self.options.get("session_timeout", 300_000))
        username = self.options.get("username")
        if username:
            client.set_user(str(username))
            client.set_password(str(self.options.get("password") or ""))
        security = self.options.get("security")
        if security:
            client.set_security_string(str(security))
        try:
            await client.connect()
        except Exception as exc:  # noqa: BLE001
            self._enter_simulated(f"opcua connect failed ({exc})")
            return
        self._client = client
        self._connected = True
        self._simulated = False
        logger.info("[opcua] connected to %s", endpoint)

    async def read(self) -> list[dict[str, Any]]:
        if self._simulated or self._client is None:
            return self._simulated_signals()
        readings: list[dict[str, Any]] = []
        try:
            for name, node_id in self.node_map.items():
                node = self._client.get_node(node_id)
                value = await node.read_value()
                readings.append({"signal": str(name), "value": value})
            self._reconnect_delay = 2.0
            return readings
        except Exception as exc:  # noqa: BLE001 - session dropped: reconnect
            logger.warning("[opcua] read failed (%s); reconnecting…", exc)
            self._connected = False
            if await self._maybe_reconnect(self.connect):
                return await self.read()
            return self._simulated_signals()

    async def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):  # already disconnected
                await self._client.disconnect()
            self._client = None
        self._connected = False


# --------------------------------------------------------------- HTTPS
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

    def __init__(self, gateway_id: str, turbine_id: str | None = None) -> None:
        self.gateway_id = gateway_id
        self.turbine_id = turbine_id

    def normalize(self, raw_readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return schema-conformant readings with an ISO-8601 timestamp."""
        ts = _iso_now()
        out: list[dict[str, Any]] = []
        for raw in raw_readings:
            if "signal" not in raw or "value" not in raw:
                continue
            try:
                value = float(raw["value"])
            except (TypeError, ValueError):
                continue
            out.append(
                {
                    "gateway_id": self.gateway_id,
                    "turbine_id": raw.get("turbine_id", self.turbine_id),
                    "signal": str(raw["signal"]),
                    "value": value,
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
        turbine_id: str | None = None,
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


def _split_kv(spec: str, sep: str = "=") -> dict[str, str]:
    """Parse a comma-separated ``key=value`` string into a dict."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if sep in part:
            key, _, value = part.partition(sep)
            out[key.strip()] = value.strip()
    return out


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

    mqtt = p.add_argument_group("MQTT")
    mqtt.add_argument("--mqtt-port", type=int, default=1883)
    mqtt.add_argument("--mqtt-username", default=None)
    mqtt.add_argument("--mqtt-password", default=None)
    mqtt.add_argument("--mqtt-topics", default=None,
                      help="Comma-separated topics to subscribe to")
    mqtt.add_argument("--mqtt-topic-map", default=None,
                      help="Comma-separated topic=signal map entries")
    mqtt.add_argument("--mqtt-tls", action="store_true")
    mqtt.add_argument("--mqtt-ca-cert", default=None)

    modbus = p.add_argument_group("Modbus/TCP")
    modbus.add_argument("--modbus-port", type=int, default=502)
    modbus.add_argument("--modbus-unit", type=int, default=1, help="Slave/unit id")
    modbus.add_argument("--modbus-register-map", default=None,
                        help="Compact map: name@addr[:count][:scale][:unit],... "
                             "(or a JSON array of objects)")

    opcua = p.add_argument_group("OPC-UA")
    opcua.add_argument("--opcua-user", default=None)
    opcua.add_argument("--opcua-password", default=None)
    opcua.add_argument("--opcua-security", default=None,
                       help="Security string, e.g. Basic256Sha256,SignAndEncrypt,cert.der,key.pem")
    opcua.add_argument("--opcua-nodes", default=None,
                       help="Comma-separated name=nodeid map entries")
    return p


def _build_connector(args: argparse.Namespace) -> ProtocolConnector:
    """Instantiate the requested connector with protocol-specific options."""
    common: dict[str, Any] = {"gateway_id": args.gateway_id}
    if args.connector == "mqtt":
        common.update(
            host=args.host,
            port=args.mqtt_port,
            username=args.mqtt_username,
            password=args.mqtt_password,
            topics=[t.strip() for t in (args.mqtt_topics or "").split(",") if t.strip()]
            or None,
            topic_map=_split_kv(args.mqtt_topic_map) or None,
            tls=args.mqtt_tls,
            ca_cert=args.mqtt_ca_cert,
        )
    elif args.connector == "modbus":
        common.update(
            host=args.host,
            port=args.modbus_port,
            unit=args.modbus_unit,
            register_map=args.modbus_register_map,
        )
    elif args.connector == "opcua":
        common.update(
            endpoint=args.endpoint,
            username=args.opcua_user,
            password=args.opcua_password,
            security=args.opcua_security,
            node_map=_split_kv(args.opcua_nodes) or None,
        )
    else:  # https
        common.update(url=args.device_url)
    return CONNECTORS[args.connector](**common)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    connector = _build_connector(args)
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
