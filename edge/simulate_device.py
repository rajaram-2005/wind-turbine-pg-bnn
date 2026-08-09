#!/usr/bin/env python3
"""Simulated ESP32/STM32 device — validates the MCU-to-cloud path.

Emits exactly the JSON batch that ``edge/esp32/aerovigil_telemetry.ino`` and
``edge/stm32/aerovigil_stm32_main.c`` send to ``POST /api/hardware/stream``,
so you can validate DNS, firewall, TLS, and the AeroVigil ingestion pipeline
before flashing any hardware.

Examples
--------
    python edge/simulate_device.py --server http://localhost:8080
    python edge/simulate_device.py --server http://localhost:8080 \
        --gateway-id esp32-gw-01 --turbine-id WTG-ESP-01 --interval 5 --count 3

Runs forever by default; ``--count N`` stops after N batches. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class SimulatedTurbine:
    """Random-walk telemetry around a healthy operating point."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.vib = 2.5
        self.gearbox_temp = 62.0
        self.generator_temp = 74.0
        self.wind = 9.0
        self.operating_hours = 31250.0

    def sweep(self) -> dict[str, float]:
        clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
        self.vib = clamp(self.vib + self._rng.uniform(-0.3, 0.3), 0.5, 12.0)
        self.gearbox_temp = clamp(self.gearbox_temp + self._rng.uniform(-2, 2), 45.0, 95.0)
        self.generator_temp = clamp(
            self.generator_temp + self._rng.uniform(-2, 2), 55.0, 110.0
        )
        self.wind = clamp(self.wind + self._rng.uniform(-0.8, 0.8), 2.0, 22.0)
        rpm = 900.0 + max(0.0, min(900.0, self.wind * 100.0 - 200.0)) / 900.0 * 900.0
        if self.wind < 3.0:
            power = 0.0
        elif self.wind >= 12.0:
            power = 2500.0
        else:
            power = 2500.0 * ((self.wind - 3.0) / 9.0) ** 3
        return {
            "vibration_rms": round(self.vib, 3),
            "gearbox_temp": round(self.gearbox_temp, 2),
            "generator_temp": round(self.generator_temp, 2),
            "generator_rpm": round(rpm, 1),
            "power_output": round(power, 1),
            "wind_speed": round(self.wind, 2),
            "operating_hours": round(self.operating_hours, 1),
        }


UNITS = {
    "vibration_rms": "mm/s",
    "gearbox_temp": "C",
    "generator_temp": "C",
    "generator_rpm": "rpm",
    "power_output": "kW",
    "wind_speed": "m/s",
    "operating_hours": "h",
}


def build_batch(gateway_id: str, turbine_id: str, device: SimulatedTurbine,
                hours_per_sample: float) -> dict:
    readings = []
    values = device.sweep()
    device.operating_hours += hours_per_sample
    ts = iso_now()
    for signal, value in values.items():
        readings.append(
            {
                "gateway_id": gateway_id,
                "turbine_id": turbine_id,
                "signal": signal,
                "value": value,
                "unit": UNITS[signal],
                "quality": "good",
                "timestamp": ts,
            }
        )
    return {"gateway_id": gateway_id, "readings": readings}


def post_batch(server: str, batch: dict, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        server.rstrip("/") + "/api/hardware/stream",
        data=json.dumps(batch).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="http://localhost:8080",
                        help="AeroVigil unified app origin (default %(default)s)")
    parser.add_argument("--gateway-id", default="esp32-gw-01")
    parser.add_argument("--turbine-id", default="WTG-ESP-01")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="seconds between batches (default %(default)s)")
    parser.add_argument("--count", type=int, default=0,
                        help="stop after N batches (0 = run forever)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = SimulatedTurbine(args.seed)
    print(f"[sim] {args.gateway_id} -> {args.server}/api/hardware/stream "
          f"(turbine {args.turbine_id}, every {args.interval:g}s)")

    sent = 0
    while args.count == 0 or sent < args.count:
        batch = build_batch(args.gateway_id, args.turbine_id, device,
                            args.interval / 3600.0)
        try:
            ack = post_batch(args.server, batch)
            assets = ack.get("assets") or []
            asset = assets[0] if assets else {}
            print(
                f"[sim] batch {sent + 1}: received={ack.get('received')} "
                f"source={asset.get('advisory_source')} "
                f"rul={asset.get('predicted_rul_days')}d "
                f"health={asset.get('health_score')}"
            )
        except urllib.error.HTTPError as exc:
            print(f"[sim] HTTP {exc.code}: {exc.read()[:200]!r}", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[sim] unreachable ({exc}); retrying…", file=sys.stderr)
        sent += 1
        if args.count == 0 or sent < args.count:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
