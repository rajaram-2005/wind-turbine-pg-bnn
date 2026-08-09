# Edge microcontroller guide — connecting ESP32 / STM32 to AeroVigil

AeroVigil ingests field telemetry through one canonical endpoint on the unified
app (default `http://<server>:8080`):

```
POST /api/hardware/stream
```

Anything that can POST JSON — an ESP32, an STM32 with a network coprocessor, a
Raspberry Pi, or an industrial gateway — can join the platform. Every batch is
WAL-persisted, updates the asset's digital twin, computes a model advisory
(six-signal PG-BNN when all inputs are present), and refreshes the fleet.

```
 ┌────────────┐   WiFi / cellular    ┌───────────────────────────────┐
 │  ESP32 /   │ ──── HTTPS/HTTP ───▶ │  AeroVigil unified app :8080  │
 │  STM32     │  POST /api/hardware/ │  ┌─────────────────────────┐  │
 │  sensor    │        stream        │  │ SQLite store (WAL)      │  │
 │  node      │                      │  │ digital twin + advisory │  │
 └────────────┘                      │  │ fleet + MIKA/KAI mesh   │  │
                                     │  └─────────────────────────┘  │
                                     └───────────────────────────────┘
```

## 1. Wire format

```json
{
  "gateway_id": "esp32-gw-01",
  "readings": [
    {
      "gateway_id": "esp32-gw-01",
      "turbine_id": "WTG-ESP-01",
      "signal": "vibration_rms",
      "value": 2.51,
      "unit": "mm/s",
      "quality": "good",
      "timestamp": "2026-08-09T05:00:00.000Z"
    }
  ]
}
```

Signal names the platform understands (aliases fold onto canonical channels):

| Signal you send                | Folds onto              | Used by                                  |
|--------------------------------|-------------------------|------------------------------------------|
| `vibration_rms` / `vibration`  | twin vibration channel  | twin physics + six-signal PG-BNN         |
| `gearbox_temp` / `bearing_temp`| twin temperature        | twin physics + six-signal PG-BNN         |
| `generator_temp`               | —                       | six-signal PG-BNN                        |
| `generator_rpm` / `rpm`        | twin rpm                | twin physics                             |
| `power_output`                 | —                       | six-signal PG-BNN + power-curve chart    |
| `wind_speed`                   | —                       | six-signal PG-BNN + power-curve chart    |
| `operating_hours`              | —                       | six-signal PG-BNN                        |
| `oil_viscosity_cst`            | twin viscosity          | twin physics                             |
| `load_pct`                     | twin load               | twin physics                             |

Send **all seven bold signals** (`vibration_rms`, `gearbox_temp`,
`generator_temp`, `generator_rpm`, `power_output`, `wind_speed`,
`operating_hours`) and every batch earns a real model advisory
(`advisory_source: "stream-model-six-signal"`). Partial batches still update
the twin and fall back to the clearly-labeled demo heuristic.

## 2. Try it without hardware first

```bash
python edge/simulate_device.py --server http://localhost:8080 \
    --gateway-id esp32-gw-01 --turbine-id WTG-ESP-01 --interval 5
```

The simulator emits exactly the JSON the ESP32 firmware sends, so you can
validate DNS, firewall, TLS, and the ingestion pipeline before flashing
anything. Watch the asset appear in the console's Fleet, Digital Twin
(charts), and Hardware pages.

## 3. ESP32

Firmware: [`esp32/aerovigil_telemetry.ino`](esp32/aerovigil_telemetry.ino)
(Arduino IDE or PlatformIO — see [`esp32/platformio.ini`](esp32/platformio.ini)).

1. Set `WIFI_SSID`, `WIFI_PASS`, `AEROVIGIL_SERVER`, `GATEWAY_ID`,
   `TURBINE_ID` at the top of the sketch.
2. Flash an ESP32 (any variant with WiFi; ESP32-S3/C3 work too).
3. The node syncs NTP, samples sensors every `SAMPLE_SECONDS`, batches
   `BATCH_SIZE` sweeps, and POSTs them. Offline batches retry with capped
   exponential backoff; the serial monitor logs every ack, including the
   advisory RUL the cloud returned.

Suggested sensor wiring (analog channels through a simple conditioning
circuit; the sketch ships in SIMULATED mode and shows exactly where to plug
real readings):

| Measurement      | Typical sensor                          | ESP32 pin |
|------------------|------------------------------------------|-----------|
| Vibration RMS    | MEMS accel + envelope detector / SW-420  | GPIO 34 (ADC1_CH6) |
| Gearbox temp     | LM35 / NTC thermistor                    | GPIO 35 (ADC1_CH7) |
| Generator temp   | NTC thermistor                           | GPIO 32 (ADC1_CH4) |
| RPM              | Hall sensor pulse train                  | GPIO 27 (interrupt) |
| Wind speed       | Anemometer pulse train                   | GPIO 26 (interrupt) |
| Power output     | CT clamp + burden resistor               | GPIO 33 (ADC1_CH5) |

## 4. STM32

Reference firmware: [`stm32/aerovigil_stm32_main.c`](stm32/aerovigil_stm32_main.c).

STM32 boards usually reach the network through a coprocessor, so the reference
targets the common **ESP8266/ESP32 AT-bridge** pattern over UART:

- STM32 samples ADC channels (vibration, temperature, RPM, …),
- builds the same AeroVigil JSON frame,
- transmits it with `AT+CIPSTART` / `AT+CIPSEND` HTTP POST commands.

If your board has native networking (STM32H7 + LAN8742 Ethernet, NUCLEO with
an X-NUCLEO WiFi/WiFi shield, LwIP + MQTT), keep the JSON builder and swap the
transport; the AeroVigil side is unchanged.

## 5. MQTT alternative

Prefer broker-based ingestion? Run the Python gateway with the MQTT connector
and let MCUs publish small messages instead of HTTP:

```bash
python hardware_agent.py --connector mqtt --host broker.local \
    --gateway-id esp32-gw-01 --server http://localhost:8080
```

The MCU publishes to `aerovigil/<gateway-id>/<turbine-id>`:

```json
{"signal": "vibration_rms", "value": 2.51, "unit": "mm/s"}
```

`hardware_agent.py` normalizes, timestamps, buffers offline, and forwards to
`/api/hardware/stream` — useful where HTTP egress is blocked.

## 6. Production notes

- **TLS / reverse proxy.** The unified app is advisory-only and unauthenticated
  by design; never expose port 8080 directly to the internet. Put it behind
  nginx/Caddy/Tailscale Funnel with TLS and point `AEROVIGIL_SERVER` at the
  `https://` origin. The ESP32 sketch supports HTTPS — set the CA bundle or
  use `client.setInsecure()` for self-signed dev certs.
- **Provisioning.** One `gateway_id` per physical gateway, one `turbine_id`
  per asset. Twins are created automatically on first telemetry; no manual
  registration step is needed.
- **Duty cycle.** 10-minute SCADA cadence matches the platform's telemetry
  window defaults; 5–60 s is fine for demos. Batching reduces radio-on time.
- **Offline resilience.** Buffer on the device (the ESP32 sketch keeps the
  last batches in RAM; add SPIFFS/SD for power-loss durability) or use the
  MQTT path, where `hardware_agent.py` provides a SQLite store-and-forward
  buffer out of the box.
