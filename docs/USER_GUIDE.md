# 🛠️ AeroVigil User Guide — hardware & software setup

This guide walks you through setting up AeroVigil for a real wind turbine or
a test rig: **which sensors to buy and where they go**, how to wire the data
into the AeroVigil server, how to install and run the software, and how to
turn on **email health reports and alerts**.

> ⚠️ **Safety first.** Work on turbines is performed by qualified personnel
> under LOTO (lock-out/tag-out) procedures. AeroVigil is **advisory only** —
> it never sends commands to the turbine. Everything below is monitoring
> hardware; nothing actuates the machine.

---

## 1. What you are building

```
                    FIELD (per turbine)                        OFFICE / CLOUD
 ┌──────────────────────────────────────────────────┐   ┌──────────────────────────────┐
 │  Sensors on the turbine                          │   │  AeroVigil server             │
 │  ┌─────────────┐  ┌──────────────┐  ┌─────────┐  │   │  ┌────────────────────────┐  │
 │  │ SCADA/PLC   │  │ CMS          │  │ Fire &  │  │   │  │ unified app :8080       │  │
 │  │ (temps, RPM,│  │ (vibration,  │  │ smoke   │  │   │  │  • fault detection      │  │
 │  │ pressures,  │  │ oil sensors) │  │ sensors │  │   │  │  • digital twin + RUL   │  │
 │  │ pitch/yaw)  │  └──────┬───────┘  └────┬────┘  │   │  │  • fleet dashboard      │  │
 │  └──────┬──────┘         │               │       │   │  └──────────┬─────────────┘  │
 │         └────────────────┴───────────────┴───────┼───▶  POST /api/hardware/stream  │
 │                    Edge node                     │   │            │                │
 │           ESP32 / STM32 / Raspberry Pi           │   │            ▼                │
 │           (edge/ firmware, WiFi/LTE/Modbus)      │   │   Email alerts & reports    │
 └──────────────────────────────────────────────────┘   │   ops@… (CRITICAL/HIGH)    │
                                                        │   maintenance@… (digests)  │
                                                        └──────────────────────────────┘
```

The five canonical SCADA signals are the minimum (the model was trained on
them); every extra sensor unlocks more fault types in the catalog.

---

## 2. Hardware — sensor bill of materials

The full sensor catalog (with placement, technology, output and the faults
each sensor feeds) is in `src/faults/sensors.py` and via
`python main.py faults --sensors` or `GET /api/faults/sensors`.

### Minimum viable set (drivetrain monitoring)

| # | Sensor | Placement | Faults it reveals |
| --- | --- | --- | --- |
| 1 | Vibration sensor (accelerometer) | Main bearing + gearbox HSS | HS-01, GB-09, GB-10, GB-11, BR-03, GN-07 |
| 2 | Bearing/oil temperature (RTD) | Gearbox oil sump + main bearing | GB-01, GB-15, HS-01 |
| 3 | HSS speed encoder | Gearbox output shaft | BR-02 |
| 4 | Online viscometer | Gearbox oil line | GB-02, GB-03, GB-13 |
| 5 | Water-in-oil sensor | Gearbox oil line | GB-04 |
| 6 | Oil level sensor | Gearbox sump | GB-06 |
| 7 | Oil pressure transmitter | Gearbox supply | GB-12 |
| 8 | Filter ΔP switch | Filter housing | GB-07 |
| 9 | Generator winding RTD | Stator slots | GN-01, GN-03 |
| 10 | Smoke + heat detector | Nacelle | NS-06, BR-05, GB-15, EL-06 |

### Recommended full set (add per subsystem)

| Subsystem | Recommended sensors |
| --- | --- |
| Rotor & blades | 1P vibration (VB-01), blade strain gauge (VB-05), acoustic emission (VB-08), ice detector (WS-04), lightning counter (FS-06), blade fire alarm (FS-03) |
| Pitch | Pitch encoders ×3 (PS-01), pitch motor current (PS-02), hydraulic pressure (PS-06), accumulator/feather-time (PS-07) |
| Hub & main shaft | Main bearing RTD (TH-01), grease debris sensor (OC-08), axial displacement probe (RM-03), torque sensor (RM-04) |
| Gearbox | Oil sensors (OC-01…OC-09), gearbox vibration (VB-02), bearing envelope (VB-03), oil smoke detector (FS-01) |
| HSS & brake | Brake wear (PS-05), brake disc temp (TH-10), brake fire alarm (FS-03), HSS speed (RM-01) |
| Generator | Winding RTD (TH-03), bearing TC (TH-04), slip-ring temp (TH-11), partial discharge (EL-03), insulation monitor (EL-05), MCSA (EL-06) |
| Yaw | Yaw encoder (PS-03), yaw drive current (PS-04), yaw brake pressure (PS-06) |
| Tower & foundation | Tower accelerometer (VB-04), tilt sensor (VB-07), foundation bolt audits (DA-04) |
| Nacelle & sensors | Anemometer ×2 (WS-01), vane ×2 (WS-03), nacelle T/RH (TH-06), smoke/heat/flame (FS-01/02/03), suppression status (FS-05) |
| Cooling & hydraulics | Coolant temp (TH-09), coolant flow/pressure (TH-13), hydraulic pressure (PS-06), hydraulic oil condition (PS-08) |
| Electrical & power | Converter IGBT temp (TH-08), transformer temp (TH-07), power meter (EL-01), THD analyzer (EL-02), PD sensor (EL-03), DC-link monitor (EL-04), cabinet fire alarm (FS-03) |
| SCADA & comms | Controller PLC (DA-01), RTU/gateway (DA-02), NTP time sync (DA-05) |

### Edge hardware options (this repo ships firmware)

| Device | Firmware | Use case |
| --- | --- | --- |
| ESP32 | `edge/esp32/aerovigil_telemetry.ino` | Low-cost WiFi sensor node (vibration, temperature, oil) |
| STM32 | `edge/stm32/aerovigil_stm32_main.c` | Industrial node with analog 4–20 mA inputs |
| Raspberry Pi | any Python | Protocol gateway (Modbus/OPC-UA/MQTT) or full edge AI |
| Server | any x86/ARM VM | Unified app (FastAPI) + email alerts; see Docker/k8s below |

---

## 3. Software — setup step by step

### 3.1 Prerequisites

- Python **3.9–3.12** (tested on 3.11), `git`, ~4 GB disk
- Optional: Docker 24+ (container deployment), PlatformIO (ESP32 flashing),
  an SMTP account (Gmail, Outlook, or your company relay) for email alerts

### 3.2 Install the software

```bash
git clone https://github.com/rajaram-2005/wind-turbine-pg-bnn.git
cd wind-turbine-pg-bnn

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[api,dev,demo]"     # core + API + tests + demo UI
```

### 3.3 Configure

Edit `configs/default.yaml` (or set env vars — env wins):

```bash
# ── Email alerts & health reports ──────────────────────────────────
export AV_NOTIFY_MODE=smtp            # auto | smtp | eml | off
export AV_SMTP_HOST=smtp.gmail.com
export AV_SMTP_PORT=587
export AV_SMTP_USER=aerovigil@yourfarm.com
export AV_SMTP_PASSWORD='app-password'
export AV_SMTP_FROM=aerovigil@yourfarm.com
export AV_ALERT_RECIPIENTS='ops@yourfarm.com, oncall@yourfarm.com'
export AV_REPORT_RECIPIENTS='maintenance@yourfarm.com'

# ── Optional: trained model / serving ──────────────────────────────
# export AV_MODEL_PATH=artifacts/pg_bnn.pt
```

Secrets never go in the repository — they live in the environment (or your
deployment's secret store; see `k8s/secret.yaml`).

### 3.4 Train the PG-BNN (optional — a demo checkpoint ships with the repo)

```bash
python main.py train --config configs/default.yaml --epochs 100
python main.py evaluate --checkpoint artifacts/pg_bnn.pt
```

### 3.5 Run the server

```bash
uvicorn src.unified_app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080  → console, /api/docs → API reference
```

Or with Docker:

```bash
docker compose up --build        # unified app + optional Redis broker
# or Kubernetes: kubectl apply -k k8s/
```

### 3.6 Connect field data (pick one)

**A. Edge node → HTTP** (easiest; ESP32/STM32/RPi):

```bash
# simulate a turbine sending live readings every 10 s:
python edge/simulate_device.py --url http://<server>:8080 --turbine WTG-001
```

Real nodes POST the same wire format (see `edge/README.md`):

```json
POST /api/hardware/stream
{
  "gateway_id": "esp32-gw-01",
  "readings": [{
    "turbine_id": "WTG-001",
    "signal": "vibration_rms",
    "value": 2.51,
    "unit": "mm/s",
    "quality": "good",
    "timestamp": "2026-08-17T05:00:00.000Z"
  }]
}
```

Signal names fold onto the canonical channels automatically
(`vibration_rms` → `vibration_mms`, `gearbox_temp` → `temperature_c`, ...).
See the alias table in `src/api/gateway_routes.py`.

**B. Industrial protocols** — the gateway routes support MQTT, Modbus/TCP
and OPC-UA (endpoints and register maps documented in the README's
hardware-gateway section, `config.json`).

**C. Bulk CSV** (lab/retrofit):

```bash
python main.py faults --snapshot examples/fault_payload.json
python main.py notify --fleet examples/fleet.csv --recipient ops@… --report
```

### 3.7 Flash the ESP32 node (optional)

```bash
cd edge/esp32
pip install platformio
pio run -t upload                 # sets WiFi SSID/password + server URL in the .ino
```

---

## 4. Email health reports & alerts

| Trigger | Email | When |
| --- | --- | --- |
| **CRITICAL** fault (fire, overspeed, tooth breakage, …) | Immediate alert `[AeroVigil CRITICAL]` | instantly, per (asset, fault), re-alert after 6 h |
| **HIGH** fault (viscosity too low, water contamination, …) | Immediate alert `[AeroVigil HIGH]` | instantly, re-alert after 24 h |
| **MEDIUM / LOW** faults | Rolled into the periodic health report | with the digest |
| Fleet digest | `[AeroVigil Health Report]` | on demand / cron |

Severity **escalation** (e.g. MEDIUM → CRITICAL) always alerts immediately,
even inside a cooldown. Without SMTP configured, emails are written as
standard `.eml` files under `artifacts/notifications/` so you can preview and
test delivery offline.

```bash
# CLI
python main.py notify --snapshot examples/fault_payload.json --recipient ops@… --model NREL-5MW
python main.py notify --fleet examples/fleet.csv --recipient maintenance@… --report

# API
curl -X POST http://localhost:8080/api/notifications/send -H 'Content-Type: application/json' \
  -d @examples/fault_payload.json
curl http://localhost:8080/api/notifications/status

# Cron digest (Linux): every day 06:00
0 6 * * * cd /opt/aerovigil && .venv/bin/python main.py notify \
  --fleet examples/fleet.csv --report --subject "Daily fleet health" >> /var/log/aerovigil.log 2>&1
```

See `docs/NOTIFICATIONS.md` for the full alert-policy reference.

---

## 5. Verify the installation

```bash
# 1. Health check
curl http://localhost:8080/health

# 2. Fault catalog + sensors available
python main.py faults --list | python -c "import json,sys; print(json.load(sys.stdin)['summary']['total_fault_types'], 'fault types')"
python main.py faults --sensors --subsystem gearbox | python -c "import json,sys; d=json.load(sys.stdin); print(d['summary']['n_sensors_filtered'], 'gearbox sensors')"

# 3. End-to-end: detect + email
AV_ALERT_RECIPIENTS=ops@example.com python main.py notify \
  --snapshot examples/fault_payload.json --model NREL-5MW
ls artifacts/notifications/*.eml        # or check the recipient's inbox

# 4. Live stream into the twin (fires detection + alerts automatically)
python edge/simulate_device.py --url http://localhost:8080 --turbine WTG-001
curl "http://localhost:8080/api/twin/status?asset_id=WTG-001&model=NREL-5MW" | \
  python -c "import json,sys; print(json.load(sys.stdin)['last_state']['fault_report']['overall_status'])"
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No emails, no `.eml` files | `AV_NOTIFY_MODE=off` or no recipients | Check `GET /api/notifications/status`; set `AV_ALERT_RECIPIENTS` |
| SMTP login fails | App password required (Gmail) | Use an app password / allow relay from server IP |
| `delivered: false` with SMTP error | Firewall port 587 blocked | Open outbound 587 (or use 465 with `AV_SMTP_TLS=0`) |
| Telemetry rejected | Unknown signal name | Use the alias table in `src/api/gateway_routes.py` |
| No faults detected on real data | Thresholds are defaults | Tune per-OEM limits in the detector or `TurbineSpec` |
| Alerts too chatty / too quiet | Cooldown settings | Adjust `cooldown_hours` in `configs/default.yaml` |
| ESP32 won't connect | Wrong server URL/port | Node must reach `http://<server>:8080/api/hardware/stream` |

---

## 7. Fleet-scale deployment

- **One server, many turbines** — the twin registry holds assets LRU-bounded;
  SQLite persists state (`src/data/store.py`).
- **Docker**: `docker compose up --build` (see `docker-compose.yml`).
- **Kubernetes**: `kubectl apply -k k8s/` (deployment, PVC, secrets, ingress).
- **Edge AI**: export the model to ONNX (`python main.py export`) and run
  `src/deployment/cpp_inference` on the edge node for offline RUL scoring.
- **Mobile**: the Flutter console (`apps/aerovigilai_flutter`) talks to the
  same `/api` surface.
