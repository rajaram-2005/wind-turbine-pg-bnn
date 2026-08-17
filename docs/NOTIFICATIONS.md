# 📧 Email Health Reports & Alerts

AeroVigil emails the **health report** for every asset on request, and it
**alerts the receiver immediately** when a fault is important enough
(CRITICAL / HIGH). This document is the full reference for the notification
system (`src/notifications/emailer.py`).

## Alert policy

| Severity | Behaviour | Re-alert cooldown |
| --- | --- | --- |
| CRITICAL | Instant email `[AeroVigil CRITICAL] …` | 6 hours |
| HIGH | Instant email `[AeroVigil HIGH] …` | 24 hours |
| MEDIUM | Digest only (health report) | — |
| LOW | Digest only (health report) | — |

Rules:

1. **First sighting** of a CRITICAL/HIGH fault → email immediately.
2. **Dedupe** — the same (asset, fault) is not re-emailed until its cooldown
   elapses (state persists in `artifacts/notifications/alert_state.json`).
3. **Escalation** — if a known fault reappears at a *higher* severity than
   last sent, a new alert fires immediately, ignoring the cooldown.
4. **Suppression check** — suppression released with no fire evidence is
   reported as a fire-suppression system fault (NS-07), not a fire.

## What each email contains

* **Alert email** — asset id, timestamp, health score, oil score, overall
  status, and a table of every detected fault: severity badge, fault id,
  name, subsystem, evidence, confidence, recommended actions. Plain-text
  alternative included. Every email carries the advisory-only banner.
* **Health report email** — one row per asset: status, health score, oil
  score, fault count, top findings.

## Configuration

Secrets go in the environment (`AV_*`); non-secrets can live in
`configs/default.yaml` under `notifications:`.

| Env var | Default | Purpose |
| --- | --- | --- |
| `AV_NOTIFY_MODE` | `auto` | `smtp` \| `eml` \| `off`; `auto` = SMTP if host set, else `.eml` files |
| `AV_SMTP_HOST` | — | SMTP server (e.g. `smtp.gmail.com`) |
| `AV_SMTP_PORT` | `587` | SMTP port |
| `AV_SMTP_TLS` | `1` | STARTTLS on/off |
| `AV_SMTP_USER` | — | Login user |
| `AV_SMTP_PASSWORD` | — | Login password / app password |
| `AV_SMTP_FROM` | `aerovigil@localhost` | From address |
| `AV_ALERT_RECIPIENTS` | — | Comma/`;` separated alert recipients |
| `AV_REPORT_RECIPIENTS` | — | Comma/`;` separated digest recipients |
| `AV_NOTIFY_DIR` | `artifacts/notifications` | `.eml` fallback + alert state location |

YAML equivalent (`configs/default.yaml`):

```yaml
notifications:
  mode: auto
  smtp_host: ""
  smtp_port: 587
  smtp_from: aerovigil@localhost
  smtp_tls: true
  alert_recipients: []
  report_recipients: []
  artifact_dir: artifacts/notifications
  cooldown_hours: {CRITICAL: 6.0, HIGH: 24.0, MEDIUM: 168.0}
  alert_severities: [CRITICAL, HIGH]
```

## Using it

### CLI

```bash
# Immediate alert for any CRITICAL/HIGH fault in the snapshot
python main.py notify --snapshot examples/fault_payload.json --model NREL-5MW \
  --recipient ops@windfarm.com

# Fleet health digest (one email, all assets)
python main.py notify --fleet examples/fleet.csv --recipient maintenance@windfarm.com \
  --report --subject "Daily fleet health"

# Force a report even when no severe fault exists
python main.py notify --snapshot healthy.json --recipient ops@windfarm.com --report
```

### API

```bash
# Send (alert or report) for a snapshot
curl -X POST http://localhost:8080/api/notifications/send \
  -H 'Content-Type: application/json' -d @examples/fault_payload.json

# Notifier configuration (no secrets)
curl http://localhost:8080/api/notifications/status
```

### Digital twin

Attach a notifier to a twin and every `update_state` fires detection **and**
alerts automatically:

```python
from src.digital_twin.specs import get_spec
from src.digital_twin.twin import WindTurbineDigitalTwin
from src.notifications import EmailNotifier

twin = WindTurbineDigitalTwin("WTG-001", get_spec("NREL-5MW"), notifier=EmailNotifier())
# every twin.update_state(...) now emails new CRITICAL/HIGH findings
```

### Cron digest (Linux)

```cron
0 6 * * * cd /opt/aerovigil && .venv/bin/python main.py notify \
  --fleet examples/fleet.csv --report --subject "Daily fleet health" \
  >> /var/log/aerovigil.log 2>&1
```

## Offline / preview mode

With no SMTP host configured, mode resolves to `eml`: every notification is
written as a standard `.eml` file under `artifacts/notifications/`. Open it
in any mail client (or `cat` it) to preview exactly what recipients receive,
and use it in air-gapped deployments.

```bash
AV_ALERT_RECIPIENTS=ops@example.com python main.py notify \
  --snapshot examples/fault_payload.json
ls artifacts/notifications/*.eml
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `channel: skipped` | No recipients configured — set `AV_ALERT_RECIPIENTS` / `AV_REPORT_RECIPIENTS` |
| SMTP auth failure | Use an app password (Gmail/Outlook) or whitelist the server IP |
| Emails landing in spam | Set `AV_SMTP_FROM` to a real address, add SPF/DKIM records |
| Too many alerts | Raise `cooldown_hours` for the noisy severity |
| Alerts missing | Confirm the fault severity is in `alert_severities` and the tracker state file wasn't manually reset |
