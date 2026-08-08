# AeroVigilAI Web App

A separately deployable browser application for the AeroVigilAI advisory engine.
It serves the UI and the operations API from one origin, avoiding browser-side
`localhost` calls and CORS configuration.

## Run

From the repository root:

```bash
pip install -e ".[api]"
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`. The API is available at `/api/docs`.

## Hardware telemetry

The **Import hardware telemetry** panel supports a USB-mounted/local CSV chosen
in the browser and a signed HTTPS cloud CSV URL. Both require the canonical
five SCADA columns: `vibration_mms`, `temperature_c`, `rpm`,
`oil_viscosity_cst`, and `load_pct`. Import validates the data and applies the
latest row to the advisory form; raw telemetry is not persisted by default.

To attach a physics-guided framework checkpoint:

```bash
export AV_PHYSICS_GUIDED_MODEL_PATH=artifacts/pg_bnn.pt
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

All outputs are decision support only; this application issues no turbine
commands.
