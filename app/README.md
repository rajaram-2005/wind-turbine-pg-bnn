# AeroVigil Web App

A separately deployable browser application for the AeroVigil advisory engine.
It serves the UI and the operations API from one origin, avoiding browser-side
`localhost` calls and CORS configuration.

## Run

From the repository root:

```bash
pip install -e ".[api]"
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`. The API is available at `/api/docs`.

To attach a physics-guided framework checkpoint:

```bash
export AV_PHYSICS_GUIDED_MODEL_PATH=artifacts/pg_bnn.pt
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

All outputs are decision support only; this application issues no turbine
commands.
