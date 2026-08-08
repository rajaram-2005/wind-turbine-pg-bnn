# AeroVigilAI client applications

| Folder | Client | Platforms | UI/UX |
|---|---|---|---|
| [`aerovigilai_flutter/`](aerovigilai_flutter/) | AeroVigilAI native operator app | Windows, macOS, iOS, Android | Responsive desktop workspace and mobile field-assessment flow |

The browser application remains in [`../app/`](../app/). It is intentionally
separate from the native app and is named **AeroVigilAI** in its user-facing UI.

All clients connect to the same advisory-only service. Start it with:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

See the native app README for platform-specific localhost and LAN addresses.
