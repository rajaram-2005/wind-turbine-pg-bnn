# AeroVigilAI Native Console (Flutter)

Cross-platform operator console for the AeroVigilAI predictive-maintenance
framework. Runs natively on **Windows, macOS, Android, and iOS** from a single
codebase and talks to the canonical AeroVigil server on `http://localhost:8080`.

## Screens

| Screen | Purpose |
|--------|---------|
| **Digital Twin** | Live gauges (generator RPM, gearbox temperature) + synchronized time-series charts of live state & simulation metrics. |
| **Data Ingestion** | Offline USB/CSV file selection via `file_picker`, uploaded over HTTPS. |
| **Fleet Reports** | Paginated data table of aggregate fleet health. |
| **AeroZip Telemetry** | Compression ratio, bandwidth-reduction and restoration status. |
| **Low-Level Inference** | Submit manual JSON payloads directly to `/api/model`. |

All network traffic is routed through the centralized `ApiService`
(`lib/services/api_service.dart`), whose base URL defaults to the canonical
`http://localhost:8080`.

## Getting started

```bash
cd apps/aerovigilai_flutter
flutter pub get
```

### Run per platform

```bash
flutter run -d windows      # Windows desktop
flutter run -d macos        # macOS desktop
flutter run -d android      # Android device / emulator
flutter run -d ios          # iOS simulator / device
```

> **Android emulator note:** `localhost` on the host is reachable at
> `http://10.0.2.2:8080` from the emulator. Override the base URL:
>
> ```dart
> final api = ApiService(baseUrl: 'http://10.0.2.2:8080');
> ```

### Platform scaffolding

This directory ships the Dart source (`lib/`) and `pubspec.yaml`. Generate the
native platform folders once with:

```bash
flutter create --platforms=windows,macos,android,ios .
```

This leaves the existing `lib/` untouched and adds the `windows/`, `macos/`,
`android/`, and `ios/` runners.

## Responsiveness

The shell uses a `NavigationRail` on wide screens (≥ 720 px, desktops/tablets)
and a `NavigationBar` on narrow screens (phones), so the same screens adapt
across all four target platforms.
