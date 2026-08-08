# AeroVigilAI native apps

This Flutter client is intentionally separate from the browser app. It has
native, responsive UI/UX for **Windows**, **macOS**, **iOS**, and **Android**,
while calling the same advisory-only backend.

## 1. Start the local API

From the repository root:

```bash
pip install -e ".[api]"
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

## 2. Select the right local URL

| Client | Default API URL | Notes |
|---|---|---|
| Windows / macOS desktop | `http://127.0.0.1:8080` | Backend runs on the same computer. |
| iOS Simulator | `http://127.0.0.1:8080` | Backend runs on the Mac. |
| Android Emulator | `http://10.0.2.2:8080` | Android’s alias for the host computer. |
| Physical iOS / Android device | Your computer’s LAN IP | Phone and computer must share a network. |

Override the URL for a LAN device or deployed backend:

```bash
flutter run --dart-define=AEROVIGILAI_API_URL=http://192.168.1.50:8080
```

Do not use `localhost` on a physical phone: it refers to the phone, not the
computer running the API.

## Generate platform runners once

The shared Dart source is versioned here without generated Xcode, Gradle, and
Visual Studio runner files. On a development machine with Flutter installed,
generate them once from this folder:

```bash
flutter create --platforms=windows,macos,ios,android .
```

This preserves `lib/main.dart` and creates the platform-native project files.
For Android HTTP development, use the emulator URL shown above or configure
Android network security for your development LAN endpoint.

## Run

```bash
flutter pub get
flutter run -d windows   # Windows
flutter run -d macos     # macOS
flutter run -d ios       # iOS simulator/device
flutter run -d android   # Android emulator/device
```

The app has separate desktop and mobile layouts. Both use only relative
advisory data from the backend and never issue turbine commands.
