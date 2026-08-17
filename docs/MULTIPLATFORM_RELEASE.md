# 🌐 AeroVigil Multiplatform Build & Release Guide

AeroVigil AI is built to operate across diverse environments: utility control rooms (Windows/macOS desktops), field technician mobile tablets (Android/iOS), on-premise Linux gateways, and web browsers.

This guide details how to build, package, and release AeroVigil across all supported platforms.

---

## 📱 Platform Target Matrix

| Platform | Target Artifact | Minimum OS / Requirements | CI Runner |
| :--- | :--- | :--- | :--- |
| **Windows** | `aerovigil-windows-x64.zip` | Windows 10 (1909+) x64 · 4 GB RAM | `windows-latest` |
| **macOS** | `aerovigil-macos-universal.zip` | macOS 12 Monterey+ · Apple Silicon & Intel | `macos-latest` |
| **Linux** | `aerovigil-linux-x64.tar.gz` | GTK 3.22+, Ubuntu 20.04+, Debian 11+ | `ubuntu-latest` |
| **Android** | `aerovigil-android.apk` | Android 8.0 (API 26)+ · arm64-v8a / armeabi-v7a | `ubuntu-latest` (Java 17) |
| **iOS** | `aerovigil-ios.ipa` | iOS 15+ (requires macOS + Apple Developer signing) | `macos-latest` |
| **Web Console** | Static Bundle (`web_console/dist`) | Any modern browser (Chrome, Edge, Safari, Firefox) | Any host |
| **Edge C++** | `aerovigil_cpp_engine` | C++17, CMake 3.16+, ONNX Runtime 1.20+ | Ubuntu / Windows / macOS |

---

## ⚡ Automated CI/CD Release (GitHub Actions)

The repository includes a GitHub Actions release workflow configured in [`.github/workflows/release-apps.yml`](../.github/workflows/release-apps.yml).

### Method 1: Release Trigger via Git Tag
Pushing any version tag matching `v*` (such as `v1.0.0`) automatically triggers build jobs for all platforms in parallel:

```bash
git tag v1.0.0
git push origin v1.0.0
```

### Method 2: Manual Dispatch via GitHub Actions UI
To build a single platform or publish to an existing release:
1. Navigate to **GitHub Actions** → **release-apps**.
2. Click **Run workflow**.
3. Choose the platform: `all`, `windows`, `macos`, `linux`, or `android`.
4. Optionally specify the release tag (e.g. `v1.0.0`).

---

## 🛠️ Local Build & Packaging

### Prerequisites
- **Flutter SDK** (≥ 3.10.0): [flutter.dev/docs/get-started/install](https://flutter.dev/docs/get-started/install)
- **Host Compilers**:
  - Windows: Visual Studio 2022 with "Desktop development with C++"
  - macOS: Xcode 14+
  - Linux: `sudo apt-get install clang cmake ninja-build pkg-config libgtk-3-dev`
  - Android: Android SDK Command-line Tools + OpenJDK 17

### One-Command Multiplatform Tool
Use the bundled Python build script to scaffold, compile, and package:

```bash
# Build all platforms supported by current host
python scripts/build_apps.py --platform all

# Build individual target
python scripts/build_apps.py --platform windows
python scripts/build_apps.py --platform macos
python scripts/build_apps.py --platform linux
python scripts/build_apps.py --platform android

# Dry-run inspection
python scripts/build_apps.py --dry-run
```

---

## 🖥️ Manual Platform Build Instructions

### 1. Windows (x64)
```bash
cd apps/aerovigilai_flutter
flutter create --platforms=windows --org ai.aerovigil --project-name aerovigilai_flutter .
flutter pub get
flutter build windows --release

# Package into zip
powershell Compress-Archive -Path build/windows/x64/runner/Release/* -DestinationPath ../../dist/apps/aerovigil-windows-x64.zip
```

### 2. macOS (Universal: Apple Silicon + Intel)
```bash
cd apps/aerovigilai_flutter
flutter create --platforms=macos --org ai.aerovigil --project-name aerovigilai_flutter .
flutter pub get
flutter build macos --release

# Package into zip preserving resource forks
ditto -c -k --sequesterRsrc --keepParent build/macos/Build/Products/Release/aerovigilai_flutter.app ../../dist/apps/aerovigil-macos-universal.zip
```

### 3. Linux (GTK 3.22+)
```bash
cd apps/aerovigilai_flutter
flutter create --platforms=linux --org ai.aerovigil --project-name aerovigilai_flutter .
flutter pub get
flutter build linux --release

# Package into tarball
tar -czf ../../dist/apps/aerovigil-linux-x64.tar.gz -C build/linux/x64/release bundle
```

### 4. Android (APK)
```bash
cd apps/aerovigilai_flutter
flutter create --platforms=android --org ai.aerovigil --project-name aerovigilai_flutter .
flutter pub get
flutter build apk --release

# Output artifact
cp build/app/outputs/flutter-apk/app-release.apk ../../dist/apps/aerovigil-android.apk
```

---

## 🔌 Connecting Consoles to AeroVigil Server

By default, all native applications connect to `http://localhost:8080`. 

To configure custom endpoints:
- **Environment Override at Build**: Pass `--dart-define=API_BASE=http://<host>:8080` during `flutter run` or `flutter build`.
- **Android Emulator**: `http://10.0.2.2:8080` routes to the host machine.
- **Remote / Offshore Wind Farm**: Pass the secure domain, e.g. `--dart-define=API_BASE=https://aerovigil.abacusai.app`.

---

## 🔐 Checksum Verification

All release builds generate a `SHA256SUMS.txt` file containing cryptographic checksums for verification:

```bash
# Verify integrity on Linux/macOS
sha256sum -c SHA256SUMS.txt

# Verify on Windows PowerShell
Get-FileHash -Algorithm SHA256 aerovigil-windows-x64.zip
```
