# 🛡️ AeroVigil Long-Term Support (LTS) Policy & 3-Year Lifecycle (2026–2029)

> **Document Version**: 1.0.0  
> **Effective Date**: August 17, 2026  
> **End of Support (LTS Window)**: August 17, 2029  
> **Next Scheduled Major Platform Update**: August 2029 (v2.0.0 LTS)  
> **Status**: Active Enterprise LTS  

---

## 1. Executive Summary

Wind power infrastructure operates on multi-year capital expenditure cycles where stability, operational predictability, and backward compatibility are essential. Industrial SCADA systems, edge IoT gateways (ESP32/STM32), and fleet digital twins cannot tolerate frequent breaking schema or API revisions.

To serve utility-scale wind farms and OEM asset managers, **AeroVigil v1.0.0** is officially designated as a **3-Year Enterprise Long-Term Support (LTS)** release. 

Under this commitment:
1. **3-Year Stability Commitment**: Once deployed, AeroVigil v1.0.0 will remain stable, supported, and fully functional without requiring mandatory breaking upgrades for **3 full years (August 2026 to August 2029)**.
2. **Next Scheduled Major Update**: The next major platform upgrade (**v2.0.0 LTS**) is scheduled for release in **August 2029**, establishing a predictable 3-year enterprise upgrade cadence.
3. **Continuous Maintenance & Security**: Throughout the 2026–2029 window, AeroVigil AI provides non-breaking patch updates (`v1.0.x`), critical security fixes, sensor catalog enhancements, and performance optimizations.

---

## 2. Release Lifecycle & Support Timeline

```
2026-08 (v0.1.0) ───► 2026-08 (v1.0.0 LTS) ══════════════════════════════════════► 2029-08 (v2.0.0 LTS)
 Initial Release        Foundational Production LTS            3-Year LTS Window          Next Major Upgrade
  • PG-BNN core          • Whole-turbine diagnostics (12 subsys) • Zero breaking changes   • Next 3-yr cycle
  • MLOps pipeline       • Native desktop & mobile consoles      • Security patches        • 12-mo migration
  • Advisory safety      • Edge firmware & IoT connectivity      • Backward compatibility    window
```

### Support Schedule Matrix

| Version | Release Date | Support Tier | End of Active Support | End of Life (EOL) | Upgrade Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v0.1.0** | August 2026 | Initial / Community | August 2026 | August 2026 | Upgrade to v1.0.0 |
| **v1.0.0 LTS** | **August 17, 2026** | **Active Enterprise LTS** | **August 17, 2029** | **August 17, 2030** *(Extended)* | **v2.0.0 LTS (Aug 2029)** |
| **v1.0.x** | 2026 – 2029 | Maintenance & Security | August 17, 2029 | August 17, 2030 | Drop-in patch |
| **v2.0.0 LTS** | August 2029 | Next Planned LTS | August 2032 | August 2033 | Next 3-Year Cycle |

---

## 3. Strict Backward Compatibility Guarantees (2026–2029)

During the 3-year LTS lifecycle, AeroVigil guarantees strict backward compatibility across all integration boundaries:

### 3.1. REST & WebSocket API Boundary
- **Route Immobility**: All public routes under `/api`, `/advisory`, `/twin`, `/faults`, `/notifications`, `/jobs`, `/hardware`, and `/reports` will preserve their endpoints, query parameters, and response structures.
- **Contract Immutability**: Pydantic response models will never remove or rename existing fields. Any new analytical metrics will be purely additive and optional.
- **Advisory-Only Contract**: The advisory-only safety contract (`enforce_safety_contract`) remains strictly enforced across all responses.

### 3.2. SCADA Telemetry & Schema Stability
- The canonical 6-signal PG-BNN input schema (`wind_speed_ms`, `rotor_speed_rpm`, `generator_speed_rpm`, `power_kw`, `nacelle_temp_c`, `bearing_temp_c`) and extended 12-subsystem whole-turbine telemetry specifications remain completely stable.
- SCADA ingestion pipelines deployed in 2026 will continue to parse without modification through 2029.

### 3.3. Edge Hardware & Microcontroller Protocols
- ESP32 and STM32 firmware wire protocols (serial framing, JSON payloads, store-and-forward SQLite buffers) will remain wire-compatible with `/api/hardware/stream`.
- Field-deployed microcontroller units will not require firmware reflashing during the 3-year support term.

### 3.4. ONNX Models & Edge Runtimes
- Exported ONNX model graphs (`artifacts/pg_bnn.onnx`) and C++17 inference bindings (`cpp_inference/`) are pinned to stable opsets (opset 18+) ensuring multi-year binary compatibility with ONNX Runtime 1.20+.

### 3.5. Persistence & Digital Twin State
- SQLite storage schemas (`src/data/store.py`), digital twin event logs, and notification history maintain automatic forward-compatible migrations without manual DBA intervention.

---

## 4. Maintenance & Security Patch Policy

### 4.1. Patch Scope
Patches applied to the v1.0.x LTS branch are strictly constrained to:
- **Security Vulnerability Remediation**: Addressing CVEs in runtime dependencies or code logic.
- **Bug Fixes**: Correcting numerical drift, edge-case validation errors, or UI rendering glitches.
- **Performance & Stability**: Reducing memory footprints, optimizing inference latency, and tuning store-and-forward buffers.
- **OEM Sensor Specs Updates**: Non-breaking additions of new turbine models to `src/digital_twin/specs.py`.

### 4.2. Release Versioning Scheme
AeroVigil follows Semantic Versioning (SemVer 2.0.0):
$$\text{v}\langle\text{MAJOR}\rangle.\langle\text{MINOR}\rangle.\langle\text{PATCH}\rangle$$
- **MAJOR (3-Year Cadence)**: Incremented only every 3 years for architectural platform leaps (v1.0.0 $\to$ v2.0.0 in 2029).
- **MINOR (6–12 Month Cadence)**: Non-breaking feature additions (new charts, export formats, additional diagnostic modes).
- **PATCH (As-Needed Cadence)**: Drop-in bug fixes and security updates.

---

## 5. 3-Year Upgrade & Deprecation Roadmap (2029 Transition)

To ensure seamless operations when the 3-year LTS cycle concludes in **August 2029**:

1. **12-Month Advance Notice (August 2028)**: A comprehensive deprecation notice and preview documentation for v2.0.0 LTS will be published.
2. **6-Month Migration Sandbox (February 2029)**: Pre-release candidate (`v2.0.0-rc`) test environments and schema validation tools will be made available for staging fleets.
3. **1-Year Extended Overlap Support (August 2029 – August 2030)**: v1.0.0 will receive critical security patches for an additional 12 months after v2.0.0 LTS ships, giving operators an entire year to transition their fleets.
4. **Automated Migration CLI**: A dedicated upgrade command (`python main.py migrate-v2`) will automatically migrate configuration files, SQLite historical stores, and twin state registries.

---

## 6. Verification and Compliance

To verify your deployment's LTS compliance status at any time, query the application metadata:

```bash
# Via CLI
python main.py info

# Via REST API
curl -s http://localhost:8080/health | jq .
curl -s http://localhost:8080/api | jq .
```

*For enterprise support inquiries, pilot validation, or site-specific calibration, contact [rajaram@aerovigil.ai](mailto:rajaram@aerovigil.ai) or visit [https://aerovigil.abacusai.app](https://aerovigil.abacusai.app).*
