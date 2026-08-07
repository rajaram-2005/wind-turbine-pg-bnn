# AeroVigil — Datasets & Benchmarks

> Public datasets and benchmarks relevant to wind turbine prognostics,
> bearing RUL prediction, and SCADA-based condition monitoring.

---

## 1. Bearing Datasets (Run-to-Failure)

### 1.1 IEEE PHM 2012 / FEMTO-ST (PRONOSTIA)

| Property | Detail |
|----------|--------|
| **Source** | FEMTO-ST Institute, Besançon, France |
| **Content** | 6 bearings run to failure under varying loads/speeds |
| **Signals** | Vibration (horizontal + vertical), temperature |
| **Sampling** | 4 kHz vibration, 10 s intervals |
| **Bearing type** | Ball bearing (6203-2RS) |
| **Conditions** | 3 operating conditions (load × speed) |
| **Link** | [FEMTO-ST Data](https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-challenge) |

**Use in AeroVigil:** The teaching rule was validated against degradation patterns observed in this dataset. Used for benchmarking RUL estimation accuracy.

### 1.2 IMS (University of Cincinnati / NREL)

| Property | Detail |
|----------|--------|
| **Source** | University of Cincinnati, NSF/IUCRC |
| **Content** | 4 bearings run to failure on a test rig |
| **Signals** | Vibration (4 accelerometers), temperature, RPM |
| **Sampling** | 20 kHz |
| **Bearing type** | Tapered roller bearing |
| **Failure modes** | Inner race, outer race, rolling element |
| **Link** | [IMS Data](https://www.nrel.gov/wind/test-facilities.html) |

---

## 2. Wind Turbine SCADA Data

### 2.1 NREL 5MW Reference Turbine Data

| Property | Detail |
|----------|--------|
| **Source** | National Renewable Energy Laboratory (NREL) |
| **Content** | Simulation data for 5MW offshore reference turbine |
| **Signals** | 300+ channels including all SCADA-relevant signals |
| **Duration** | Multiple 10-minute simulations across wind regimes |
| **Link** | [NREL Data Archive](https://www.nrel.gov/wind/nrel-reference-turbines.html) |

### 2.2 DTU 10MW Reference Wind Turbine

| Property | Detail |
|----------|--------|
| **Source** | DTU Wind Energy, Denmark |
| **Content** | Full aeroelastic simulation data for 10MW offshore turbine |
| **Signals** | Structural loads, SCADA channels, environmental conditions |
| **Link** | [DTU 10MW](https://dtuwinder.gitlab.io/references/10mw/) |

### 2.3 PHM Society Data Challenge (2022)

| Property | Detail |
|----------|--------|
| **Source** | PHM Society |
| **Content** | Real wind farm SCADA data with annotated fault events |
| **Signals** | Temperature, power, wind speed, RPM, status codes |
| **Link** | [PHM Challenge](https://www.phmsociety.org/phm-data-challenge/) |

### 2.4 La Haute-Borne (Engie)

| Property | Detail |
|----------|--------|
| **Source** | Engie Green (now part of ENGIE) |
| **Content** | 2+ years of 10-minute SCADA data from 4 turbines |
| **Turbines** | 4× 2MW Vestas V110 |
| **Signals** | 40+ channels per turbine |
| **Link** | [Data Portal](https://opendata-rengines.green/) |

---

## 3. Turbofan / Rotating Machinery (Transfer Learning)

### 3.1 NASA C-MAPSS

| Property | Detail |
|----------|--------|
| **Source** | NASA Ames Prognostics Center |
| **Content** | 218 turbofan engine degradation trajectories |
| **Signals** | 21 sensor channels, 3 operational settings |
| **Datasets** | FD001–FD004 (varying fault modes & operating conditions) |
| **Link** | [NASA Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |

**Relevance:** While not wind turbine data, C-MAPSS is the canonical benchmark for RUL prediction algorithms. Methods validated here can be adapted for wind turbine drivetrain prognostics via transfer learning.

### 3.2 PRONOSTIA / FEMTO-ST (see 1.1)

### 3.3 CWRU Bearing Data Center

| Property | Detail |
|----------|--------|
| **Source** | Case Western Reserve University |
| **Content** | Vibration data for bearings with various fault types |
| **Fault types** | Inner race, outer race, ball defects at various severities |
| **Link** | [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter) |

---

## 4. Environmental / Wind Resource Data

| Dataset | Description | Link |
|---------|-------------|------|
| **Global Wind Atlas** | High-resolution wind resource maps worldwide | [globalwindatlas.info](https://globalwindatlas.info/) |
| **ERA5 Reanalysis** | Hourly weather data (wind, temperature) at 31 km grid | [ECMWF](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) |
| **NREL WIND Toolkit** | US wind resource data at 2 km resolution | [NREL](https://www.nrel.gov/grid/wind-toolkit.html) |
| **EM-DAT** | International disaster database (includes wind farm incidents) | [EM-DAT](https://www.emdat.be/) |

---

## 5. How AeroVigil Uses These Resources

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  PHM 2012 + IMS bearings ──→ Teaching rule validation       │
│       │                                                      │
│       ▼                                                      │
│  NREL 5MW + DTU 10MW ────→ Turbine spec library             │
│       │                     (8 OEM profiles)                 │
│       ▼                                                      │
│  ERA5 + Wind Atlas ────────→ Regional climate modifiers      │
│       │                     (6 climate zones)                │
│       ▼                                                      │
│  C-MAPSS ─────────────────→ Transfer learning benchmark      │
│       │                                                      │
│       ▼                                                      │
│  Synthetic fleet (28,400 ──→ EPIC model training             │
│  samples, 8 OEMs,                                           │
│  6 regions, 4 fault modes)                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Data Requirements for Production Deployment

For site-specific calibration (beyond the demo), operators should provide:

| Data | Minimum | Preferred | Source |
|------|---------|-----------|--------|
| SCADA data (10-min avg) | 12 months | 3+ years | Turbine SCADA system |
| Failure logs | All drivetrain events | With exact failure dates | CMMS / maintenance records |
| Bearing specs | Model + manufacturer | Full ISO 281 parameters | OEM documentation |
| Ambient conditions | Wind speed + temperature | Pressure, humidity, icing | Met mast or NWP data |
| Maintenance history | Major interventions | Full work order history | CMMS system |

---

*For questions about data formats or integration, see the [API documentation](../README.md#rest-api).*

*Last updated: August 2026*
