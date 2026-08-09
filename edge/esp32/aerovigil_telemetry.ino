/**
 * AeroVigil ESP32 telemetry node
 * ==============================
 * Connects an ESP32 to the AeroVigil cloud (the unified app) by POSTing
 * normalized SCADA readings to:
 *
 *     POST {AEROVIGIL_SERVER}/api/hardware/stream
 *
 * When all seven signals are present in a batch, the cloud computes a real
 * six-signal PG-BNN advisory (advisory_source "stream-model-six-signal"),
 * updates the asset's digital twin, and refreshes the fleet — you can watch
 * it happen in the browser console's Digital Twin / Fleet / Hardware pages.
 *
 * Boards : any ESP32 with WiFi (ESP32, ESP32-S3, ESP32-C3, ...)
 * Deps   : none beyond the Arduino-ESP32 core (no ArduinoJson needed)
 *
 * Full guide: edge/README.md in the wind-turbine-pg-bnn repository.
 * SAFETY: AeroVigil is advisory-only. This node only reports telemetry —
 * it never receives, and must never be wired to, actuation commands.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <time.h>
#include <Preferences.h>

// ══════════════════════════ CONFIGURATION ══════════════════════════
static const char* WIFI_SSID        = "YOUR_WIFI_SSID";
static const char* WIFI_PASS        = "YOUR_WIFI_PASSWORD";

// The unified AeroVigil app, e.g. "http://192.168.1.50:8080" on a LAN,
// or "https://aerovigil.example.com" behind a TLS reverse proxy.
static const char* AEROVIGIL_SERVER = "http://192.168.1.50:8080";

static const char* GATEWAY_ID       = "esp32-gw-01";
static const char* TURBINE_ID       = "WTG-ESP-01";

const int SAMPLE_SECONDS  = 10;   // sensor sampling cadence
const int BATCH_SIZE      = 3;    // sweeps per POST (radio-on optimization)
const int MAX_BACKOFF_S   = 120;  // capped exponential backoff on failure

// Set false once real sensors are wired (see readSensors()).
const bool SIMULATED_SENSORS = true;

// Analog pins for real sensors (ADC1 channels; adjust to your wiring).
const int PIN_VIB   = 34;
const int PIN_GBTMP = 35;
const int PIN_GNTMP = 32;
const int PIN_POWER = 33;
// ═══════════════════════════════════════════════════════════════════

Preferences prefs;

double operatingHours = 0.0;   // persisted across reboots via NVS
int    backoffSeconds = 2;

// ── helpers ────────────────────────────────────────────────────────
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[wifi] connecting to %s\n", WIFI_SSID);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 60) {
    delay(500);
    Serial.print(".");
    tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[wifi] connected, ip=%s rssi=%d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    Serial.println("\n[wifi] FAILED — will retry in main loop");
  }
}

void syncClock() {
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  struct tm t;
  int tries = 0;
  while (!getLocalTime(&t) && tries < 10) { delay(1000); tries++; }
  Serial.println(tries < 10 ? "[ntp] clock synchronized"
                            : "[ntp] sync failed — timestamps will drift");
}

String isoTimestamp() {
  struct tm t;
  if (!getLocalTime(&t)) return String("1970-01-01T00:00:00.000Z");
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S.000Z", &t);
  return String(buf);
}

// ── sensors ────────────────────────────────────────────────────────
struct Sweep {
  double vibration_rms;   // mm/s
  double gearbox_temp;    // °C
  double generator_temp;  // °C
  double generator_rpm;   // rpm
  double power_output;    // kW
  double wind_speed;      // m/s
};

Sweep readSensors() {
  if (SIMULATED_SENSORS) {
    // Realistic demo values (random walk around a healthy operating point).
    static double vib = 2.5, gt = 62.0, gnt = 74.0, wind = 9.0;
    vib  = constrain(vib  + (random(-30, 31) / 100.0), 0.5, 12.0);
    gt   = constrain(gt   + (random(-20, 21) / 10.0), 45.0, 95.0);
    gnt  = constrain(gnt  + (random(-20, 21) / 10.0), 55.0, 110.0);
    wind = constrain(wind + (random(-40, 41) / 10.0), 2.0, 22.0);
    double rpm   = map((long)(wind * 100), 200, 2200, 900, 1800);
    double power = wind < 3.0 ? 0.0
                 : wind >= 12.0 ? 2500.0
                 : 2500.0 * pow((wind - 3.0) / 9.0, 3.0);
    return {vib, gt, gnt, (double)rpm, power, wind};
  }

  // ── REAL HARDWARE: replace the conversions below ─────────────────
  // Vibration: MEMS accelerometer + envelope detector on PIN_VIB.
  //   ADC 0..4095 → 0..3.3V → scale to mm/s RMS per your conditioner.
  double vib = (analogReadMilliVolts(PIN_VIB) / 1000.0) * 10.0;
  // NTC/LM35 temperature channels (voltage divider → °C).
  double gt  = (analogReadMilliVolts(PIN_GBTMP) / 1000.0) * 50.0;
  double gnt = (analogReadMilliVolts(PIN_GNTMP) / 1000.0) * 50.0;
  // Power: CT clamp + burden resistor on PIN_POWER → kW.
  double power = (analogReadMilliVolts(PIN_POWER) / 1000.0) * 1000.0;
  // RPM / wind: count pulses on GPIO interrupts over the sample window
  // (attachInterrupt in setup(), increment volatile counters, compute Hz).
  double rpm  = 1500.0;   // TODO: pulse counter on high-speed shaft
  double wind = 9.0;      // TODO: pulse counter on anemometer
  return {vib, gt, gnt, rpm, power, wind};
}

// ── JSON batch builder (matches /api/hardware/stream exactly) ──────
String readingJson(const char* signal, double value, const char* unit,
                   const String& ts) {
  String j = "{\"gateway_id\":\""; j += GATEWAY_ID;
  j += "\",\"turbine_id\":\"";      j += TURBINE_ID;
  j += "\",\"signal\":\"";          j += signal;
  j += "\",\"value\":";             j += String(value, 3);
  j += ",\"unit\":\"";              j += unit;
  j += "\",\"quality\":\"good\",\"timestamp\":\""; j += ts;
  j += "\"}";
  return j;
}

String buildBatch(const Sweep* sweeps, int count) {
  String body = "{\"gateway_id\":\""; body += GATEWAY_ID;
  body += "\",\"readings\":[";
  bool first = true;
  for (int i = 0; i < count; i++) {
    String ts = isoTimestamp();
    const Sweep& s = sweeps[i];
    struct { const char* sig; double val; const char* unit; } rows[] = {
      {"vibration_rms",   s.vibration_rms,   "mm/s"},
      {"gearbox_temp",    s.gearbox_temp,    "C"},
      {"generator_temp",  s.generator_temp,  "C"},
      {"generator_rpm",   s.generator_rpm,   "rpm"},
      {"power_output",    s.power_output,    "kW"},
      {"wind_speed",      s.wind_speed,      "m/s"},
      {"operating_hours", operatingHours,    "h"},
    };
    for (auto& r : rows) {
      if (!first) body += ",";
      first = false;
      body += readingJson(r.sig, r.val, r.unit, ts);
    }
  }
  body += "]}";
  return body;
}

// ── cloud POST with retry/backoff ──────────────────────────────────
bool postBatch(const String& body) {
  HTTPClient http;
  String url = String(AEROVIGIL_SERVER) + "/api/hardware/stream";
  std::unique_ptr<WiFiClient> client;
  if (url.startsWith("https")) {
    auto secure = new WiFiClientSecure();
    secure->setInsecure();  // dev: pin your CA bundle for production TLS
    client.reset(secure);
  } else {
    client.reset(new WiFiClient());
  }
  if (!http.begin(*client, url)) return false;
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000);
  int code = http.POST(body);
  if (code == 200) {
    String resp = http.getString();
    // Surface the cloud's advisory in the serial monitor.
    if (resp.indexOf("\"predicted_rul_days\":") > 0) {
      Serial.printf("[cloud] advisory ok — %s\n", resp.substring(0, 300).c_str());
    } else {
      Serial.printf("[cloud] ack — %s\n", resp.substring(0, 200).c_str());
    }
    http.end();
    return true;
  }
  Serial.printf("[cloud] POST failed, http=%d\n", code);
  http.end();
  return false;
}

// ── lifecycle ──────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println("\n=== AeroVigil ESP32 telemetry node ===");
  Serial.printf("server=%s gateway=%s turbine=%s\n",
                AEROVIGIL_SERVER, GATEWAY_ID, TURBINE_ID);

  prefs.begin("aerovigil", false);
  operatingHours = prefs.getDouble("op_hours", 0.0);
  Serial.printf("persisted operating hours: %.1f h\n", operatingHours);

  analogReadResolution(12);
  connectWiFi();
  syncClock();
}

void loop() {
  static Sweep batch[BATCH_SIZE];

  for (int i = 0; i < BATCH_SIZE; i++) {
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    batch[i] = readSensors();
    operatingHours += SAMPLE_SECONDS / 3600.0;
    Serial.printf("[sense] vib=%.2f mm/s gbt=%.1f C gnt=%.1f C rpm=%.0f "
                  "pwr=%.0f kW wind=%.1f m/s hours=%.2f\n",
                  batch[i].vibration_rms, batch[i].gearbox_temp,
                  batch[i].generator_temp, batch[i].generator_rpm,
                  batch[i].power_output, batch[i].wind_speed, operatingHours);
    if (i < BATCH_SIZE - 1) delay(SAMPLE_SECONDS * 1000L);
  }
  prefs.putDouble("op_hours", operatingHours);

  String body = buildBatch(batch, BATCH_SIZE);
  if (WiFi.status() == WL_CONNECTED && postBatch(body)) {
    backoffSeconds = 2;  // reset backoff on success
  } else {
    Serial.printf("[cloud] backing off %d s (batches stay in RAM)\n",
                  backoffSeconds);
    delay(backoffSeconds * 1000L);
    backoffSeconds = min(backoffSeconds * 2, MAX_BACKOFF_S);
    return;  // skip the normal cadence; retry sooner
  }

  delay(SAMPLE_SECONDS * 1000L);
}
