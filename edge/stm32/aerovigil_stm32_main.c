/**
 * AeroVigil STM32 telemetry node (reference)
 * ==========================================
 * Connects an STM32 to the AeroVigil cloud via a network coprocessor.
 * STM32 MCUs typically reach WiFi/Ethernet through an AT-command modem
 * (ESP8266 / ESP32 running AT firmware), so this reference implements the
 * common UART AT-bridge transport:
 *
 *     STM32 --UART(AT)--> ESP8266/ESP32 --WiFi--> AeroVigil :8080
 *                                                   POST /api/hardware/stream
 *
 * The AeroVigil JSON frame built here is byte-for-byte the same schema the
 * ESP32 sketch and edge/simulate_device.py send, so the cloud treats all
 * three identically.
 *
 * Target   : any STM32 with HAL (CubeMX-generated project)
 * Periph.  : ADC1 (telemetry), USART2 (modem @115200), TIM for sampling
 * SAFETY   : advisory-only. This node only reports telemetry.
 *
 * Porting  : if your board has native networking (STM32H7 + Ethernet,
 *            X-NUCLEO WiFi, LwIP+MQTT), keep build_aerovigil_frame() and
 *            replace the AT transport with your socket layer.
 */

#include "main.h"      /* CubeMX-generated: handles, HAL init              */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── Configuration ─────────────────────────────────────────────── */
#define AEROVIGIL_HOST     "192.168.1.50"          /* or TLS origin       */
#define AEROVIGIL_PORT     8080
#define AEROVIGIL_PATH     "/api/hardware/stream"
#define GATEWAY_ID         "stm32-gw-01"
#define TURBINE_ID         "WTG-STM-01"
#define SAMPLE_MS          10000
#define MODEM_UART         huart2
#define ADC_HANDLE         hadc1

extern UART_HandleTypeDef MODEM_UART;
extern ADC_HandleTypeDef  ADC_HANDLE;

/* ── Small utilities ───────────────────────────────────────────── */
static void uart_send(const char *s) {
    HAL_UART_Transmit(&MODEM_UART, (uint8_t *)s, strlen(s), 2000);
}

/* Send a command and wait for `expect` in the reply (blocking, simple). */
static int at_cmd(const char *cmd, const char *expect, uint32_t timeout_ms) {
    char rx[256] = {0};
    uint16_t len = 0;
    uart_send(cmd);
    uart_send("\r\n");
    uint32_t start = HAL_GetTick();
    while (HAL_GetTick() - start < timeout_ms) {
        uint8_t c;
        if (HAL_UART_Receive(&MODEM_UART, &c, 1, 10) == HAL_OK && len < sizeof(rx) - 1) {
            rx[len++] = (char)c;
            rx[len] = '\0';
            if (strstr(rx, expect)) return 1;
        }
    }
    return 0;
}

/* ── Telemetry sampling ────────────────────────────────────────── */
typedef struct {
    float vibration_rms;   /* mm/s  */
    float gearbox_temp;    /* deg C */
    float generator_temp;  /* deg C */
    float generator_rpm;   /* rpm   */
    float power_output;    /* kW    */
    float wind_speed;      /* m/s   */
    float operating_hours; /* h     */
} Sweep;

static uint16_t adc_read_channel(uint32_t channel) {
    ADC_ChannelConfTypeDef cfg = {0};
    cfg.Channel = channel;
    cfg.Rank = ADC_REGULAR_RANK_1;
    cfg.SamplingTime = ADC_SAMPLETIME_47CYCLES_5;
    HAL_ADC_ConfigChannel(&ADC_HANDLE, &cfg);
    HAL_ADC_Start(&ADC_HANDLE);
    HAL_ADC_PollForConversion(&ADC_HANDLE, 10);
    uint16_t raw = (uint16_t)HAL_ADC_GetValue(&ADC_HANDLE);
    HAL_ADC_Stop(&ADC_HANDLE);
    return raw;
}

static float operating_hours = 0.0f;

static Sweep read_sensors(void) {
    /* SIMULATED values so the reference runs without wiring. Replace the
       four conversions below with your real ADC/interrupt math.        */
    Sweep s;
    s.vibration_rms = 2.5f + ((rand() % 100) - 50) / 200.0f;
    s.gearbox_temp  = 62.0f + ((rand() % 100) - 50) / 50.0f;
    s.generator_temp= 74.0f + ((rand() % 100) - 50) / 50.0f;
    s.wind_speed    = 9.0f + ((rand() % 100) - 50) / 25.0f;
    s.generator_rpm = 900.0f + s.wind_speed * 50.0f;
    s.power_output  = s.wind_speed < 3.0f ? 0.0f : 2500.0f;

    /* REAL HARDWARE examples (uncomment + calibrate):
       uint16_t vib_raw = adc_read_channel(ADC_CHANNEL_0);
       s.vibration_rms  = vib_raw * 3.3f / 4095.0f * 10.0f;   // mm/s
       uint16_t tmp_raw = adc_read_channel(ADC_CHANNEL_1);
       s.gearbox_temp   = tmp_raw * 3.3f / 4095.0f * 50.0f;   // deg C
       RPM / wind: count timer-capture pulses over SAMPLE_MS.           */
    (void)adc_read_channel;
    return s;
}

/* ── Build the exact AeroVigil JSON frame ──────────────────────── */
static int append_reading(char *dst, int off, const char *signal, float value,
                          const char *unit) {
    return sprintf(dst + off,
        "%s{\"gateway_id\":\"%s\",\"turbine_id\":\"%s\",\"signal\":\"%s\","
        "\"value\":%.3f,\"unit\":\"%s\",\"quality\":\"good\",\"timestamp\":0}",
        off ? "," : "", GATEWAY_ID, TURBINE_ID, signal, (double)value, unit);
}

static int build_aerovigil_frame(char *dst, size_t cap, const Sweep *s) {
    int off = 0;
    off += snprintf(dst + off, cap - off, "{\"gateway_id\":\"%s\",\"readings\":[",
                    GATEWAY_ID);
    off += append_reading(dst, off, "vibration_rms",   s->vibration_rms,   "mm/s");
    off += append_reading(dst, off, "gearbox_temp",    s->gearbox_temp,    "C");
    off += append_reading(dst, off, "generator_temp",  s->generator_temp,  "C");
    off += append_reading(dst, off, "generator_rpm",   s->generator_rpm,   "rpm");
    off += append_reading(dst, off, "power_output",    s->power_output,    "kW");
    off += append_reading(dst, off, "wind_speed",      s->wind_speed,      "m/s");
    off += append_reading(dst, off, "operating_hours", operating_hours,    "h");
    off += snprintf(dst + off, cap - off, "]}");
    /* NOTE: timestamp:0 lets the cloud stamp server time; send an ISO-8601
       string instead if you keep an RTC.                                 */
    return off;
}

/* ── AT-bridge HTTP POST transport ─────────────────────────────── */
static char frame[1024];
static char atbuf[256];

static int post_to_cloud(const char *body, int body_len) {
    if (!at_cmd("AT", "OK", 1000)) return 0;
    if (!at_cmd("AT+CIPMUX=0", "OK", 1000)) return 0;

    snprintf(atbuf, sizeof(atbuf),
             "AT+CIPSTART=\"TCP\",\"%s\",%d", AEROVIGIL_HOST, AEROVIGIL_PORT);
    if (!at_cmd(atbuf, "OK", 5000)) return 0;

    /* Assemble a minimal HTTP/1.1 POST and ship it via CIPSEND. */
    static char http[1400];
    int hlen = snprintf(http, sizeof(http),
        "POST %s HTTP/1.1\r\nHost: %s:%d\r\nContent-Type: application/json\r\n"
        "Content-Length: %d\r\nConnection: close\r\n\r\n%s",
        AEROVIGIL_PATH, AEROVIGIL_HOST, AEROVIGIL_PORT, body_len, body);

    snprintf(atbuf, sizeof(atbuf), "AT+CIPSEND=%d", hlen);
    if (!at_cmd(atbuf, ">", 2000)) { at_cmd("AT+CIPCLOSE", "OK", 1000); return 0; }
    uart_send(http);

    return at_cmd("", "SEND OK", 8000);
}

/* ── Main loop ─────────────────────────────────────────────────── */
int main(void) {
    HAL_Init();
    SystemClock_Config();   /* CubeMX-generated */
    MX_GPIO_Init();
    MX_USART2_UART_Init();  /* 115200 8N1 to the modem */
    MX_ADC1_Init();

    /* Bring the modem up (ESP8266/ESP32 AT firmware). */
    at_cmd("AT+RST", "ready", 3000);
    at_cmd("AT+CWMODE=1", "OK", 1000);
    at_cmd("AT+CWJAP=\"YOUR_WIFI_SSID\",\"YOUR_WIFI_PASSWORD\"", "OK", 15000);

    uint32_t last = HAL_GetTick();
    for (;;) {
        if (HAL_GetTick() - last >= SAMPLE_MS) {
            last = HAL_GetTick();
            Sweep s = read_sensors();
            operating_hours += SAMPLE_MS / 3600000.0f;
            s.operating_hours = operating_hours;

            int len = build_aerovigil_frame(frame, sizeof(frame), &s);
            if (post_to_cloud(frame, len)) {
                HAL_GPIO_TogglePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin); /* ok */
            } else {
                HAL_GPIO_TogglePin(LED_RED_GPIO_Port, LED_RED_Pin);   /* retry */
            }
        }
        HAL_Delay(10);
    }
}
