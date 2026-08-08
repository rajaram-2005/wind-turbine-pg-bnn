"""Hardware telemetry intake for USB-imported and cloud-hosted CSV files.

USB files are read by a client and submitted as text; the service never reads
an arbitrary path on a user's computer. Cloud imports accept HTTPS URLs only
and reject local/private hosts to avoid server-side request forgery.
"""
from __future__ import annotations

import csv
import io
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

REQUIRED_CHANNELS = ("vibration_mms", "temperature_c", "rpm", "oil_viscosity_cst", "load_pct")
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_ROWS = 100_000


def _validate_cloud_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("cloud_url must be an HTTPS URL")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError("cloud_url host could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("cloud_url must not resolve to a private or local address")
    return url


def fetch_cloud_csv(url: str) -> str:
    """Fetch a bounded CSV from a signed public HTTPS cloud URL."""
    _validate_cloud_url(url)
    try:
        with urlopen(url, timeout=15) as response:  # noqa: S310 - URL is validated above
            content = response.read(MAX_CSV_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"could not fetch cloud telemetry: {exc}") from exc
    if len(content) > MAX_CSV_BYTES:
        raise ValueError(f"cloud CSV exceeds {MAX_CSV_BYTES // (1024 * 1024)} MB limit")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("cloud telemetry must be UTF-8 CSV") from exc


def parse_telemetry_csv(csv_text: str) -> tuple[list[dict[str, float]], list[str]]:
    """Validate and normalize a bounded hardware telemetry CSV."""
    if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError(f"CSV exceeds {MAX_CSV_BYTES // (1024 * 1024)} MB limit")
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV must include a header row")
    missing = [column for column in REQUIRED_CHANNELS if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"CSV missing hardware telemetry columns: {missing}")
    rows: list[dict[str, float]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(rows) >= MAX_ROWS:
            raise ValueError(f"CSV exceeds {MAX_ROWS} row limit")
        try:
            rows.append({column: float(row[column]) for column in REQUIRED_CHANNELS})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric telemetry at CSV row {row_number}") from exc
    if not rows:
        raise ValueError("CSV contains no telemetry rows")
    return rows, list(reader.fieldnames)


def import_hardware_telemetry(*, source: str, csv_text: str | None = None, cloud_url: str | None = None) -> dict[str, Any]:
    """Import a USB/browser CSV or cloud-hosted CSV and return a safe preview."""
    if source == "usb":
        if not csv_text:
            raise ValueError("usb import requires csv_text read from a selected file")
        raw = csv_text
    elif source == "cloud":
        if not cloud_url:
            raise ValueError("cloud import requires cloud_url")
        raw = fetch_cloud_csv(cloud_url)
    else:
        raise ValueError("source must be 'usb' or 'cloud'")
    rows, columns = parse_telemetry_csv(raw)
    return {
        "source": source,
        "rows_imported": len(rows),
        "columns": columns,
        "latest_telemetry": rows[-1],
        "advisory_only": True,
    }
