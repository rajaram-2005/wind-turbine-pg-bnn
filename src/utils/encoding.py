"""UTF-8 I/O helpers for the CLI entrypoints.

Python's default text encoding is locale-dependent (``cp1252`` on Windows,
``ascii``/others in some containers), so reading UTF-8 payloads or writing
non-ASCII report text (``σ``, ``⚠️``, ``°C``, em dashes) can raise
``UnicodeDecodeError`` / ``UnicodeEncodeError``. These helpers pin standard
streams to UTF-8 explicitly.
"""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Reconfigure stdin/stdout/stderr to UTF-8 where the stream supports it.

    Best-effort: streams that do not support ``reconfigure`` (e.g. test
    capture objects or non-``TextIOWrapper`` streams) are left untouched.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass
