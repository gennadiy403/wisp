"""Loguru setup for console, daemon log, and benchmark spans."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


_EVENT_SYMBOLS = {
    "rec_start": "●",
    "rec_stop": "■",
    "transcript": "→",
    "note_saved": "✎",
    "merge": "⇪",
    "hallucination": "·",
}

_LEVEL_TO_SYMBOL = {
    "DEBUG": " ",
    "INFO": "●",
    "SUCCESS": "✓",
    "WARNING": "⚠",
    "ERROR": "✗",
    "CRITICAL": "✗",
}


def _console_sink(message):
    record = message.record
    event = record["extra"].get("event")
    symbol = _EVENT_SYMBOLS.get(event) or _LEVEL_TO_SYMBOL.get(record["level"].name, "·")
    sys.stdout.write(f"{symbol} {record['message']}\n")
    sys.stdout.flush()


def configure_logging(log_dir: Path, bench_mode: bool = False) -> None:
    logger.remove()
    logger.add(_console_sink, level="INFO")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "govori.log",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra} | {message}",
            encoding="utf-8",
            enqueue=True,
            colorize=False,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize file logging: {e}")

    if bench_mode:
        logger.add(
            log_dir / "bench.jsonl",
            rotation="50 MB",
            level="DEBUG",
            serialize=True,
            filter=lambda r: "stage" in r["extra"],
        )
