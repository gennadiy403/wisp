"""PERF-01 instrumentation: contextmanager-based spans + BENCH_MODE summary.

Stages emit to loguru with extra={"stage": name, "elapsed_ms": ms}. The
`logging_setup` bench sink picks these up and writes JSON lines to
bench.jsonl when BENCH_MODE=1. This module additionally accumulates
samples for an atexit-printed p50/p95 table.

Do NOT import this from modules that run at startup before logging
is configured — span() emits via loguru, which falls back to the
default stderr sink if configure_logging hasn't run. Acceptable, but
noisy.

# Important: BENCH_MODE is read once at module import time. Set it BEFORE
# invoking `python -m govori`, e.g., `BENCH_MODE=1 python -m govori`.
# Exporting it after the process starts has no effect — the env check
# has already happened.
"""

from __future__ import annotations

import atexit
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from loguru import logger


BENCH_MODE = os.environ.get("BENCH_MODE") == "1"

_samples: dict[str, list[float]] = defaultdict(list)


@contextmanager
def span(name: str, **extra) -> Iterator[None]:
    """Time the wrapped block. Emits a DEBUG event with stage + elapsed_ms.

    When BENCH_MODE=1, also accumulates the sample for the atexit summary.

    Usage:
        with span("encode"):
            ...PyAV encode loop...

        with span("api_call", provider="groq", model="whisper-large-v3-turbo"):
            result = client.audio.transcriptions.create(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.bind(stage=name, elapsed_ms=round(elapsed_ms, 2), **extra).debug(
            f"span {name} {elapsed_ms:.1f}ms"
        )
        if BENCH_MODE:
            _samples[name].append(elapsed_ms)


def record_event(name: str, elapsed_ms: float, **extra) -> None:
    """For spans that can't be a contextmanager (e.g., the fn-release-to-stop
    gap measured across thread boundaries). Emits the same shape of event."""
    logger.bind(stage=name, elapsed_ms=round(elapsed_ms, 2), **extra).debug(
        f"span {name} {elapsed_ms:.1f}ms"
    )
    if BENCH_MODE:
        _samples[name].append(elapsed_ms)


def _print_summary() -> None:
    if not BENCH_MODE or not _samples:
        return
    print("\n── PERF-01 summary ─────────────────────────────────")
    print(f"{'stage':<25} {'n':>4} {'p50':>8} {'p95':>8} {'mean':>8}")
    for stage, samples in sorted(_samples.items()):
        s = sorted(samples)
        n = len(s)
        p50 = s[n // 2]
        p95 = s[int(n * 0.95)] if n >= 20 else s[-1]
        mean = sum(s) / n
        print(f"{stage:<25} {n:>4} {p50:>7.1f}ms {p95:>7.1f}ms {mean:>7.1f}ms")


atexit.register(_print_summary)
