"""Entry point: `python -m govori` runs the daemon."""

from __future__ import annotations

import threading

import AppKit
import sounddevice as sd
from loguru import logger

from . import macos
from .cli import main as cli_dispatch
from .hotkey import _tap_health_check, install_monitor
from .hud import setup_hud
from .predict import setup_predict


def _run_daemon():
    """The post-CLI part: set up Cocoa app, install hotkey, poll runloop."""
    logger.info("Govori started. Hold fn to record.")
    macos.install_signal_handlers()

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    setup_hud()
    setup_predict()
    tap = install_monitor()
    threading.Thread(target=_tap_health_check, args=(tap,), daemon=True).start()

    try:
        sd.query_devices(kind="input")
    except sd.PortAudioError:
        logger.warning("No microphone detected. Plug one in before recording.")

    run_loop = AppKit.NSRunLoop.mainRunLoop()
    while not macos.is_shutdown_requested():
        run_loop.runMode_beforeDate_(
            AppKit.NSDefaultRunLoopMode,
            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.5),
        )
    macos.perform_shutdown()
    AppKit.NSApp.terminate_(None)


def main():
    cli_dispatch()
    from .macos import _ensure_singleton

    _ensure_singleton()
    _run_daemon()


if __name__ == "__main__":
    main()
