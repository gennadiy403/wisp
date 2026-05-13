"""macOS integration helpers: paste, keys, singleton, and shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import AppKit
import Quartz
from loguru import logger

from .state import state

_shutdown_event = threading.Event()


def paste_text(text):
    pb = AppKit.NSPasteboard.generalPasteboard()
    old_clipboard = pb.stringForType_(AppKit.NSPasteboardTypeString)
    pb.clearContents()
    pb.setString_forType_(text, AppKit.NSPasteboardTypeString)
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    ev = Quartz.CGEventCreateKeyboardEvent(src, 0x09, True)
    Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
    ev = Quartz.CGEventCreateKeyboardEvent(src, 0x09, False)
    Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def _restore():
        time.sleep(0.15)
        pb.clearContents()
        if old_clipboard:
            pb.setString_forType_(old_clipboard, AppKit.NSPasteboardTypeString)

    threading.Thread(target=_restore, daemon=True).start()


def _press_enter():
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    ev = Quartz.CGEventCreateKeyboardEvent(src, 0x24, True)
    Quartz.CGEventSetFlags(ev, 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
    ev = Quartz.CGEventCreateKeyboardEvent(src, 0x24, False)
    Quartz.CGEventSetFlags(ev, 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def _delete_chars(n):
    """Send n Backspace key events to erase the previously-pasted text."""
    if n <= 0:
        return
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for _ in range(n):
        ev = Quartz.CGEventCreateKeyboardEvent(src, 0x33, True)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        ev = Quartz.CGEventCreateKeyboardEvent(src, 0x33, False)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def request_shutdown(*_):
    with state.lock:
        state.shutdown_requested = True
    _shutdown_event.set()


def install_signal_handlers():
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def is_shutdown_requested():
    return _shutdown_event.is_set()


def perform_shutdown():
    logger.info("Shutdown requested - cleaning up")
    with state.lock:
        stream = state.audio_stream
        state.audio_stream = None
        state.recording = False
        state.transcribing = False
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            logger.warning(f"Audio stream close failed: {e}")
    logger.complete()


def _find_other_govori_pids():
    """Return PIDs of other running govori daemons (excluding self)."""
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", r"govori\.py|-m govori"], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids = []
    for line in out.strip().splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid == my_pid:
            continue
        pids.append(pid)
    return pids


def _ensure_singleton():
    """If another govori daemon is running, offer to kill it and take over."""
    pids = _find_other_govori_pids()
    if not pids:
        return
    pid_list = ", ".join(str(p) for p in pids)
    logger.warning(f"Govori is already running (PID {pid_list}).")
    if not sys.stdin.isatty():
        logger.error("Another instance is active - refusing to start.")
        sys.exit(1)
    try:
        ans = input("  Kill it and take over? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        logger.info("")
        sys.exit(1)
    if ans in ("n", "no", "н", "нет"):
        logger.info("Aborted.")
        sys.exit(1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as e:
            logger.error(f"Cannot stop PID {pid}: {e}")
            sys.exit(1)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _find_other_govori_pids():
            break
        time.sleep(0.1)
    remaining = _find_other_govori_pids()
    if remaining:
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        time.sleep(0.3)
        remaining = _find_other_govori_pids()
    if remaining:
        logger.error(f"Failed to stop PID(s) {remaining}. Aborting.")
        sys.exit(1)
    logger.info("Replaced previous instance.")
