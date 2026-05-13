"""Global fn-key monitor and recording state transitions."""

from __future__ import annotations

import sys
import threading
import time

import CoreFoundation
import Quartz
from loguru import logger

from . import config as cfg
from .audio import _show_recording_hud, _start_mic_stream, cancel_recording, stop_and_transcribe
from .hud import _route_mouse_to_hud, _tooltip, set_hud
from .state import begin_recording, state


FN_KEYCODE = 63
FN_FLAG = 0x800000
prev_fn_down = False
_fn_press_time = 0.0
_shift_held = False
_option_held = False


def cg_event_callback(proxy, event_type, event, refcon):
    global prev_fn_down, _fn_press_time, _shift_held, _option_held

    if event_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        _route_mouse_to_hud(event_type, event)
        return event

    keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
    flags_now = Quartz.CGEventGetFlags(event)

    prev_shift_held = _shift_held
    prev_option_held = _option_held
    _shift_held = bool(flags_now & Quartz.kCGEventFlagMaskShift)
    _option_held = bool(flags_now & Quartz.kCGEventFlagMaskAlternate)

    with state.lock:
        recording = state.recording
        transcribing = state.transcribing

    if recording and _shift_held and not prev_shift_held:
        with state.lock:
            if cfg.NOTES_CFG:
                state.note_mode = not state.note_mode
                state.predict_mode = False
                note_on = state.note_mode
            else:
                note_on = None
        if note_on is None:
            logger.warning("notes plugin not installed - shift+fn disabled")
        else:
            set_hud(True, "note" if note_on else "recording")
            logger.debug(f"note_mode={'on' if note_on else 'off'}")

    if recording and _option_held and not prev_option_held:
        with state.lock:
            state.predict_mode = not state.predict_mode
            state.note_mode = False
            predict_on = state.predict_mode
        set_hud(True, "predict" if predict_on else "recording")
        logger.debug(f"predict_mode={'on' if predict_on else 'off'}")

    if event_type == Quartz.kCGEventKeyDown and keycode == 53 and (recording or transcribing):
        threading.Thread(target=cancel_recording, daemon=True).start()
        return event

    if keycode in (36, 76) and recording and event_type == Quartz.kCGEventKeyDown:
        with state.lock:
            state.auto_send = not state.auto_send
            auto_send = state.auto_send
        logger.debug(f"auto_send={'on' if auto_send else 'off'}")

        def _undo_enter():
            time.sleep(0.05)
            src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            ev = Quartz.CGEventCreateKeyboardEvent(src, 0x06, True)
            Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            ev = Quartz.CGEventCreateKeyboardEvent(src, 0x06, False)
            Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

        threading.Thread(target=_undo_enter, daemon=True).start()
        return event

    if keycode != FN_KEYCODE:
        return event

    is_down = bool(flags_now & FN_FLAG)

    if is_down and not prev_fn_down:
        _fn_press_time = time.time()
        if _shift_held and cfg.NOTES_CFG:
            note = True
            predict = False
        elif _option_held:
            predict = True
            note = False
        else:
            predict = False
            note = False
        should_start_mic = begin_recording(predict=predict, note=note, auto_send=False)
        logger.debug(
            f"Mode set shift={_shift_held} option={_option_held} note={note} predict={predict}"
        )
        if should_start_mic:
            threading.Thread(target=_start_mic_stream, daemon=True).start()

            def _show_hud_delayed():
                time.sleep(0.20)
                with state.lock:
                    active = state.recording and not state.cancelled
                if active:
                    _show_recording_hud()

            threading.Thread(target=_show_hud_delayed, daemon=True).start()
    elif not is_down and prev_fn_down:
        now = time.time()
        held = now - _fn_press_time
        with state.lock:
            state.fn_release_ts = now
            recording = state.recording
        if held < 0.20:
            threading.Thread(
                target=lambda: cancel_recording(skip_hud=True, quiet=True),
                daemon=True,
            ).start()
        elif recording:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()

    prev_fn_down = is_down
    return event


def install_monitor():
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp),
        cg_event_callback,
        None,
    )
    if tap is None:
        logger.error("CGEventTap failed. Check Accessibility permission.")
        sys.exit(1)

    src = CoreFoundation.CFMachPortCreateRunLoopSource(None, tap, 0)
    CoreFoundation.CFRunLoopAddSource(
        CoreFoundation.CFRunLoopGetMain(), src, CoreFoundation.kCFRunLoopCommonModes,
    )
    Quartz.CGEventTapEnable(tap, True)
    logger.info("Hotkey monitor installed.")
    return tap


def _tap_health_check(tap):
    """Poll CGEventTap health every 7s. Daemon thread exits with main process."""
    tap_was_disabled = False
    while True:
        time.sleep(7)
        try:
            enabled = Quartz.CGEventTapIsEnabled(tap)
        except Exception:
            continue
        if not enabled and not tap_was_disabled:
            tap_was_disabled = True
            try:
                Quartz.CGEventTapEnable(tap, True)
            except Exception:
                pass
            cancel_recording(skip_hud=True, quiet=True)
            set_hud(True, mode="error_fatal", tooltip=_tooltip("accessibility_revoked"))
            with state.lock:
                state.health_monitor_owns_hud = True
            logger.warning("Accessibility revoked -- attempting re-enable")
        elif not enabled and tap_was_disabled:
            try:
                Quartz.CGEventTapEnable(tap, True)
            except Exception:
                pass
        elif enabled and tap_was_disabled:
            tap_was_disabled = False
            with state.lock:
                owns_hud = state.health_monitor_owns_hud
                fatal = state.hud_error_mode == "error_fatal"
                state.health_monitor_owns_hud = False
            if owns_hud and fatal:
                set_hud(False)
            logger.info("Accessibility restored.")
