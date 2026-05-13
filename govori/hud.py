"""Cocoa HUD windows, tooltip text, and click-to-retry routing."""

from __future__ import annotations

import threading

import AppKit
import Quartz
from loguru import logger

from . import config as cfg
from .state import state

TOOLTIP_STRINGS = {
    "en": {
        "api_timeout": "Transcription timed out. Click to retry.",
        "api_network": "Connection lost",
        "api_server": "Server error. Click to retry.",
        "retry_attempt": "Retrying... (attempt {n}/{total})",
        "attempt_progress": "No connection {n}/{total}{dots}",
        "retry_exhausted": "Transcription failed. Try recording again.",
        "no_mic": "No microphone found.",
        "mic_denied": "Microphone access denied.",
        "accessibility_revoked": "Accessibility revoked \u2014 hotkeys disabled.",
    },
    "ru": {
        "api_timeout": "Транскрипция не ответила. Нажми для повтора.",
        "api_network": "\u041e\u0431\u0440\u044b\u0432 \u0441\u043e\u0435\u0434\u0438\u043d\u0435\u043d\u0438\u044f",
        "api_server": "Ошибка сервера. Нажми для повтора.",
        "retry_attempt": "Повтор... (попытка {n}/{total})",
        "attempt_progress": "\u041d\u0435\u0442 \u0441\u043e\u0435\u0434\u0438\u043d\u0435\u043d\u0438\u044f {n}/{total}{dots}",
        "retry_exhausted": "Не удалось распознать. Попробуй записать ещё раз.",
        "no_mic": "Микрофон не найден.",
        "mic_denied": "Доступ к микрофону запрещён.",
        "accessibility_revoked": "Доступ отозван \u2014 горячие клавиши отключены.",
    },
}


def _tooltip(key, **kwargs):
    """Get localized tooltip text by key."""
    lang = cfg.CONFIG.language
    if lang not in TOOLTIP_STRINGS:
        lang = "en"
    text = TOOLTIP_STRINGS[lang].get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# ── HUD ───────────────────────────────────────────────────────────────────────
hud_window = None
hud_label  = None
hud_container = None

_HUD_S = 32


def setup_hud():
    global hud_window, hud_label, hud_container

    screen = AppKit.NSScreen.mainScreen().frame()
    # Position: bottom-left corner (aligned with optional Hammerspoon status HUD).
    x = 6
    y = 0
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    win = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(x, y, _HUD_S, _HUD_S), style,
        AppKit.NSBackingStoreBuffered, False,
    )
    win.setLevel_(AppKit.NSFloatingWindowLevel + 1)
    win.setOpaque_(False)
    win.setBackgroundColor_(AppKit.NSColor.clearColor())
    win.setIgnoresMouseEvents_(True)
    win.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )

    container = AppKit.NSView.alloc().initWithFrame_(
        AppKit.NSMakeRect(0, 0, _HUD_S, _HUD_S)
    )
    container.setWantsLayer_(True)
    container.layer().setBackgroundColor_(
        AppKit.NSColor.colorWithRed_green_blue_alpha_(0.10, 0.10, 0.10, 0.85).CGColor()
    )
    container.layer().setCornerRadius_(_HUD_S / 2)

    label = AppKit.NSTextField.labelWithString_("●")
    label.setFrame_(AppKit.NSMakeRect(0, (_HUD_S - 18) / 2, _HUD_S, 18))
    label.setFont_(AppKit.NSFont.systemFontOfSize_(14))
    label.setTextColor_(AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0))
    label.setAlignment_(AppKit.NSTextAlignmentCenter)
    container.addSubview_(label)

    pulse = Quartz.CABasicAnimation.animationWithKeyPath_("opacity")
    pulse.setFromValue_(1.0)
    pulse.setToValue_(0.4)
    pulse.setDuration_(0.8)
    pulse.setAutoreverses_(True)
    pulse.setRepeatCount_(float('inf'))
    pulse.setTimingFunction_(
        Quartz.CAMediaTimingFunction.functionWithName_(Quartz.kCAMediaTimingFunctionEaseInEaseOut)
    )
    label.setWantsLayer_(True)
    label.layer().addAnimation_forKey_(pulse, "pulse")

    win.contentView().addSubview_(container)

    hud_window = win
    hud_label  = label
    hud_container = container

    _setup_tooltip()
    _setup_countdown()


# ── Countdown digit panel (floats above HUD during retry attempts) ───────────
_countdown_panel = None
_countdown_label = None

_COUNTDOWN_W = _HUD_S
_COUNTDOWN_H = 18


def _setup_countdown():
    """Small transparent panel that shows a single white digit above the HUD."""
    global _countdown_panel, _countdown_label
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(6, _HUD_S + 2, _COUNTDOWN_W, _COUNTDOWN_H), style,
        AppKit.NSBackingStoreBuffered, False,
    )
    panel.setLevel_(AppKit.NSFloatingWindowLevel + 1)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setIgnoresMouseEvents_(True)
    panel.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    label = AppKit.NSTextField.labelWithString_("")
    label.setFrame_(AppKit.NSMakeRect(0, 0, _COUNTDOWN_W, _COUNTDOWN_H))
    label.setFont_(AppKit.NSFont.boldSystemFontOfSize_(13))
    label.setTextColor_(
        AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.65)
    )
    label.setAlignment_(AppKit.NSTextAlignmentCenter)
    panel.contentView().addSubview_(label)
    panel.orderOut_(None)
    _countdown_panel = panel
    _countdown_label = label


def _show_countdown(count):
    if _countdown_panel is None:
        return
    _countdown_label.setStringValue_(str(count))
    _countdown_panel.orderFrontRegardless()


def _hide_countdown():
    if _countdown_panel is not None:
        _countdown_panel.orderOut_(None)


# ── Tooltip companion panel ──────────────────────────────────────────────────
_tooltip_panel = None
_tooltip_label = None


_TOOLTIP_X = 42   # HUD ends at x=38 (6+32) → small gap keeps pill edges readable
_TOOLTIP_W_MAX = 380   # wrap point for very long messages
_TOOLTIP_PAD_X = 12
_TOOLTIP_PAD_Y = 7

# Tooltip background tints tie the plate to the HUD state so the pair reads as
# one status element (Practical UI: "Use system colours to indicate status").
_TOOLTIP_BG_BY_STATE = {
    "error_retryable": (0.30, 0.22, 0.08, 0.70),  # amber, semi-transparent
    "error_fatal":     (0.32, 0.10, 0.10, 0.72),  # red, semi-transparent
    "transcribing":    (0.10, 0.10, 0.10, 0.70),  # neutral
}
_TOOLTIP_BORDER_BY_STATE = {
    "error_retryable": (1.0, 0.75, 0.25, 0.35),
    "error_fatal":     (1.0, 0.35, 0.35, 0.40),
    "transcribing":    (1.0, 1.0, 1.0, 0.08),
}


def _setup_tooltip():
    """Create tooltip companion NSPanel positioned next to HUD."""
    global _tooltip_panel, _tooltip_label
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(_TOOLTIP_X, 0, _TOOLTIP_W_MAX, _HUD_S), style,
        AppKit.NSBackingStoreBuffered, False,
    )
    panel.setLevel_(AppKit.NSFloatingWindowLevel + 1)
    panel.setOpaque_(False)
    # Window background must be clear; all visuals live on the content layer so
    # the rounded border renders once, not stacked over a rectangular window fill.
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setHasShadow_(False)
    panel.contentView().setWantsLayer_(True)
    layer = panel.contentView().layer()
    layer.setCornerRadius_(_HUD_S / 2)  # match HUD roundness → pill shape
    layer.setMasksToBounds_(True)
    layer.setBorderWidth_(1.0)
    layer.setBackgroundColor_(
        AppKit.NSColor.colorWithRed_green_blue_alpha_(0.10, 0.10, 0.10, 0.70).CGColor()
    )
    layer.setBorderColor_(
        AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.12).CGColor()
    )
    panel.setIgnoresMouseEvents_(True)
    panel.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    label = AppKit.NSTextField.labelWithString_("")
    label.setFont_(AppKit.NSFont.systemFontOfSize_(12))
    label.setTextColor_(AppKit.NSColor.colorWithRed_green_blue_alpha_(0.95, 0.95, 0.95, 1.0))
    label.setPreferredMaxLayoutWidth_(_TOOLTIP_W_MAX - 2 * _TOOLTIP_PAD_X)
    label.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
    panel.contentView().addSubview_(label)
    panel.orderOut_(None)
    _tooltip_panel = panel
    _tooltip_label = label


def _show_tooltip(text, mode="transcribing"):
    """Show tooltip panel with text. Must be called on main queue.

    Height grows with wrapped content; the panel stays vertically centered on the
    HUD circle so the two read as one paired status element (Fitts's law).
    Background tint follows `mode` to tie the plate to the HUD status colour.
    """
    if _tooltip_panel is None:
        return

    bg = _TOOLTIP_BG_BY_STATE.get(mode, _TOOLTIP_BG_BY_STATE["transcribing"])
    border = _TOOLTIP_BORDER_BY_STATE.get(mode, _TOOLTIP_BORDER_BY_STATE["transcribing"])
    layer = _tooltip_panel.contentView().layer()
    layer.setBackgroundColor_(
        AppKit.NSColor.colorWithRed_green_blue_alpha_(*bg).CGColor()
    )
    layer.setBorderColor_(
        AppKit.NSColor.colorWithRed_green_blue_alpha_(*border).CGColor()
    )

    # Measure text size precisely via attributed string — sizeToFit with
    # preferredMaxLayoutWidth can cache stale values across calls.
    attrs = {AppKit.NSFontAttributeName: _tooltip_label.font()}
    measured = AppKit.NSString.stringWithString_(text).sizeWithAttributes_(attrs)
    text_w = int(measured.width) + 4  # +4 antialiasing slack
    text_h = int(measured.height) + 2
    _tooltip_label.setStringValue_(text)
    panel_w = min(_TOOLTIP_W_MAX, text_w + 2 * _TOOLTIP_PAD_X)
    h = max(_HUD_S, text_h + 2 * _TOOLTIP_PAD_Y)
    label_y = (h - text_h) // 2
    _tooltip_label.setFrame_(
        AppKit.NSMakeRect(_TOOLTIP_PAD_X, label_y, panel_w - 2 * _TOOLTIP_PAD_X, text_h)
    )
    panel_y = int(_HUD_S / 2 - h / 2)
    _tooltip_panel.setFrame_display_(
        AppKit.NSMakeRect(_TOOLTIP_X, panel_y, panel_w, h), True
    )
    _tooltip_panel.orderFrontRegardless()


def _hide_tooltip():
    """Hide tooltip panel. Must be called on main queue."""
    if _tooltip_panel is not None:
        _tooltip_panel.orderOut_(None)


# ── HUD click + cursor routing ───────────────────────────────────────────────


def _hud_click_action():
    if state.hud_error_mode != "error_retryable":
        return
    with state.lock:
        if state.retry_buffer is None or state.retry_in_progress:
            retry_exhausted = False
            retry_attempt = None
        elif state.retry_count >= 3:
            retry_exhausted = True
            retry_attempt = None
        else:
            state.retry_count += 1
            state.retry_in_progress = True
            retry_exhausted = False
            retry_attempt = state.retry_count
    if retry_exhausted:
        set_hud(True, mode="error_fatal", tooltip=_tooltip("retry_exhausted"))
        return
    if retry_attempt is None:
        return
    set_hud(True, mode="transcribing", tooltip=_tooltip("retry_attempt", n=retry_attempt, total=3))
    from . import transcribe
    threading.Thread(target=transcribe.retry_transcription, daemon=True).start()

_hud_pressed = False


def _hud_apply_press(pressed):
    """Visual press feedback: white fill + dark icon while button is held."""
    if hud_container is None or hud_label is None:
        return
    if pressed:
        hud_container.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.95).CGColor()
        )
        hud_label.setTextColor_(
            AppKit.NSColor.colorWithRed_green_blue_alpha_(0.15, 0.15, 0.15, 1.0)
        )
    else:
        hud_container.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithRed_green_blue_alpha_(0.10, 0.10, 0.10, 0.85).CGColor()
        )
        # Restore icon color for the current error mode
        if state.hud_error_mode == "error_retryable":
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.85, 0.3, 1.0)
            )
        elif state.hud_error_mode == "error_fatal":
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0)
            )


def _point_inside_hud(event):
    """Return True if the CG event location falls inside the HUD panel frame."""
    if hud_window is None or not hud_window.isVisible():
        return False
    loc = Quartz.CGEventGetLocation(event)
    frame = hud_window.frame()
    screen_h = AppKit.NSScreen.mainScreen().frame().size.height
    y_cocoa = screen_h - loc.y  # CG is top-origin, Cocoa is bottom-origin
    return (
        frame.origin.x <= loc.x <= frame.origin.x + frame.size.width
        and frame.origin.y <= y_cocoa <= frame.origin.y + frame.size.height
    )


def _route_mouse_to_hud(event_type, event):
    """Route clicks to the HUD because its non-activating NSPanel can't receive
    native mouse events. Cursor changes aren't possible for background apps on
    macOS — press animation carries the interactive affordance instead.

    mouseDown inside → fill HUD white. mouseUp inside → fire retry.
    mouseUp outside (drag-out) cancels the press silently.
    """
    global _hud_pressed
    is_error = state.hud_error_mode in ("error_retryable", "error_fatal")
    inside = is_error and _point_inside_hud(event)

    if event_type == Quartz.kCGEventLeftMouseDown:
        if inside:
            _hud_pressed = True
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(
                lambda: _hud_apply_press(True)
            )
        return

    if event_type == Quartz.kCGEventLeftMouseUp:
        if _hud_pressed:
            was_pressed = _hud_pressed
            _hud_pressed = False
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(
                lambda: _hud_apply_press(False)
            )
            # Only fire action if mouse is still over the HUD at release
            if was_pressed and inside:
                AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_hud_click_action)
        return




def set_hud(visible, mode="recording", tooltip=None, count=None):
    def _update():
        is_error = mode in ("error_retryable", "error_fatal")

        if mode == "recording":
            hud_label.setStringValue_("●")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0)
            )
        elif mode == "transcribing":
            hud_label.setStringValue_("◎")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.85, 0.3, 1.0)
            )
        elif mode == "countdown":
            # "Connection broken" glyph: downwards zigzag arrow reads as a severed
            # signal line. Warm warning orange signals the disconnected state.
            hud_label.setStringValue_("\u21af")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.55, 0.25, 1.0)
            )
            hud_label.setWantsLayer_(True)
            hud_label.layer().removeAnimationForKey_("pulse")
        elif mode == "predict":
            hud_label.setStringValue_("✦")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.7, 0.4, 1.0, 1.0)
            )
        elif mode == "note":
            hud_label.setStringValue_("✎")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.4, 0.9, 0.6, 1.0)
            )
        elif mode == "note_saved":
            hud_label.setStringValue_("✓")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(0.4, 1.0, 0.5, 1.0)
            )
        elif mode == "note_error":
            hud_label.setStringValue_("✗")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0)
            )
        elif mode == "error_retryable":
            hud_label.setStringValue_("\u21bb")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.85, 0.3, 1.0)
            )
            # Slow pulse: opacity 1.0->0.6, duration 1.2s
            hud_label.setWantsLayer_(True)
            pulse = Quartz.CABasicAnimation.animationWithKeyPath_("opacity")
            pulse.setFromValue_(1.0)
            pulse.setToValue_(0.6)
            pulse.setDuration_(1.2)
            pulse.setAutoreverses_(True)
            pulse.setRepeatCount_(float('inf'))
            pulse.setTimingFunction_(
                Quartz.CAMediaTimingFunction.functionWithName_(
                    Quartz.kCAMediaTimingFunctionEaseInEaseOut
                )
            )
            hud_label.layer().addAnimation_forKey_(pulse, "pulse")
            state.hud_error_mode = "error_retryable"
        elif mode == "error_fatal":
            hud_label.setStringValue_("\u2717")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0)
            )
            # Static -- remove any existing animation
            hud_label.setWantsLayer_(True)
            hud_label.layer().removeAnimationForKey_("pulse")
            state.hud_error_mode = "error_fatal"

        # Mouse events: clickable for error modes, ignored for all others
        if is_error:
            hud_window.setIgnoresMouseEvents_(False)
        else:
            hud_window.setIgnoresMouseEvents_(True)
            if not tooltip:
                _hide_tooltip()
            state.hud_error_mode = None
            # Remove error pulse animation when leaving error mode
            hud_label.setWantsLayer_(True)
            hud_label.layer().removeAnimationForKey_("pulse")

        # Countdown digit panel visibility follows mode
        if mode == "countdown" and count is not None:
            _show_countdown(count)
        else:
            _hide_countdown()

        if visible:
            hud_window.setFrameOrigin_(AppKit.NSMakePoint(6, 0))
            hud_window.orderFrontRegardless()
        else:
            hud_window.orderOut_(None)
            _hide_countdown()

    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_update)

    # Tooltip plate intentionally disabled — the HUD icon alone carries state.
    # The `tooltip` parameter stays accepted so call sites don't need to change.
    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_hide_tooltip)
