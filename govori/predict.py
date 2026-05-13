"""Predict/rephrase mode UI."""
from __future__ import annotations
import json
import threading
import AppKit
from loguru import logger
from . import config as cfg
from .hud import set_hud
from .macos import _delete_chars, paste_text
from .transcribe import _get_predict_client
_predict_controller = None

def generate_rephrasings(text):
    """Generate 3 alternative phrasings of the given text, preserving meaning."""
    try:
        resp = _get_predict_client().chat.completions.create(model=cfg.CONFIG.predict_model, messages=[{'role': 'system', 'content': 'You are a rephrasing assistant. Given a piece of text, produce 3 distinct alternative phrasings that preserve the original meaning but vary in wording, tone, or structure. Keep roughly the same length and the SAME language as the input. Do not add or remove information. Return JSON: {"rephrasings": ["...", "...", "..."]}'}, {'role': 'user', 'content': text}], response_format={'type': 'json_object'}, max_tokens=400, temperature=0.7)
        data = json.loads(resp.choices[0].message.content)
        items = data.get('rephrasings', [])
        if isinstance(items, list) and len(items) >= 1:
            return [str(v) for v in items[:3]]
    except Exception as e:
        logger.info(f'Rephrase error: {e}')
    return []

class PredictController(AppKit.NSObject):
    _rephrasings = []
    _pasted_len = 0

    def pickRephrasing_(self, sender):
        idx = sender.tag()
        if 0 <= idx < len(self._rephrasings):
            text = self._rephrasings[idx]
            n = self._pasted_len
            logger.info(f'✦ rephrase: {text}')

            def _replace():
                _delete_chars(n)
                paste_text(text + ' ')
            threading.Thread(target=_replace, daemon=True).start()

def setup_predict():
    global _predict_controller
    _predict_controller = PredictController.alloc().init()
    return _predict_controller

def show_predict_menu(original_text):
    """Generate rephrasings and show NSMenu. The original text has already
    been pasted by stop_and_transcribe; if the user picks a rephrasing we
    delete those chars and paste the replacement. On dismiss we leave the
    original in place."""
    set_hud(True, 'predict')
    rephrasings = generate_rephrasings(original_text)
    set_hud(False)
    if not rephrasings:
        logger.info('(no rephrasings — keeping original)')
        return
    _predict_controller._rephrasings = rephrasings
    _predict_controller._pasted_len = len(original_text) + 1
    menu = AppKit.NSMenu.alloc().init()
    menu.setAutoenablesItems_(False)
    menu.setMinimumWidth_(300)
    menu.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark))
    for i, reph in enumerate(rephrasings):
        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(reph, 'pickRephrasing:', str(i + 1))
        item.setTarget_(_predict_controller)
        item.setEnabled_(True)
        item.setTag_(i)
        item.setKeyEquivalentModifierMask_(0)
        menu.addItem_(item)
    loc = AppKit.NSEvent.mouseLocation()
    menu.popUpMenuPositioningItem_atLocation_inView_(None, loc, None)
