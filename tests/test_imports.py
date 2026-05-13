"""Smoke test: each module imports without side effects.

Phase 3 (DIST-04) will replace this with proper pytest suites.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports_have_no_side_effects():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import govori.config
        import govori.state
        import govori.logging_setup
        import govori.transcribe
        import govori.audio
        import govori.notes
        import govori.macos
        import govori.hud
        import govori.predict
        import govori.hotkey
        import govori.onboarding
        import govori.cli
        import govori.notes_cli

    out = buf.getvalue()
    assert "Govori ready" not in out
    assert "Hotkey monitor" not in out


if __name__ == "__main__":
    test_imports_have_no_side_effects()
    print("OK")
