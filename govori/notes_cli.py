"""Interactive notes picker and voice amendment UI.

This module deliberately keeps print() calls because curses/fzf fallback output
is TTY-only and relies on ANSI escapes.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import config as cfg
from .notes import _get_anthropic_client, _resolve_path, _split_frontmatter, _update_frontmatter_amended
from .state import PERMANENT_API_ERROR
from .transcribe import _encode_and_transcribe

def _read_index_entries(limit=30):
    if not cfg.NOTES_CFG:
        return []
    # Search a few recent months to collect up to `limit` entries.
    entries = []
    now = datetime.datetime.now()
    seen = set()
    for months_back in range(0, 12):
        d = now - datetime.timedelta(days=30 * months_back)
        path = _resolve_path(cfg.NOTES_CFG["index_file"], d)
        if not path.exists() or str(path) in seen:
            continue
        seen.add(str(path))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            # Skip malformed/empty entries
            if not entry.get("path") or not entry.get("summary"):
                continue
            entries.append(entry)
        if len(entries) >= limit * 3:
            break
    entries.sort(key=lambda e: e.get("created", ""), reverse=True)
    return entries[:limit]


def _curses_pick(entries):
    """Arrow-key picker using stdlib curses. Returns entry or None."""
    import curses

    def _draw(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        try:
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
        except Exception:
            pass

        idx = 0
        top = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            list_h = max(3, h - 4)
            # Keep selection in view
            if idx < top:
                top = idx
            elif idx >= top + list_h:
                top = idx - list_h + 1

            header = " govori notes — ↑/↓ select · Enter open · q quit "
            stdscr.addnstr(0, 0, header.ljust(w)[:w], w, curses.color_pair(3) | curses.A_BOLD)

            for row, i in enumerate(range(top, min(top + list_h, len(entries)))):
                e = entries[i]
                created = e.get("created", "")[:16].replace("T", " ")
                ctxs = ",".join(e.get("contexts") or [])
                summary = (e.get("summary") or "").replace("\n", " ")
                line = f"  {created}  [{ctxs}]  {summary}"
                line = line[: w - 1]
                attr = curses.color_pair(2) | curses.A_BOLD if i == idx else 0
                try:
                    stdscr.addnstr(row + 1, 0, line.ljust(w - 1), w - 1, attr)
                except curses.error:
                    pass

            footer = f" {idx + 1}/{len(entries)} "
            try:
                stdscr.addnstr(h - 1, 0, footer.ljust(w)[:w], w, curses.A_DIM)
            except curses.error:
                pass
            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(entries)
            elif ch in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(entries)
            elif ch in (curses.KEY_HOME, ord("g")):
                idx = 0
            elif ch in (curses.KEY_END, ord("G")):
                idx = len(entries) - 1
            elif ch == curses.KEY_PPAGE:
                idx = max(0, idx - list_h)
            elif ch == curses.KEY_NPAGE:
                idx = min(len(entries) - 1, idx + list_h)
            elif ch in (ord("\n"), curses.KEY_ENTER, 10, 13):
                return idx
            elif ch in (ord("q"), 27):
                return None

    try:
        selected = curses.wrapper(_draw)
    except Exception as e:
        print(f"(curses failed: {e})", flush=True)
        return "FALLBACK"
    if selected is None:
        return None
    return entries[selected]


def _fzf_pick(entries):
    import shutil, subprocess
    fzf = shutil.which("fzf")
    lines = []
    for i, e in enumerate(entries):
        created = e.get("created", "")[:16].replace("T", " ")
        ctxs = ",".join(e.get("contexts") or [])
        summary = (e.get("summary") or "").replace("\t", " ").replace("\n", " ")
        lines.append(f"{i}\t{created}  [{ctxs}]  {summary}")
    if fzf:
        preview = "awk -F'\\t' '{print $1}' <<< {} | xargs -I{} sh -c 'cat \"$0\"' " \
                  "$(printf '%s\\n' " + " ".join(f"'{e['path']}'" for e in entries) + " | sed -n \"$(({}+1))p\")"
        # Simpler: use a temp python preview via env
        env_paths = "\n".join(e["path"] for e in entries)
        preview_cmd = f"python3 -c 'import sys,os; paths=os.environ[\"GOVORI_PATHS\"].split(chr(10)); i=int(sys.argv[1]); p=paths[i]; print(open(p).read()) if os.path.exists(p) else print(\"(missing)\")' {{1}}"
        try:
            result = subprocess.run(
                [fzf, "--delimiter=\t", "--with-nth=2..",
                 "--preview", preview_cmd, "--preview-window=right:60%:wrap",
                 "--height=80%", "--reverse"],
                input="\n".join(lines), text=True, capture_output=True,
                env={**os.environ, "GOVORI_PATHS": env_paths},
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            idx = int(result.stdout.split("\t", 1)[0])
            return entries[idx]
        except Exception as e:
            print(f"(fzf failed: {e})", flush=True)
    # Fallback: numbered menu
    print()
    for i, e in enumerate(entries):
        created = e.get("created", "")[:16].replace("T", " ")
        ctxs = ",".join(e.get("contexts") or [])
        summary = (e.get("summary") or "")[:80]
        print(f"  [{i:2d}] {created}  \033[36m[{ctxs}]\033[0m  {summary}")
    print()
    try:
        raw = input("Select # (or q to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw in ("", "q", "Q"):
        return None
    try:
        return entries[int(raw)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None


def _record_until_enter():
    """Record audio from mic until user presses Enter. Returns numpy array or None."""
    chunks = []

    def cb(indata, frames, time_info, status):
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(samplerate=cfg.SAMPLE_RATE, channels=1, dtype="float32", callback=cb)
        stream.start()
    except sd.PortAudioError as e:
        err_str = str(e).lower()
        if "permission" in err_str or "denied" in err_str:
            print("! Microphone access denied — grant in System Settings → Privacy & Security → Microphone", flush=True)
        else:
            print(f"! No microphone available ({e})", flush=True)
        return None

    print("\n🎙  Recording... press [Enter] to stop.", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    stream.stop()
    stream.close()
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0).flatten()


def _amend_via_haiku(original_text, instruction):
    anthropic_client = _get_anthropic_client()
    if anthropic_client is None:
        return None
    system = """You edit a user's voice note based on their spoken instruction.
Rules:
- If instruction says "добавь/add/append" — append the new content as a new paragraph to the existing note.
- If instruction says "перепиши/rewrite/replace" — produce a rewritten version preserving the original intent.
- If instruction says "убери/удали/remove/delete X" — remove that part from the note.
- If instruction is itself additional content without a verb, treat as append.
- Preserve the original language of the note.
- Return ONLY the full new note body (no frontmatter, no commentary, no markdown fences)."""
    user = f"ORIGINAL NOTE:\n{original_text}\n\nINSTRUCTION (voice):\n{instruction}"
    try:
        resp = anthropic_client.messages.create(
            model=cfg.NOTES_CFG["classifier_model"],
            max_tokens=2000,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"Amend error: {e}", flush=True)
        return None


def _split_frontmatter(md):
    """Returns (frontmatter_lines, body)."""
    lines = md.splitlines()
    if not lines or lines[0].strip() != "---":
        return [], md
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return [], md
    return lines[: end + 1], "\n".join(lines[end + 1 :]).lstrip("\n")


def _update_frontmatter_amended(fm_lines, timestamp):
    """Add or extend `amended:` list in frontmatter."""
    for i, line in enumerate(fm_lines):
        if line.startswith("amended:"):
            try:
                arr = json.loads(line.split(":", 1)[1].strip())
                if not isinstance(arr, list):
                    arr = []
            except Exception:
                arr = []
            arr.append(timestamp)
            fm_lines[i] = f"amended: {json.dumps(arr, ensure_ascii=False)}"
            return fm_lines
    # Insert before closing ---
    fm_lines.insert(-1, f"amended: {json.dumps([timestamp], ensure_ascii=False)}")
    return fm_lines


def cli_notes(args):
    """Interactive picker + voice amendment."""
    import difflib
    if not cfg.NOTES_CFG:
        print("notes plugin not installed. Run: govori plugin init notes")
        return
    entries = _read_index_entries(limit=30)
    if not entries:
        print("No notes found.")
        return

    # Priority: fzf (if installed) → curses (tty) → numbered menu
    import shutil
    picked = None
    if shutil.which("fzf"):
        picked = _fzf_pick(entries)
    elif sys.stdin.isatty() and sys.stdout.isatty():
        picked = _curses_pick(entries)
        if picked == "FALLBACK":
            picked = _fzf_pick(entries)
    else:
        picked = _fzf_pick(entries)
    if not picked:
        return

    path = Path(picked["path"])
    if not path.exists():
        print(f"File missing: {path}")
        return

    original = path.read_text(encoding="utf-8")
    fm_lines, body = _split_frontmatter(original)

    print("\n" + "─" * 60)
    print(f"\033[36m{path.name}\033[0m")
    print("─" * 60)
    print(body.strip() or "(empty)")
    print("─" * 60)

    try:
        choice = input("\n[r] record voice edit  [o] open in editor  [q] quit: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if choice in ("", "q"):
        return
    if choice == "o":
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(path)])
        return
    if choice != "r":
        return

    audio = _record_until_enter()
    if audio is None or len(audio) / cfg.SAMPLE_RATE < 0.3:
        print("(too short)")
        return

    print("■ Transcribing…", flush=True)
    instruction = _encode_and_transcribe(audio)
    if instruction is PERMANENT_API_ERROR or not instruction:
        print("(transcription failed)")
        return
    print(f"→ {instruction}")

    print("✎ Amending via Claude…", flush=True)
    new_body = _amend_via_haiku(body.strip(), instruction)
    if not new_body:
        print("(amend failed)")
        return

    # Diff
    print("\n" + "─" * 60)
    print("\033[33mDIFF:\033[0m")
    diff = difflib.unified_diff(
        body.strip().splitlines(), new_body.splitlines(),
        lineterm="", fromfile="before", tofile="after",
    )
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"\033[32m{line}\033[0m")
        elif line.startswith("-") and not line.startswith("---"):
            print(f"\033[31m{line}\033[0m")
        else:
            print(line)
    print("─" * 60)

    try:
        confirm = input("\nApply? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if confirm not in ("y", "yes", "д", "да"):
        print("Cancelled.")
        return

    now = datetime.datetime.now().astimezone()
    ts = now.isoformat(timespec="seconds")
    if fm_lines:
        fm_lines = _update_frontmatter_amended(fm_lines, ts)
        path.write_text("\n".join(fm_lines) + "\n\n" + new_body.strip() + "\n", encoding="utf-8")
    else:
        path.write_text(new_body.strip() + "\n", encoding="utf-8")
    print(f"✓ Saved: {path}")

