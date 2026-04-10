#!/usr/bin/env python3
"""
Wisp — voice dictation for macOS.
Hold fn to record, release to transcribe and paste.

Modes:
  fn          — dictate → paste into cursor
  Option+fn   — dictate → predict (autocomplete menu)
  Shift+fn    — dictate → classify + save as note (requires notes plugin)

Plugins live in ~/.config/wisp/plugins/<name>/.
"""

import sys
import os
import io
import json
import time
import threading
import datetime
import re
from pathlib import Path

import numpy as np
import av
import sounddevice as sd
from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import yaml
except ImportError:
    yaml = None

import AppKit
import Quartz
import signal
import CoreFoundation

# ── Paths ────────────────────────────────────────────────────────────────────
CONFIG_DIR  = Path.home() / ".config" / "wisp"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
PLUGINS_DIR = CONFIG_DIR / "plugins"

# ── Config loading ───────────────────────────────────────────────────────────
def _load_yaml(path):
    """Load a YAML file. Falls back to json-style if PyYAML not installed."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text) or {}
    # Minimal fallback — won't handle all YAML but covers simple cases
    try:
        return json.loads(text)
    except Exception:
        return {}


def _load_yaml_list(path):
    """Load a YAML file expected to contain a list."""
    data = _load_yaml(path)
    if isinstance(data, list):
        return data
    return []


def load_config():
    """Load base config from ~/.config/wisp/config.yaml."""
    defaults = {
        "language": "ru",
        "model": "whisper-1",
        "sample_rate": 16000,
        "whisper_prompt": "",
    }
    cfg = _load_yaml(CONFIG_FILE)
    defaults.update(cfg)
    return defaults


def load_plugins():
    """Discover and load all plugins from ~/.config/wisp/plugins/."""
    plugins = {}
    if not PLUGINS_DIR.exists():
        return plugins
    for d in sorted(PLUGINS_DIR.iterdir()):
        if not d.is_dir():
            continue
        plugin_yaml = d / "plugin.yaml"
        if not plugin_yaml.exists():
            continue
        meta = _load_yaml(plugin_yaml)
        meta["_dir"] = d
        meta["_name"] = d.name

        # Load contexts if present
        contexts_file = d / "contexts.yaml"
        if contexts_file.exists():
            meta["contexts"] = _load_yaml_list(contexts_file)

        # Load stuck tasks if present
        stuck_file = d / "stuck.yaml"
        if stuck_file.exists():
            meta["stuck"] = _load_yaml_list(stuck_file)

        plugins[d.name] = meta
    return plugins


def build_whisper_prompt(config, plugins):
    """Merge base whisper_prompt with plugin-level prompts."""
    parts = []
    base = config.get("whisper_prompt", "").strip()
    if base:
        parts.append(base)
    for name, plugin in plugins.items():
        p = plugin.get("whisper_prompt", "").strip()
        if p:
            parts.append(p)
    return " ".join(parts)


def build_notes_config(plugins):
    """Extract notes plugin config (contexts, stuck, paths, classifier)."""
    notes = plugins.get("notes")
    if not notes:
        return None

    contexts = notes.get("contexts", [])
    stuck = notes.get("stuck", [])

    valid_contexts = {c["key"] for c in contexts}
    valid_stuck = {s["key"] for s in stuck}

    # Build description strings for the classifier prompt
    contexts_desc = "\n".join(
        f"- {c['key']}: {c['description']}" for c in contexts
    )
    stuck_desc = "\n".join(
        f"- {s['key']}: {s['description']}" for s in stuck
    )

    # Resolve output paths
    output_dir = notes.get("output_dir", "~/wisp-notes/{year}/{month}")
    index_file = notes.get("index_file", "~/wisp-notes/index/recent.jsonl")

    return {
        "classifier_model": notes.get("classifier_model", "claude-haiku-4-5-20251001"),
        "output_dir": output_dir,
        "index_file": index_file,
        "valid_contexts": valid_contexts,
        "valid_stuck": valid_stuck,
        "contexts_desc": contexts_desc,
        "stuck_desc": stuck_desc,
    }


# ── Load everything ──────────────────────────────────────────────────────────
CONFIG  = load_config()
PLUGINS = load_plugins()

SAMPLE_RATE    = CONFIG["sample_rate"]
LANGUAGE       = CONFIG["language"]
MODEL          = "gpt-4o-transcribe" if "--gpt" in sys.argv else CONFIG["model"]
WHISPER_PROMPT = build_whisper_prompt(CONFIG, PLUGINS)
NOTES_CFG      = build_notes_config(PLUGINS)

VALID_TYPES   = {"idea", "commitment", "observation", "todo", "decision", "question", "other"}
VALID_URGENCY = {"low", "medium", "high"}

# ── CLI subcommands ──────────────────────────────────────────────────────────
def cli_plugin(args):
    """Handle `wisp plugin <subcommand>` CLI."""
    if not args:
        print("Usage: wisp plugin <list|add|init|remove>")
        sys.exit(1)

    sub = args[0]

    if sub == "list":
        if not PLUGINS:
            print("No plugins installed.")
        for name, meta in PLUGINS.items():
            desc = meta.get("description", "")
            trigger = meta.get("trigger", "n/a")
            print(f"  {name:20s} trigger={trigger:12s}  {desc}")

    elif sub == "init":
        if len(args) < 2:
            print("Usage: wisp plugin init <name>")
            sys.exit(1)
        name = args[1]
        dest = PLUGINS_DIR / name
        if dest.exists():
            print(f"Plugin '{name}' already exists at {dest}")
            sys.exit(1)
        dest.mkdir(parents=True)
        (dest / "plugin.yaml").write_text(
            f"name: {name}\n"
            f"description: My custom plugin\n"
            f"trigger: shift+fn\n"
            f"classifier_model: claude-haiku-4-5-20251001\n"
            f"\n"
            f"output_dir: ~/wisp-notes/{{year}}/{{month}}\n"
            f"index_file: ~/wisp-notes/index/recent.jsonl\n"
            f"\n"
            f"whisper_prompt: \"\"\n",
            encoding="utf-8",
        )
        (dest / "contexts.yaml").write_text(
            "- key: work\n"
            "  description: My day job\n"
            "\n"
            "- key: personal\n"
            "  description: Personal life, family, health\n",
            encoding="utf-8",
        )
        (dest / "stuck.yaml").write_text(
            "# Optional: ongoing tasks to link notes to\n"
            "# - key: my_task\n"
            "#   description: What this task is about\n",
            encoding="utf-8",
        )
        print(f"Plugin scaffold created at {dest}/")
        print(f"Edit contexts.yaml to define your contexts, then restart wisp.")

    elif sub == "remove":
        if len(args) < 2:
            print("Usage: wisp plugin remove <name>")
            sys.exit(1)
        name = args[1]
        dest = PLUGINS_DIR / name
        if not dest.exists():
            print(f"Plugin '{name}' not found.")
            sys.exit(1)
        import shutil
        shutil.rmtree(dest)
        print(f"Removed plugin '{name}'.")

    else:
        print(f"Unknown subcommand: {sub}")
        print("Usage: wisp plugin <list|init|remove>")
        sys.exit(1)


def cli_main():
    """Route CLI subcommands before starting the daemon."""
    args = sys.argv[1:]
    # Filter out flags like --gpt
    positional = [a for a in args if not a.startswith("--")]
    if positional and positional[0] == "plugin":
        cli_plugin(positional[1:])
        sys.exit(0)


# Run CLI routing before anything else
cli_main()

# ── Whisper hallucination filter ─────────────────────────────────────────────
WHISPER_HALLUCINATIONS = {
    "продолжение следует", "спасибо за просмотр", "спасибо за внимание",
    "субтитры создал", "субтитры сделал", "субтитры подготовил",
    "подписывайтесь на канал", "подпишитесь на канал",
    "до свидания", "до новых встреч", "пока",
    "thanks for watching", "thank you for watching",
    "to be continued", "subscribe", "like and subscribe",
    "you", "the end", "bye",
    ".", "..", "...", "",
    "ご視聴ありがとうございました",
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Anthropic client — lazy, only if note mode is used
_anthropic_client = None

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    if Anthropic is None:
        print("anthropic package not installed — run: pip install anthropic", flush=True)
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY not set — check ~/.config/wisp/env", flush=True)
        return None
    _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client

# ── State ─────────────────────────────────────────────────────────────────────
_state_lock  = threading.Lock()
recording    = False
transcribing = False
audio_chunks = []
audio_stream = None
auto_send    = False
cancelled    = False
predict_mode = False
note_mode    = False

print("Wisp ready.", flush=True)
if NOTES_CFG:
    n_ctx = len(NOTES_CFG["valid_contexts"])
    print(f"  notes plugin: {n_ctx} contexts loaded", flush=True)
else:
    print("  notes plugin: not installed (shift+fn disabled)", flush=True)

# ── HUD ───────────────────────────────────────────────────────────────────────
hud_window = None
hud_label  = None

_HUD_S = 32

def setup_hud():
    global hud_window, hud_label

    screen = AppKit.NSScreen.mainScreen().frame()
    x = (screen.size.width - _HUD_S) / 2
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    win = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(x, 60, _HUD_S, _HUD_S), style,
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


def set_hud(visible, mode="recording"):
    def _update():
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
        if visible:
            screen = AppKit.NSScreen.mainScreen().frame()
            hud_window.setFrameOrigin_(
                AppKit.NSMakePoint((screen.size.width - _HUD_S) / 2, 60)
            )
            hud_window.orderFrontRegardless()
        else:
            hud_window.orderOut_(None)
    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_update)

# ── Audio ─────────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_chunks.append(indata.copy())


def start_recording():
    global recording, audio_chunks, audio_stream, auto_send, cancelled
    with _state_lock:
        if recording:
            return
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception:
                pass
            audio_stream = None
        recording    = True
        auto_send    = False
        cancelled    = False
        audio_chunks = []
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback,
        )
        audio_stream.start()
    if note_mode:
        hud_mode = "note"
        icon = "✎"
    elif predict_mode:
        hud_mode = "predict"
        icon = "✦"
    else:
        hud_mode = "recording"
        icon = "●"
    set_hud(True, hud_mode)
    print(f"{icon} Recording…", flush=True)


def _encode_and_transcribe(audio):
    """Encode mono float32 audio → OGG/Opus → Whisper. Returns text or None."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    buf = io.BytesIO()
    buf.name = "audio.ogg"
    audio_int16 = (audio * 32767).astype(np.int16)
    container = av.open(buf, mode="w", format="ogg")
    stream = container.add_stream("libopus", rate=SAMPLE_RATE, layout="mono")
    frame = av.AudioFrame.from_ndarray(audio_int16.reshape(1, -1), format="s16", layout="mono")
    frame.rate = SAMPLE_RATE
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    buf.seek(0)

    try:
        result = client.audio.transcriptions.create(
            model=MODEL,
            file=buf,
            language=LANGUAGE,
            temperature=0,
            prompt=WHISPER_PROMPT,
        )
        return result.text.strip()
    except Exception as e:
        print(f"API error: {e}", flush=True)
        return None


def _is_hallucination(text):
    text_check = text.lower().strip().rstrip(".!?,;:…").strip()
    return (
        text_check in WHISPER_HALLUCINATIONS
        or text.lower().strip() in WHISPER_HALLUCINATIONS
    )


def _note_pipeline_background(audio, duration_sec):
    """Full note pipeline: transcribe → filter → classify → save. No HUD updates."""
    text = _encode_and_transcribe(audio)
    if text is None:
        return
    if _is_hallucination(text):
        print(f"(hallucination filtered: {text})", flush=True)
        return
    if not text:
        print("(empty)", flush=True)
        return
    print(f"→ {text}", flush=True)
    save_as_note(text, duration_sec, silent=True)


def stop_and_transcribe():
    global recording, audio_stream, transcribing
    with _state_lock:
        recording = False
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None

        if not audio_chunks or cancelled:
            set_hud(False)
            return

    total_samples = sum(len(c) for c in audio_chunks)
    if total_samples / SAMPLE_RATE < 0.5:
        set_hud(False)
        print("(too short)", flush=True)
        return

    audio = np.concatenate(audio_chunks, axis=0).flatten()

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.001:
        set_hud(False)
        print(f"(silence, rms={rms:.4f})", flush=True)
        return

    # ── NOTE MODE: fire-and-forget ────────────────────────────────────────────
    if note_mode:
        if not NOTES_CFG:
            print("notes plugin not installed — run: wisp plugin init notes", flush=True)
            set_hud(False)
            return
        duration = total_samples / SAMPLE_RATE
        audio_copy = audio.copy()
        set_hud(True, "note_saved")
        print("✓ Note captured (background pipeline running)", flush=True)

        def _hide_check():
            time.sleep(1.2)
            set_hud(False)
        threading.Thread(target=_hide_check, daemon=True).start()

        threading.Thread(
            target=lambda a=audio_copy, d=duration: _note_pipeline_background(a, d),
            daemon=True,
        ).start()
        return

    # ── NORMAL / PREDICT MODE: blocking transcription, paste into cursor ──────
    transcribing = True
    set_hud(True, "transcribing")
    print("■ Transcribing…", flush=True)

    text = _encode_and_transcribe(audio)
    transcribing = False
    set_hud(False)

    if text is None or cancelled:
        return

    if _is_hallucination(text):
        print(f"(hallucination filtered: {text})", flush=True)
        return

    if text:
        print(f"→ {text}", flush=True)
        paste_text(text + " ")
        if predict_mode:
            AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(
                lambda t=text: show_predict_menu(t)
            )
        elif auto_send:
            time.sleep(0.3)
            _press_enter()
    else:
        print("(empty)", flush=True)


def cancel_recording():
    global recording, transcribing, audio_stream, audio_chunks, cancelled, predict_mode, note_mode
    with _state_lock:
        cancelled    = True
        recording    = False
        transcribing = False
        predict_mode = False
        note_mode    = False
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None
        audio_chunks = []
    set_hud(False)
    print("(cancelled)", flush=True)

# ── Paste / Enter ─────────────────────────────────────────────────────────────
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

# ── Note mode (save transcription as classified markdown) ────────────────────
def _sanitize_slug(s, maxlen=40):
    s = (s or "note").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s or "note")[:maxlen]


def _validate_meta(data):
    """Coerce classifier output into the schema, dropping invalid values."""
    if not NOTES_CFG:
        return data

    contexts = data.get("contexts") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    contexts = [c for c in contexts if c in NOTES_CFG["valid_contexts"]]
    if not contexts:
        # Fall back to first defined context
        contexts = [next(iter(NOTES_CFG["valid_contexts"]))] if NOTES_CFG["valid_contexts"] else ["default"]

    type_ = data.get("type", "other")
    if type_ not in VALID_TYPES:
        type_ = "other"

    urgency = data.get("urgency", "low")
    if urgency not in VALID_URGENCY:
        urgency = "low"

    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t).strip().lower() for t in tags if t][:4]

    related = data.get("related_stuck") or []
    if isinstance(related, str):
        related = [related]
    related = [r for r in related if r in NOTES_CFG["valid_stuck"]]

    title = str(data.get("title") or "note").strip()

    return {
        "title": title,
        "contexts": contexts,
        "type": type_,
        "urgency": urgency,
        "tags": tags,
        "related_stuck": related,
    }


def classify_note(text):
    """Classify transcribed note via Claude. Returns validated meta dict."""
    if not NOTES_CFG:
        return {"title": "note", "contexts": ["default"], "type": "other",
                "urgency": "low", "tags": [], "related_stuck": [], "review": True}

    anthropic_client = _get_anthropic_client()
    if anthropic_client is None:
        return {"title": "note", "contexts": ["default"], "type": "other",
                "urgency": "low", "tags": [], "related_stuck": [], "review": True}

    stuck_block = ""
    if NOTES_CFG["stuck_desc"]:
        stuck_block = f"""
User's ongoing stuck tasks (link note to one if relevant):
{NOTES_CFG['stuck_desc']}
"""

    system = f"""You classify voice notes for a user with multiple contexts.

User's contexts (use these exact keys):
{NOTES_CFG['contexts_desc']}
{stuck_block}
Given a transcribed note, return STRICT JSON ONLY with these fields:
- title: 2-5 word slug in latin kebab-case (e.g. "work-deploy-issue")
- contexts: array — usually ONE element. Multiple only if the note explicitly mixes projects.
- type: one of [idea, commitment, observation, todo, decision, question, other]
- urgency: one of [low, medium, high]
- tags: 1-4 short lowercase tags (free-form)
- related_stuck: array with zero or more stuck task keys (only if relevant)

Return ONLY valid JSON, no markdown, no commentary."""

    try:
        resp = anthropic_client.messages.create(
            model=NOTES_CFG["classifier_model"],
            max_tokens=400,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        return _validate_meta(data)
    except Exception as e:
        print(f"Classify error: {e}", flush=True)
        return {"title": "note", "contexts": ["default"], "type": "other",
                "urgency": "low", "tags": [], "related_stuck": [], "review": True}


def _resolve_path(template, now):
    """Resolve path template with {year}, {month}, ~ expansion."""
    s = template.replace("{year}", now.strftime("%Y")).replace("{month}", now.strftime("%m"))
    return Path(os.path.expanduser(s))


def save_as_note(text, duration_sec, silent=False):
    """Classify + write markdown file + append to recent index."""
    if not NOTES_CFG:
        print("notes plugin not configured", flush=True)
        return

    try:
        meta = classify_note(text)

        now = datetime.datetime.now().astimezone()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M")
        slug = _sanitize_slug(meta["title"])
        note_id = f"{date_str}_{time_str}_{slug}"

        target_dir = _resolve_path(NOTES_CFG["output_dir"], now)
        target_dir.mkdir(parents=True, exist_ok=True)
        note_path = target_dir / f"{note_id}.md"

        fm_lines = [
            "---",
            f"id: {note_id}",
            f"created: {now.isoformat(timespec='seconds')}",
            "source: voice",
            f"duration_sec: {int(round(duration_sec))}",
            f"contexts: {json.dumps(meta['contexts'], ensure_ascii=False)}",
            f"type: {meta['type']}",
            f"urgency: {meta['urgency']}",
            f"tags: {json.dumps(meta['tags'], ensure_ascii=False)}",
            f"related_stuck: {json.dumps(meta['related_stuck'], ensure_ascii=False)}",
        ]
        if meta.get("review"):
            fm_lines.append("review: true")
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(text.strip())
        fm_lines.append("")

        note_path.write_text("\n".join(fm_lines), encoding="utf-8")

        # Append to index
        index_path = _resolve_path(NOTES_CFG["index_file"], now)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_entry = {
            "id": note_id,
            "created": now.isoformat(timespec="seconds"),
            "path": str(note_path),
            "contexts": meta["contexts"],
            "type": meta["type"],
            "urgency": meta["urgency"],
            "related_stuck": meta["related_stuck"],
            "summary": text.strip()[:200],
        }
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        print(
            f"✎ saved: {note_path.name} "
            f"[{', '.join(meta['contexts'])}] {meta['type']}/{meta['urgency']}",
            flush=True,
        )
        if not silent:
            set_hud(True, "note_saved")
    except Exception as e:
        print(f"save_as_note error: {e}", flush=True)
        if not silent:
            set_hud(True, "note_error")

    if not silent:
        def _hide():
            time.sleep(1.2)
            set_hud(False)
        threading.Thread(target=_hide, daemon=True).start()


# ── Predict mode (T9) ────────────────────────────────────────────────────────
_predict_controller = None


def generate_continuations(text):
    """GPT-4o-mini generates 3 text continuations."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a text autocomplete assistant. "
                        "Given the beginning of a text, suggest 3 natural continuations. "
                        "Each continuation should be 5-20 words, completing the thought. "
                        "Keep the same language as input. "
                        "Return JSON: {\"continuations\": [\"...\", \"...\", \"...\"]}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0.7,
        )
        data = json.loads(resp.choices[0].message.content)
        items = data.get("continuations", [])
        if isinstance(items, list) and len(items) >= 1:
            return [str(v) for v in items[:3]]
    except Exception as e:
        print(f"Predict error: {e}", flush=True)
    return []


class PredictController(AppKit.NSObject):
    _continuations = []

    def pickContinuation_(self, sender):
        idx = sender.tag()
        if 0 <= idx < len(self._continuations):
            text = self._continuations[idx]
            print(f"✦ predict: {text}", flush=True)
            threading.Thread(target=lambda t=text: paste_text(t), daemon=True).start()


def setup_predict():
    global _predict_controller
    _predict_controller = PredictController.alloc().init()


def show_predict_menu(original_text):
    """Generate continuations and show NSMenu."""
    set_hud(True, "predict")
    continuations = generate_continuations(original_text)
    set_hud(False)

    if not continuations:
        print("(no predictions)", flush=True)
        return

    _predict_controller._continuations = continuations

    menu = AppKit.NSMenu.alloc().init()
    menu.setAutoenablesItems_(False)
    menu.setMinimumWidth_(300)
    menu.setAppearance_(
        AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark)
    )

    for i, cont in enumerate(continuations):
        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            cont, "pickContinuation:", str(i + 1)
        )
        item.setTarget_(_predict_controller)
        item.setEnabled_(True)
        item.setTag_(i)
        item.setKeyEquivalentModifierMask_(0)
        menu.addItem_(item)

    loc = AppKit.NSEvent.mouseLocation()
    menu.popUpMenuPositioningItem_atLocation_inView_(None, loc, None)


# ── Hotkey (fn) ───────────────────────────────────────────────────────────────
FN_KEYCODE   = 63
FN_FLAG      = 0x800000
prev_fn_down = False
_fn_press_time = 0

_shift_held  = False
_option_held = False


def cg_event_callback(proxy, event_type, event, refcon):
    global prev_fn_down, _fn_press_time, _shift_held, _option_held, note_mode
    keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
    flags_now = Quartz.CGEventGetFlags(event)

    prev_shift_held = _shift_held
    _shift_held  = bool(flags_now & Quartz.kCGEventFlagMaskShift)
    _option_held = bool(flags_now & Quartz.kCGEventFlagMaskAlternate)

    # Shift TAP during recording → toggle note mode
    if recording and _shift_held and not prev_shift_held:
        if NOTES_CFG:
            note_mode = not note_mode
            set_hud(True, "note" if note_mode else "recording")
            print(f"[toggle] note_mode={'on' if note_mode else 'off'}", flush=True)
        else:
            print("notes plugin not installed — shift+fn disabled", flush=True)

    # Esc → cancel
    if event_type == Quartz.kCGEventKeyDown and keycode == 53 and (recording or transcribing):
        threading.Thread(target=cancel_recording, daemon=True).start()
        return event

    # Enter during recording → toggle auto-send + undo the inserted Enter
    if keycode in (36, 76) and recording and event_type == Quartz.kCGEventKeyDown:
        global auto_send
        auto_send = not auto_send
        print(f"auto_send={'on' if auto_send else 'off'}", flush=True)
        def _undo_enter():
            time.sleep(0.05)
            src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            ev = Quartz.CGEventCreateKeyboardEvent(src, 0x06, True)   # Z
            Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            ev = Quartz.CGEventCreateKeyboardEvent(src, 0x06, False)
            Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        threading.Thread(target=_undo_enter, daemon=True).start()
        return event

    # fn key
    if keycode != FN_KEYCODE:
        return event

    is_down = bool(flags_now & FN_FLAG)

    if is_down and not prev_fn_down:
        _fn_press_time = time.time()
        def delayed_start():
            time.sleep(0.25)
            if not prev_fn_down:
                return
            global predict_mode, note_mode
            if _shift_held and NOTES_CFG:
                note_mode    = True
                predict_mode = False
            elif _option_held:
                predict_mode = True
                note_mode    = False
            else:
                predict_mode = False
                note_mode    = False
            print(
                f"[mode] shift={_shift_held} option={_option_held} "
                f"→ note={note_mode} predict={predict_mode}",
                flush=True,
            )
            start_recording()
        threading.Thread(target=delayed_start, daemon=True).start()
    elif not is_down and prev_fn_down:
        elapsed = time.time() - _fn_press_time
        if elapsed >= 0.25 and recording:
            threading.Thread(target=stop_and_transcribe, daemon=True).start()

    prev_fn_down = is_down
    return event


def install_monitor():
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
        cg_event_callback,
        None,
    )
    if tap is None:
        print("ERROR: CGEventTap failed. Check Accessibility permission.", flush=True)
        sys.exit(1)

    src = CoreFoundation.CFMachPortCreateRunLoopSource(None, tap, 0)
    CoreFoundation.CFRunLoopAddSource(
        CoreFoundation.CFRunLoopGetMain(), src, CoreFoundation.kCFRunLoopCommonModes,
    )
    Quartz.CGEventTapEnable(tap, True)
    print("Hotkey monitor installed.", flush=True)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Wisp started. Model: {MODEL}. Hold fn to record.", flush=True)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    setup_hud()
    setup_predict()
    install_monitor()

    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    run_loop = AppKit.NSRunLoop.mainRunLoop()
    while True:
        run_loop.runMode_beforeDate_(
            AppKit.NSDefaultRunLoopMode,
            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.5),
        )
