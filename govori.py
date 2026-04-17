#!/usr/bin/env python3
"""
Govori — voice dictation for macOS.
Hold fn to record, release to transcribe and paste.

Modes:
  fn          — dictate → paste into cursor
  Option+fn   — dictate → predict (autocomplete menu)
  Shift+fn    — dictate → classify + save as note (requires notes plugin)

Plugins live in ~/.config/govori/plugins/<name>/.
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
CONFIG_DIR  = Path.home() / ".config" / "govori"
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
    """Load base config from ~/.config/govori/config.yaml."""
    defaults = {
        "language": "ru",
        "model": "whisper-1",
        "sample_rate": 16000,
        "whisper_prompt": "",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    }
    cfg = _load_yaml(CONFIG_FILE)
    defaults.update(cfg)
    return defaults


def load_plugins():
    """Discover and load all plugins from ~/.config/govori/plugins/."""
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
    output_dir = notes.get("output_dir", "~/govori-notes/{year}/{month}")
    index_file = notes.get("index_file", "~/govori-notes/index/recent.jsonl")

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

# ── Onboarding / Setup ───────────────────────────────────────────────────────
SETUP_STRINGS = {
    "en": {
        "welcome": """
\033[2m         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m
\033[36m              ✦ voice dictation for macOS\033[0m
\033[2m         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m
""",
        "step_keys": """
\033[33m  ── Step 1/3 ─ API Keys ──────────────────────────────\033[0m

  Govori needs an OpenAI API key for speech-to-text.
  \033[2mOptionally, add an Anthropic key for smart note classification.\033[0m

  \033[2mYour keys are stored locally in ~/.config/govori/env\033[0m
""",
        "ask_openai": "  \033[1mOpenAI API key\033[0m (sk-...): ",
        "ask_anthropic": "  \033[1mAnthropic API key\033[0m (sk-ant-..., Enter to skip): ",
        "keys_saved": "\n  \033[32m✓ Keys saved to ~/.config/govori/env\033[0m\n",
        "step_access": """
\033[33m  ── Step 2/3 ─ Accessibility Permission ──────────────\033[0m

  Govori needs Accessibility access to listen for the \033[1mfn\033[0m key.

  \033[36mSystem Settings → Privacy & Security → Accessibility\033[0m
  \033[36m→ Add your terminal app (Terminal / iTerm / Ghostty)\033[0m

""",
        "ask_access_done": "  \033[2mPress Enter when done...\033[0m",
        "step_hotkeys": """
\033[33m  ── Step 3/3 ─ How to Use ────────────────────────────\033[0m

  \033[1mHold fn\033[0m         →  dictate → paste at cursor
  \033[1mOption + fn\033[0m     →  dictate → autocomplete menu \033[2m(3 suggestions)\033[0m
  \033[1mShift + fn\033[0m      →  dictate → save as classified note

  \033[2mDuring recording:\033[0m
    \033[1mEnter\033[0m           toggle auto-send
    \033[1mShift\033[0m           toggle note mode
    \033[1mEsc\033[0m             cancel

  \033[2mHUD indicators:\033[0m
    \033[31m●\033[0m  recording      \033[33m◎\033[0m  transcribing
    \033[35m✦\033[0m  predicting     \033[32m✎\033[0m  note mode
    \033[32m✓\033[0m  note saved     \033[31m✗\033[0m  error

""",
        "step_plugin": """
\033[33m  ── Notes Plugin ────────────────────────────────────\033[0m

  The notes plugin classifies voice memos into contexts you define.
""",
        "ask_plugin": "  Set up notes plugin now? [\033[1mY\033[0m/n]: ",
        "plugin_created": "\n  \033[32m✓ Notes plugin created.\033[0m Edit your contexts:\n    \033[36m~/.config/govori/plugins/notes/contexts.yaml\033[0m\n",
        "plugin_skipped": "  \033[2mSkipped. Run `govori plugin init notes` later.\033[0m\n",
        "done": """
\033[2m╭──────────────────────────────────────────────────────╮\033[0m

  \033[32m✓ Setup complete!\033[0m

  Run \033[1mgovori\033[0m to start dictating.
  Run \033[1mgovori setup\033[0m to reconfigure.

\033[2m╰──────────────────────────────────────────────────────╯\033[0m
""",
        "lang_prompt": "  Language / Язык [en/ru]: ",
    },
    "ru": {
        "welcome": """
\033[2m         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m
\033[36m              ✦ голосовой ввод для macOS\033[0m
\033[2m         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m
""",
        "step_keys": """
\033[33m  ── Шаг 1/3 ─ API-ключи ─────────────────────────────\033[0m

  Govori использует OpenAI API для распознавания речи.
  \033[2mОпционально: ключ Anthropic для умной классификации заметок.\033[0m

  \033[2mКлючи хранятся локально в ~/.config/govori/env\033[0m
""",
        "ask_openai": "  \033[1mOpenAI API ключ\033[0m (sk-...): ",
        "ask_anthropic": "  \033[1mAnthropic API ключ\033[0m (sk-ant-..., Enter чтобы пропустить): ",
        "keys_saved": "\n  \033[32m✓ Ключи сохранены в ~/.config/govori/env\033[0m\n",
        "step_access": """
\033[33m  ── Шаг 2/3 ─ Разрешение Accessibility ──────────────\033[0m

  Govori нужен доступ к Accessibility чтобы слушать клавишу \033[1mfn\033[0m.

  \033[36mСистемные настройки → Конфиденциальность → Универсальный доступ\033[0m
  \033[36m→ Добавь свой терминал (Terminal / iTerm / Ghostty)\033[0m

""",
        "ask_access_done": "  \033[2mНажми Enter когда готово...\033[0m",
        "step_hotkeys": """
\033[33m  ── Шаг 3/3 ─ Как пользоваться ───────────────────────\033[0m

  \033[1mЗажми fn\033[0m        →  диктовка → вставка в курсор
  \033[1mOption + fn\033[0m     →  диктовка → меню автодополнения \033[2m(3 варианта)\033[0m
  \033[1mShift + fn\033[0m      →  диктовка → классификация + сохранение заметки

  \033[2mВо время записи:\033[0m
    \033[1mEnter\033[0m           авто-отправка
    \033[1mShift\033[0m           переключить режим заметки
    \033[1mEsc\033[0m             отмена

  \033[2mИндикаторы HUD:\033[0m
    \033[31m●\033[0m  запись         \033[33m◎\033[0m  транскрипция
    \033[35m✦\033[0m  предсказание   \033[32m✎\033[0m  режим заметки
    \033[32m✓\033[0m  заметка сохр.  \033[31m✗\033[0m  ошибка

""",
        "step_plugin": """
\033[33m  ── Плагин заметок ──────────────────────────────────\033[0m

  Плагин заметок классифицирует голосовые мемо по контекстам,
  которые ты определяешь сам.
""",
        "ask_plugin": "  Настроить плагин заметок сейчас? [\033[1mY\033[0m/n]: ",
        "plugin_created": "\n  \033[32m✓ Плагин заметок создан.\033[0m Отредактируй контексты:\n    \033[36m~/.config/govori/plugins/notes/contexts.yaml\033[0m\n",
        "plugin_skipped": "  \033[2mПропущено. Запусти `govori plugin init notes` позже.\033[0m\n",
        "done": """
\033[2m╭──────────────────────────────────────────────────────╮\033[0m

  \033[32m✓ Настройка завершена!\033[0m

  Запусти \033[1mgovori\033[0m чтобы начать диктовку.
  Запусти \033[1mgovori setup\033[0m для перенастройки.

\033[2m╰──────────────────────────────────────────────────────╯\033[0m
""",
        "lang_prompt": "  Language / Язык [en/ru]: ",
    },
}


def _ask(prompt, default=""):
    """Prompt user for input."""
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def cli_setup(force=False):
    """Interactive first-run setup / onboarding."""
    env_file = CONFIG_DIR / "env"

    # Language selection (always in both languages)
    print()
    print("\033[1m    ██████╗  ██████╗ ██╗   ██╗ ██████╗ ██████╗ ██╗\033[0m")
    print("\033[1m   ██╔════╝ ██╔═══██╗██║   ██║██╔═══██╗██╔══██╗██║\033[0m")
    print("\033[1m   ██║  ███╗██║   ██║██║   ██║██║   ██║██████╔╝██║\033[0m")
    print("\033[1m   ██║   ██║██║   ██║╚██╗ ██╔╝██║   ██║██╔══██╗██║\033[0m")
    print("\033[1m   ╚██████╔╝╚██████╔╝ ╚████╔╝ ╚██████╔╝██║  ██║██║\033[0m")
    print("\033[1m    ╚═════╝  ╚═════╝   ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚═╝\033[0m")
    print()
    lang = _ask("  \033[2mLanguage / Язык\033[0m [\033[1men\033[0m/\033[1mru\033[0m]: ", "en").lower()
    if lang not in ("en", "ru"):
        lang = "en"

    s = SETUP_STRINGS[lang]
    print(s["welcome"])

    # Step 1: API keys
    print(s["step_keys"])
    openai_key = _ask(s["ask_openai"])
    anthropic_key = _ask(s["ask_anthropic"])

    if openai_key or anthropic_key:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        if openai_key:
            lines.append(f"export OPENAI_API_KEY={openai_key}")
        elif env_file.exists():
            # Preserve existing key
            for line in env_file.read_text().splitlines():
                if line.startswith("export OPENAI_API_KEY="):
                    lines.append(line)
                    break
        if anthropic_key:
            lines.append(f"export ANTHROPIC_API_KEY={anthropic_key}")
        elif env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("export ANTHROPIC_API_KEY="):
                    lines.append(line)
                    break
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        env_file.chmod(0o600)
        print(s["keys_saved"])

    # Step 2: Accessibility
    print(s["step_access"])
    _ask(s["ask_access_done"])

    # Step 3: Hotkeys tutorial
    print(s["step_hotkeys"])

    # Notes plugin
    notes_dir = PLUGINS_DIR / "notes"
    if not notes_dir.exists():
        print(s["step_plugin"])
        setup_notes = _ask(s["ask_plugin"], "y").lower()
        if setup_notes in ("y", "yes", "д", "да", ""):
            notes_dir.mkdir(parents=True, exist_ok=True)
            (notes_dir / "plugin.yaml").write_text(
                "name: notes\n"
                "description: Classify and save voice notes with AI\n"
                "trigger: shift+fn\n"
                "classifier_model: claude-haiku-4-5-20251001\n"
                "\n"
                f"output_dir: ~/govori-notes/{{year}}/{{month}}\n"
                f"index_file: ~/govori-notes/index/recent.jsonl\n"
                "\n"
                "whisper_prompt: \"\"\n",
                encoding="utf-8",
            )
            (notes_dir / "contexts.yaml").write_text(
                "- key: work\n"
                "  description: My day job\n"
                "\n"
                "- key: personal\n"
                "  description: Personal life, family, health\n",
                encoding="utf-8",
            )
            (notes_dir / "stuck.yaml").write_text(
                "# Optional: ongoing tasks to link notes to\n"
                "# - key: my_task\n"
                "#   description: What this task is about\n",
                encoding="utf-8",
            )
            print(s["plugin_created"])
        else:
            print(s["plugin_skipped"])

    # Save language to config
    cfg_data = _load_yaml(CONFIG_FILE)
    cfg_data["language"] = lang
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if yaml:
        CONFIG_FILE.write_text(
            yaml.dump(cfg_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        CONFIG_FILE.write_text(
            json.dumps(cfg_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Mark setup as done
    (CONFIG_DIR / ".setup_done").touch()

    print(s["done"])
    sys.exit(0)


def _is_first_run():
    """Check if setup has never been run."""
    return not (CONFIG_DIR / ".setup_done").exists()


# ── CLI subcommands ──────────────────────────────────────────────────────────
def cli_plugin(args):
    """Handle `govori plugin <subcommand>` CLI."""
    if not args:
        print("Usage: govori plugin <list|add|init|remove>")
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
            print("Usage: govori plugin init <name>")
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
            f"output_dir: ~/govori-notes/{{year}}/{{month}}\n"
            f"index_file: ~/govori-notes/index/recent.jsonl\n"
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
        print(f"Edit contexts.yaml to define your contexts, then restart govori.")

    elif sub == "remove":
        if len(args) < 2:
            print("Usage: govori plugin remove <name>")
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
        print("Usage: govori plugin <list|init|remove>")
        sys.exit(1)


VERSION = "0.1.0"


def cli_main():
    """Route CLI subcommands before starting the daemon."""
    args = sys.argv[1:]

    if "--version" in args or "-v" in args:
        print(f"govori {VERSION}")
        sys.exit(0)

    # Filter out flags like --gpt
    positional = [a for a in args if not a.startswith("--")]

    if positional and positional[0] == "setup":
        cli_setup(force=True)

    if positional and positional[0] == "plugin":
        cli_plugin(positional[1:])
        sys.exit(0)

    if positional and positional[0] == "notes":
        # Defer execution until module-level helpers are defined.
        # Handled near __main__.
        globals()["_NOTES_CLI_ARGS"] = positional[1:]
        return

    if positional and positional[0] == "note":
        # Text input subcommand: all args after "note" are joined as note body.
        # Read from stdin if no args provided (govori note < file.txt).
        if len(positional) < 2 and sys.stdin.isatty():
            print("Usage: govori note <text>  |  echo <text> | govori note")
            sys.exit(1)
        if len(positional) >= 2:
            text = " ".join(positional[1:])
        else:
            text = sys.stdin.read()
        globals()["_NOTE_CLI_TEXT"] = text
        return

    # Auto-trigger setup on first run
    if _is_first_run():
        cli_setup()


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

_api_key_env = CONFIG.get("api_key_env") or "OPENAI_API_KEY"
_api_key = os.environ.get(_api_key_env)
if not _api_key:
    print(f"{_api_key_env} not set — check ~/.config/govori/env", flush=True)
    sys.exit(1)
_base_url = CONFIG.get("base_url")
client = OpenAI(api_key=_api_key, base_url=_base_url) if _base_url else OpenAI(api_key=_api_key)

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
        print("ANTHROPIC_API_KEY not set — check ~/.config/govori/env", flush=True)
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

if "_NOTES_CLI_ARGS" not in globals() and "_NOTE_CLI_TEXT" not in globals():
    print("Govori ready.", flush=True)
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
            hud_window.setFrameOrigin_(AppKit.NSMakePoint(6, 0))
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
    save_or_merge_note(text, duration_sec)


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
    if rms < 0.0001:
        set_hud(False)
        print(f"(silence, rms={rms:.4f})", flush=True)
        return

    # ── NOTE MODE: fire-and-forget ────────────────────────────────────────────
    if note_mode:
        if not NOTES_CFG:
            print("notes plugin not installed — run: govori plugin init notes", flush=True)
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


# ── Merge-check pipeline ─────────────────────────────────────────────────────
MERGE_WINDOW_HOURS = 6
MERGE_CONFIDENCE_THRESHOLD = 0.85


def _find_merge_candidates(contexts, hours=MERGE_WINDOW_HOURS):
    """Return recent index entries matching context, within time window."""
    if not NOTES_CFG or not contexts:
        return []
    cutoff = datetime.datetime.now().astimezone() - datetime.timedelta(hours=hours)
    entries = _read_index_entries(limit=20)
    out = []
    ctx_set = set(contexts)
    for e in entries:
        try:
            created = datetime.datetime.fromisoformat(e.get("created", ""))
        except Exception:
            continue
        if created < cutoff:
            continue
        if not (set(e.get("contexts") or []) & ctx_set):
            continue
        if not Path(e.get("path", "")).exists():
            continue
        out.append(e)
    return out


def _decide_merge(new_text, candidates):
    """Ask Haiku whether new_text continues one of candidates.
    Returns dict: {action: 'new'|'merge', target_id, confidence, reason}."""
    if not candidates:
        return {"action": "new", "target_id": None, "confidence": 1.0, "reason": "no candidates"}
    anthropic_client = _get_anthropic_client()
    if anthropic_client is None:
        return {"action": "new", "target_id": None, "confidence": 0.0, "reason": "no anthropic"}

    cand_block = "\n".join(
        f"[{i}] id={e['id']}  ({e.get('type','')}/{e.get('urgency','')})  {e.get('summary','')[:160]}"
        for i, e in enumerate(candidates)
    )
    system = """You decide whether a new voice note is a CONTINUATION of an existing recent note or a NEW standalone thought.

Rules:
- MERGE only if the new text clearly extends, corrects, or adds detail to ONE existing note on the SAME specific topic.
- If the new text introduces a different subject, decision, or action — it's NEW.
- When in doubt — prefer NEW. A false merge silently loses information; a false new just creates one extra file.

Return STRICT JSON only:
{"action": "new" | "merge", "target_index": <int or null>, "confidence": <0.0-1.0>, "reason": "<short>"}"""
    user = f"EXISTING RECENT NOTES:\n{cand_block}\n\nNEW TEXT:\n{new_text}"
    try:
        resp = anthropic_client.messages.create(
            model=NOTES_CFG["classifier_model"],
            max_tokens=200,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        action = data.get("action", "new")
        idx = data.get("target_index")
        conf = float(data.get("confidence", 0.0))
        reason = data.get("reason", "")
        target_id = None
        if action == "merge" and isinstance(idx, int) and 0 <= idx < len(candidates):
            target_id = candidates[idx]["id"]
        else:
            action = "new"
        return {"action": action, "target_id": target_id, "confidence": conf, "reason": reason,
                "candidate": candidates[idx] if target_id else None}
    except Exception as e:
        print(f"merge decision error: {e}", flush=True)
        return {"action": "new", "target_id": None, "confidence": 0.0, "reason": str(e)}


def _confirm_merge(decision, new_text):
    """Hook for user confirmation. Currently auto-resolves by threshold.
    Future: will show HUD panel and wait for user choice.
    Returns final decision dict (same shape)."""
    if decision["action"] == "merge" and decision["confidence"] >= MERGE_CONFIDENCE_THRESHOLD:
        return decision
    # Conservative fallback: don't merge
    return {"action": "new", "target_id": None,
            "confidence": decision["confidence"],
            "reason": f"below threshold ({decision['confidence']:.2f}) — " + decision.get("reason", "")}


def _apply_merge_append(candidate, new_text, duration_sec):
    """Append new_text as a timestamped block to the candidate's markdown file."""
    path = Path(candidate["path"])
    now = datetime.datetime.now().astimezone()
    ts_short = now.strftime("%H:%M")

    original = path.read_text(encoding="utf-8")
    fm_lines, body = _split_frontmatter(original)
    if fm_lines:
        fm_lines = _update_frontmatter_amended(fm_lines, now.isoformat(timespec="seconds"))

    appended = body.rstrip() + f"\n\n## {ts_short} (voice)\n{new_text.strip()}\n"
    if fm_lines:
        path.write_text("\n".join(fm_lines) + "\n\n" + appended, encoding="utf-8")
    else:
        path.write_text(appended, encoding="utf-8")

    # Append a merge record to the index (not a new file, but trail visibility)
    try:
        index_path = _resolve_path(NOTES_CFG["index_file"], now)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        merge_entry = {
            "id": f"{candidate['id']}+merge_{now.strftime('%H%M')}",
            "created": now.isoformat(timespec="seconds"),
            "path": str(path),
            "contexts": candidate.get("contexts", []),
            "type": candidate.get("type", ""),
            "urgency": candidate.get("urgency", ""),
            "merged_into": candidate["id"],
            "summary": new_text.strip()[:200],
        }
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(merge_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"(merge index write failed: {e})", flush=True)

    print(f"⇪ merged into: {path.name} (conf n/a)", flush=True)


def save_or_merge_note(text, duration_sec):
    """Entry point: classify → merge-check → either merge or save as new."""
    if not NOTES_CFG:
        print("notes plugin not configured", flush=True)
        return
    try:
        meta = classify_note(text)
        candidates = _find_merge_candidates(meta["contexts"])
        decision = _decide_merge(text, candidates)
        decision = _confirm_merge(decision, text)

        if decision["action"] == "merge" and decision.get("target_id"):
            cand = decision.get("candidate") or next(
                (c for c in candidates if c["id"] == decision["target_id"]), None
            )
            if cand:
                _apply_merge_append(cand, text, duration_sec)
                return
        # Fall through: save as new note (reuses classification we already computed)
        _save_note_with_meta(text, duration_sec, meta)
    except Exception as e:
        print(f"save_or_merge_note error: {e}", flush=True)


def _save_note_with_meta(text, duration_sec, meta):
    """Write note using a pre-computed meta dict (avoids re-classifying)."""
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

# ── Notes CLI (picker + voice amend) ─────────────────────────────────────────
def _read_index_entries(limit=30):
    if not NOTES_CFG:
        return []
    # Search a few recent months to collect up to `limit` entries.
    entries = []
    now = datetime.datetime.now()
    seen = set()
    for months_back in range(0, 12):
        d = now - datetime.timedelta(days=30 * months_back)
        path = _resolve_path(NOTES_CFG["index_file"], d)
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
    """Record audio from mic until user presses Enter. Returns numpy array."""
    chunks = []

    def cb(indata, frames, time_info, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=cb)
    stream.start()
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
            model=NOTES_CFG["classifier_model"],
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
    if not NOTES_CFG:
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
        os.system(f"{editor} {path}")
        return
    if choice != "r":
        return

    audio = _record_until_enter()
    if audio is None or len(audio) / SAMPLE_RATE < 0.3:
        print("(too short)")
        return

    print("■ Transcribing…", flush=True)
    instruction = _encode_and_transcribe(audio)
    if not instruction:
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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "_NOTES_CLI_ARGS" in globals():
        cli_notes(_NOTES_CLI_ARGS)
        sys.exit(0)
    if "_NOTE_CLI_TEXT" in globals():
        text = _NOTE_CLI_TEXT.strip()
        if not text:
            print("(empty note)")
            sys.exit(1)
        if not NOTES_CFG:
            print("notes plugin not installed. Run: govori plugin init notes")
            sys.exit(1)
        print(f"→ {text[:120]}{'…' if len(text) > 120 else ''}", flush=True)
        save_or_merge_note(text, duration_sec=0)
        sys.exit(0)
    print(f"Govori started. Model: {MODEL}. Hold fn to record.", flush=True)

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
