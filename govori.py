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
import subprocess
from pathlib import Path

import numpy as np
import av
import sounddevice as sd
import openai
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
        "predict_model": "llama-3.3-70b-versatile",
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


_PROMPT_BUDGET_BYTES = 896   # Groq whisper-large-v3-turbo hard limit (UTF-8 BYTES, not chars)
_PROMPT_BUDGET_CHARS = _PROMPT_BUDGET_BYTES   # legacy alias


def _tokenize_prompt_terms(text):
    """Split a prompt string into individual vocabulary terms.

    For segments containing a colon (e.g. "Проекты: Marquiz, SKVO"), keep only
    the part after the last colon so category labels are discarded. Long phrases
    (> 40 chars) are dropped — they're instructional preamble, which Whisper
    ignores anyway.
    """
    terms = []
    for part in re.split(r'[,.;\n]', text):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            part = part.rsplit(':', 1)[1].strip()
        if part and len(part) <= 40:
            terms.append(part)
    return terms


def _notes_corpus_text(plugins):
    """Lowercased concatenation of all .md notes under plugin output_dirs.

    Used for frequency scoring of prompt terms. Returns "" when no notes exist.
    """
    bases = set()
    for plugin in plugins.values():
        out = plugin.get("output_dir")
        if not out:
            continue
        base_str = out.split('{', 1)[0].rstrip('/') or out
        bases.add(Path(base_str).expanduser())

    chunks = []
    for base in bases:
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            try:
                chunks.append(md.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
    return "\n".join(chunks).lower()


def build_whisper_prompt(config, plugins, budget_bytes=_PROMPT_BUDGET_BYTES):
    """Assemble a byte-budget-constrained Whisper prompt from config + plugin vocabularies.

    Auto-prioritises by term usage frequency in past transcriptions (case-
    insensitive substring count over notes corpus), so frequently-dictated
    terms float to the top and unseen terms get a baseline score of 1 (not
    dropped on first sight). Greedy-packs into `budget_bytes` of UTF-8 output
    (Groq whisper-large-v3-turbo enforces a byte limit, not a char limit —
    each cyrillic character costs 2 bytes, so an all-cyrillic prompt of
    budget_bytes/2 chars is the practical maximum).
    """
    raw_parts = []
    base_cfg = config.get("whisper_prompt", "").strip()
    if base_cfg:
        raw_parts.append(base_cfg)
    for plugin in plugins.values():
        p = plugin.get("whisper_prompt", "").strip()
        if p:
            raw_parts.append(p)

    seen = set()
    terms = []
    for part in raw_parts:
        for t in _tokenize_prompt_terms(part):
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(t)
    if not terms:
        return ""

    corpus = _notes_corpus_text(plugins)
    scored = sorted(
        (-max(1, corpus.count(t.lower())), idx, t) for idx, t in enumerate(terms)
    )

    kept, dropped = [], []
    remaining = budget_bytes
    for _, _, term in scored:
        # UTF-8 byte cost: cyrillic chars cost 2 bytes each. Groq enforces a byte
        # limit, not a char limit. `, ` separator is ASCII (2 bytes).
        cost = len(term.encode("utf-8")) + (2 if kept else 0)
        if cost <= remaining:
            kept.append(term)
            remaining -= cost
        else:
            dropped.append(term)

    used = budget_bytes - remaining
    if dropped:
        preview = ", ".join(dropped[:6])
        more = f" +{len(dropped) - 6}" if len(dropped) > 6 else ""
        print(
            f"  whisper_prompt: {len(kept)} kept, {len(dropped)} dropped "
            f"({used}/{budget_bytes} bytes) — dropped: {preview}{more}",
            flush=True,
        )
    else:
        print(
            f"  whisper_prompt: {len(kept)} terms ({used}/{budget_bytes} bytes)",
            flush=True,
        )

    return ", ".join(kept)


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
\033[33m  ── Step 1/4 ─ API Keys ──────────────────────────────\033[0m

  Govori needs an OpenAI API key for speech-to-text.
  \033[2mOptionally, add an Anthropic key for smart note classification.\033[0m

  \033[2mYour keys are stored locally in ~/.config/govori/env\033[0m
""",
        "ask_openai": "  \033[1mOpenAI API key\033[0m (sk-...): ",
        "ask_anthropic": "  \033[1mAnthropic API key\033[0m (sk-ant-..., Enter to skip): ",
        "keys_saved": "\n  \033[32m✓ Keys saved to ~/.config/govori/env\033[0m\n",
        "step_privacy": """
\033[33m  ── Step 2/4 ─ Privacy Notice ────────────────────────\033[0m

  Govori sends data to cloud APIs for processing:

    \033[1mVoice audio\033[0m   -->  OpenAI Whisper API (speech-to-text)
    \033[1mNote text\033[0m     -->  Anthropic Claude API (classification)
    \033[1mDictated text\033[0m -->  OpenAI Chat Completions (predict/rephrase mode)
                       \033[2mNote: predict uses your configured base_url
                       (may be a third-party endpoint like Groq)\033[0m

  \033[2mDictation audio is not retained after a successful transcription.
  Note-mode audio IS saved locally as Opus to
  ~/life/state/govori-audio/YYYY-MM-DD/HHMMSS_Ns.opus
  (~14 MB/hour, no automatic deletion). Anthropic does not retain note
  text per their policy. Keys stay local on your machine.\033[0m

""",
        "step_access": """
\033[33m  ── Step 3/4 ─ Accessibility Permission ──────────────\033[0m

  Govori needs Accessibility access to listen for the \033[1mfn\033[0m key.

  \033[36mSystem Settings → Privacy & Security → Accessibility\033[0m
  \033[36m→ Add your terminal app (Terminal / iTerm / Ghostty)\033[0m

""",
        "ask_access_done": "  \033[2mPress Enter when done...\033[0m",
        "step_hotkeys": """
\033[33m  ── Step 4/4 ─ How to Use ────────────────────────────\033[0m

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
\033[33m  ── Шаг 1/4 ─ API-ключи ─────────────────────────────\033[0m

  Govori использует OpenAI API для распознавания речи.
  \033[2mОпционально: ключ Anthropic для умной классификации заметок.\033[0m

  \033[2mКлючи хранятся локально в ~/.config/govori/env\033[0m
""",
        "ask_openai": "  \033[1mOpenAI API ключ\033[0m (sk-...): ",
        "ask_anthropic": "  \033[1mAnthropic API ключ\033[0m (sk-ant-..., Enter чтобы пропустить): ",
        "keys_saved": "\n  \033[32m✓ Ключи сохранены в ~/.config/govori/env\033[0m\n",
        "step_privacy": """
\033[33m  ── Шаг 2/4 ─ Конфиденциальность ─────────────────────\033[0m

  Govori отправляет данные в облачные API для обработки:

    \033[1mАудио голоса\033[0m       -->  OpenAI Whisper API (распознавание речи)
    \033[1mТекст заметок\033[0m      -->  Anthropic Claude API (классификация)
    \033[1mНадиктованный текст\033[0m -->  OpenAI Chat Completions (predict / rephrase)
                            \033[2mИспользует твой настроенный base_url
                            (может быть сторонним, например Groq)\033[0m

  \033[2mАудио для обычной диктовки не сохраняется после успешной
  транскрипции. Аудио заметок (note-mode) СОХРАНЯЕТСЯ локально как Opus в
  ~/life/state/govori-audio/YYYY-MM-DD/HHMMSS_Ns.opus
  (~14 МБ/час, автоматического удаления нет). Anthropic не хранит текст
  заметок согласно их политике. Ключи остаются на твоём устройстве.\033[0m

""",
        "step_access": """
\033[33m  ── Шаг 3/4 ─ Разрешение Accessibility ──────────────\033[0m

  Govori нужен доступ к Accessibility чтобы слушать клавишу \033[1mfn\033[0m.

  \033[36mСистемные настройки → Конфиденциальность → Универсальный доступ\033[0m
  \033[36m→ Добавь свой терминал (Terminal / iTerm / Ghostty)\033[0m

""",
        "ask_access_done": "  \033[2mНажми Enter когда готово...\033[0m",
        "step_hotkeys": """
\033[33m  ── Шаг 4/4 ─ Как пользоваться ───────────────────────\033[0m

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
    lang = CONFIG.get("language", "en")
    if lang not in TOOLTIP_STRINGS:
        lang = "en"
    text = TOOLTIP_STRINGS[lang].get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


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

    # Step 2: Privacy notice
    print(s["step_privacy"])

    # Step 3: Accessibility — trigger system prompt
    import ctypes, objc
    from Foundation import NSDictionary
    _hi = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/ApplicationServices.framework/Frameworks/HIServices.framework/HIServices')
    _hi.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
    _hi.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
    _hi.AXIsProcessTrustedWithOptions(
        objc.pyobjc_id(NSDictionary.dictionaryWithObject_forKey_(True, 'AXTrustedCheckOptionPrompt'))
    )
    print(s["step_access"])
    _ask(s["ask_access_done"])

    # Step 4: Hotkeys tutorial
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


def cli_add(args):
    """Add words to vocabulary (terms.md/people.md), rebuild whisper_prompt, sync to VPS.

    Usage: govori add [-p|-t] <word>...
      -p, --people   append to people.md (format: "Имя Фамилия — контекст")
      -t, --terms    append to terms.md (default)
    """
    target = "terms"
    words = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-p", "--people"):
            target = "people"
        elif a in ("-t", "--terms"):
            target = "terms"
        elif a in ("-h", "--help"):
            print("Usage: govori add [-p|-t] <word>...")
            print("  -p, --people   append to people.md (format: 'Имя — контекст')")
            print("  -t, --terms    append to terms.md (default)")
            sys.exit(0)
        else:
            words.append(a)
        i += 1

    if not words:
        print("Usage: govori add [-p|-t] <word>...")
        sys.exit(1)

    target_file = CONFIG_DIR / ("people.md" if target == "people" else "terms.md")
    if not target_file.exists():
        print(f"Vocabulary file not found: {target_file}")
        sys.exit(1)

    def _key(s):
        """Normalize a line for dedup — for people compare only left of dash."""
        s = s.strip()
        if target == "people" and "—" in s:
            s = s.split("—", 1)[0].strip()
        return s.lower()

    existing = set()
    for line in target_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        existing.add(_key(s))

    added, skipped = [], []
    for w in words:
        k = _key(w)
        if not k:
            continue
        if k in existing:
            skipped.append(w)
        else:
            added.append(w)
            existing.add(k)

    if added:
        text = target_file.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        text += "\n".join(added) + "\n"
        target_file.write_text(text, encoding="utf-8")
        print(f"✓ {target_file.name}: +{len(added)} ({', '.join(added)})")
    if skipped:
        print(f"  skipped (already present): {', '.join(skipped)}")
    if not added:
        sys.exit(0)

    import subprocess

    refresh = CONFIG_DIR / "refresh-prompt.sh"
    if refresh.exists():
        r = subprocess.run(["bash", str(refresh)], capture_output=True, text=True)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith(("whisper_prompt:", "language:", "preview:")):
                    print(f"  {stripped}")
        else:
            print(f"✗ refresh-prompt failed: {r.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  (refresh-prompt.sh not found at {refresh}, skipping whisper_prompt rebuild)")

    r = subprocess.run(
        [
            "rsync", "-az",
            str(CONFIG_DIR / "config.yaml"),
            str(CONFIG_DIR / "terms.md"),
            str(CONFIG_DIR / "people.md"),
            "vps:.config/govori/",
        ],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print("✓ synced to VPS")
    else:
        print(f"✗ VPS sync failed: {r.stderr.strip()}", file=sys.stderr)

    sys.exit(0)


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

    if positional and positional[0] == "add":
        cli_add(positional[1:])
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
    "субтитры создавал dimatorzok",
    "редактор субтитров а.семкин корректор а.кулакова",
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
client = (
    OpenAI(api_key=_api_key, base_url=_base_url, timeout=30.0, max_retries=0)
    if _base_url
    else OpenAI(api_key=_api_key, timeout=30.0, max_retries=0)
)

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
_retry_buffer = None      # audio chunks saved for retry
_retry_count = 0          # current retry attempt count
_retry_in_progress = False
_retry_mode_snapshot = None
_hud_error_mode = None    # current error mode: "error_retryable", "error_fatal", or None
_health_monitor_owns_hud = False

# Sentinel for non-retryable API errors (4xx other than 408/429).
# Distinct from None which signals transient failures eligible for retry.
PERMANENT_API_ERROR = object()

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
    global _retry_count, _retry_in_progress
    if _hud_error_mode != "error_retryable":
        return
    with _state_lock:
        if _retry_buffer is None or _retry_in_progress:
            retry_exhausted = False
            retry_attempt = None
        elif _retry_count >= 3:
            retry_exhausted = True
            retry_attempt = None
        else:
            _retry_count += 1
            _retry_in_progress = True
            retry_exhausted = False
            retry_attempt = _retry_count
    if retry_exhausted:
        set_hud(True, mode="error_fatal", tooltip=_tooltip("retry_exhausted"))
        return
    if retry_attempt is None:
        return
    set_hud(True, mode="transcribing", tooltip=_tooltip("retry_attempt", n=retry_attempt, total=3))
    threading.Thread(target=_retry_transcription, daemon=True).start()


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
        if _hud_error_mode == "error_retryable":
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.85, 0.3, 1.0)
            )
        elif _hud_error_mode == "error_fatal":
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
    is_error = _hud_error_mode in ("error_retryable", "error_fatal")
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




def _retry_transcription():
    """Re-transcribe using _retry_buffer. Runs in daemon thread after user click."""
    global _retry_buffer, _retry_count, _retry_in_progress, _retry_mode_snapshot
    try:
        with _state_lock:
            buf_copy = _retry_buffer
            mode_snapshot = _retry_mode_snapshot or {}
        if buf_copy is None:
            return
        # Replicate the encoding step from stop_and_transcribe
        audio = np.concatenate(buf_copy, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE
        text = _encode_and_transcribe(audio, timeout=_timeout_for_duration(duration))
        if text is PERMANENT_API_ERROR:
            # Permanent error on retry — escalate to fatal, drop buffer
            with _state_lock:
                _retry_buffer = None
                _retry_count = 0
                _retry_mode_snapshot = None
            set_hud(True, mode="error_fatal", tooltip=_tooltip("api_network"))
            return
        if text is None:
            # Still failing -- show retryable error again
            set_hud(True, mode="error_retryable", tooltip=_tooltip("api_network"))
            return
        if not text or text in WHISPER_HALLUCINATIONS or _is_hallucination(text):
            print("(empty)", flush=True)
            set_hud(False)
            return
        with _state_lock:
            _retry_count = 0
            _retry_buffer = None
            _retry_mode_snapshot = None
        if mode_snapshot.get("note_mode"):
            save_or_merge_note(text, mode_snapshot.get("duration", duration))
        else:
            paste_text(text + " ")
            if mode_snapshot.get("predict_mode"):
                AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(
                    lambda t=text: show_predict_menu(t)
                )
            elif mode_snapshot.get("auto_send"):
                time.sleep(0.3)
                _press_enter()
        set_hud(False)
    finally:
        with _state_lock:
            _retry_in_progress = False


def set_hud(visible, mode="recording", tooltip=None, count=None):
    global _hud_error_mode

    def _update():
        global _hud_error_mode
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
            _hud_error_mode = "error_retryable"
        elif mode == "error_fatal":
            hud_label.setStringValue_("\u2717")
            hud_label.setTextColor_(
                AppKit.NSColor.colorWithRed_green_blue_alpha_(1.0, 0.3, 0.3, 1.0)
            )
            # Static -- remove any existing animation
            hud_label.setWantsLayer_(True)
            hud_label.layer().removeAnimationForKey_("pulse")
            _hud_error_mode = "error_fatal"

        # Mouse events: clickable for error modes, ignored for all others
        if is_error:
            hud_window.setIgnoresMouseEvents_(False)
        else:
            hud_window.setIgnoresMouseEvents_(True)
            if not tooltip:
                _hide_tooltip()
            _hud_error_mode = None
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

# ── Audio ─────────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_chunks.append(indata.copy())


def _start_mic_stream():
    # Initialize PortAudio input stream. State flags (recording, cancelled,
    # audio_chunks, _retry_count, auto_send) must already be set synchronously
    # by the caller under _state_lock.
    global audio_stream, recording
    with _state_lock:
        if not recording or cancelled:
            return  # fn-up beat us / cancelled before we started
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception:
                pass
            audio_stream = None
        try:
            audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback,
            )
            audio_stream.start()
        except sd.PortAudioError as e:
            recording = False
            audio_stream = None
            err_str = str(e).lower()
            if "permission" in err_str or "denied" in err_str:
                tooltip_key = "mic_denied"
            else:
                tooltip_key = "no_mic"
            set_hud(True, mode="error_fatal", tooltip=_tooltip(tooltip_key))
            print(f"! Mic error: {e}", flush=True)


def _show_recording_hud():
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


def _timeout_for_duration(duration_sec):
    """Scale request timeout with audio length — longer audio takes longer to process.
    Floor is 30s per SEC-03 (was 5-8s before; phase 01.1-01)."""
    if duration_sec >= 60:
        return 60.0
    if duration_sec >= 30:
        return 45.0
    if duration_sec >= 20:
        return 35.0
    return 30.0


def _encode_and_transcribe(audio, timeout=30.0):
    """Encode mono float32 audio → OGG/Opus → Whisper.
    Returns: text (str), None (transient failure — caller may retry),
    or PERMANENT_API_ERROR sentinel (4xx other than 408/429 — do not retry)."""
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
        result = client.with_options(timeout=timeout, max_retries=0).audio.transcriptions.create(
            model=MODEL,
            file=buf,
            language=LANGUAGE,
            temperature=0,
            prompt=WHISPER_PROMPT,
        )
        return result.text.strip()
    except openai.APITimeoutError:
        print(f"! Transcription timed out ({timeout}s)", flush=True)
        return None
    except openai.APIConnectionError as e:
        print(f"! Connection error: {e}", flush=True)
        return None
    except openai.APIStatusError as e:
        # 408 Request Timeout, 429 Too Many Requests, 5xx → transient (retry OK).
        # Other 4xx (400 bad request, 401 auth, 403 forbidden, 404, 422 validation) → permanent.
        if e.status_code >= 500 or e.status_code in (408, 429):
            print(f"! Server error ({e.status_code})", flush=True)
            return None
        print(f"! API error ({e.status_code}): {e} (permanent — won't retry)", flush=True)
        return PERMANENT_API_ERROR


def _transcribe_with_auto_retries(audio, duration_sec, on_progress=None, max_retries=2):
    """One initial attempt + up to max_retries auto-retries. Returns text or None.

    on_progress(attempt, total_attempts, total_sec_left) is called every second
    ONLY during retry attempts (attempt > 1). `total_sec_left` counts down the
    combined retry budget across all retries (not per-cycle), so the user sees a
    single steady countdown from `max_retries * timeout` down to 1.
    """
    timeout = _timeout_for_duration(duration_sec)
    total_attempts = max_retries + 1
    total_retry_secs = max_retries * int(timeout)
    retry_ticks_elapsed = 0

    for attempt in range(1, total_attempts + 1):
        result = {"text": None, "done": False}

        def _do():
            try:
                result["text"] = _encode_and_transcribe(audio, timeout=timeout)
            finally:
                result["done"] = True

        worker = threading.Thread(target=_do, daemon=True)
        worker.start()

        sec_left_in_attempt = int(timeout)
        while not result["done"] and sec_left_in_attempt > 0:
            if on_progress is not None and attempt > 1:
                total_remaining = total_retry_secs - retry_ticks_elapsed
                on_progress(attempt, total_attempts, total_remaining)
                retry_ticks_elapsed += 1
            time.sleep(1)
            sec_left_in_attempt -= 1

        worker.join(timeout=timeout + 2)

        if result["text"] is PERMANENT_API_ERROR:
            return PERMANENT_API_ERROR
        if result["text"] is not None:
            return result["text"]

    return None


_FOREIGN_SCRIPT_RE = re.compile(
    "["
    "　-鿿"      # CJK symbols, punctuation, hiragana, katakana, unified ideographs
    "가-힯"      # Hangul syllables
    "豈-﫿"      # CJK compatibility ideographs
    "＀-￯"      # halfwidth/fullwidth forms
    "֐-ۿ"      # Hebrew, Arabic
    "܀-޿"      # Syriac
    "ऀ-෿"      # Indic scripts
    "฀-࿿"      # Thai, Lao, Tibetan
    "က-႟"      # Myanmar
    "Ⴀ-ჿ"      # Georgian
    "ሀ-፿"      # Ethiopic
    "]"
)


def _is_hallucination(text):
    text_check = text.lower().strip().rstrip(".!?,;:…").strip()
    if text_check in WHISPER_HALLUCINATIONS or text.lower().strip() in WHISPER_HALLUCINATIONS:
        return True
    # Only Russian + Latin + digits/punct allowed; anything else = Whisper leak into another language.
    if _FOREIGN_SCRIPT_RE.search(text):
        return True
    return False


def _save_note_audio_background(audio, duration_sec):
    """Persist recorded audio as compressed Opus for later recall/verification.

    Writes to ~/life/state/govori-audio/YYYY-MM-DD/HHMMSS_Nsec.opus.
    Silently no-op on any error — audio persistence is best-effort and must not
    affect note-save success.

    Sizing: libopus @ 24kbps voice ≈ 3 KB/sec. A 10s note → ~30 KB. Month of
    heavy use → <30 MB. Retention is the user's problem (no auto-cleanup yet).
    """
    try:
        import datetime as _dt
        now = _dt.datetime.now()
        date_dir = Path.home() / "life" / "state" / "govori-audio" / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        stem = now.strftime("%H%M%S") + f"_{int(round(duration_sec))}s"
        out_path = date_dir / f"{stem}.opus"

        # Normalize + encode mirroring _transcribe_http_call's approach.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        audio_n = (audio / peak * 0.9) if peak > 0 else audio
        audio_int16 = (audio_n * 32767).astype(np.int16)

        container = av.open(str(out_path), mode="w", format="ogg")
        stream = container.add_stream("libopus", rate=SAMPLE_RATE, layout="mono")
        # Voice-optimized: 32 kbps is comfortable for speech recall + spot-check
        # of whisper errors. At 16kHz mono this is ~4 KB/sec → 14 MB/hour.
        stream.bit_rate = 32_000
        frame = av.AudioFrame.from_ndarray(audio_int16.reshape(1, -1), format="s16", layout="mono")
        frame.rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        size_kb = out_path.stat().st_size / 1024
        print(f"✓ audio saved: {out_path.name} ({size_kb:.1f} KB)", flush=True)
    except Exception as e:
        print(f"! audio-save failed (non-fatal): {e}", flush=True)


def _note_pipeline_background(audio, duration_sec):
    """Full note pipeline: transcribe → filter → classify → save. No HUD updates."""
    global _retry_buffer, _retry_count, _retry_mode_snapshot
    threading.Thread(
        target=lambda a=audio, d=duration_sec: _save_note_audio_background(a, d),
        daemon=True,
    ).start()
    text = _transcribe_with_auto_retries(audio, duration_sec)
    if text is PERMANENT_API_ERROR:
        set_hud(True, mode="error_fatal", tooltip=_tooltip("api_network"))
        return
    if text is None:
        with _state_lock:
            _retry_buffer = [audio]  # already concatenated, wrap in list for retry compat
            _retry_count = 0
            _retry_mode_snapshot = {
                "note_mode": True,
                "predict_mode": False,
                "auto_send": False,
                "duration": duration_sec,
            }
        set_hud(True, mode="error_retryable", tooltip=_tooltip("api_network"))
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
    global recording, audio_stream, transcribing, _retry_buffer, _retry_count, _retry_mode_snapshot
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
    print(f"[debug] chunks={len(audio_chunks)} total_samples={total_samples} dur={total_samples/SAMPLE_RATE:.2f}s", flush=True)
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

    duration = total_samples / SAMPLE_RATE

    def _show_progress(n, total, sec_left):
        set_hud(True, mode="countdown", count=sec_left)

    text = _transcribe_with_auto_retries(audio, duration, on_progress=_show_progress)
    transcribing = False

    if text is PERMANENT_API_ERROR:
        if not cancelled:
            set_hud(True, mode="error_fatal", tooltip=_tooltip("api_network"))
        else:
            set_hud(False)
        return

    if text is None:
        if not cancelled:
            with _state_lock:
                _retry_buffer = list(audio_chunks)  # copy for retry safety (Pitfall 4)
                _retry_count = 0
                _retry_mode_snapshot = {
                    "note_mode": bool(note_mode),
                    "predict_mode": bool(predict_mode),
                    "auto_send": bool(auto_send),
                    "duration": duration,
                }
            set_hud(True, mode="error_retryable", tooltip=_tooltip("api_network"))
        else:
            set_hud(False)
        return

    set_hud(False)

    if cancelled:
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


def cancel_recording(skip_hud=False, quiet=False):
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
    if not skip_hud:
        set_hud(False)
    if not quiet:
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


def segment_by_context(text, contexts):
    """Разбить текст заметки на секции '## <context>' когда затронуто
    несколько проектов. Наследует временные/приоритетные маркеры («завтра»,
    «срочно», «до пятницы») из общего начала ко всем секциям — они относятся
    ко всем упомянутым проектам пока явно не переопределены.

    Fallback: если contexts пуст или содержит 1 элемент — возвращает текст
    без изменений. Если Haiku недоступен или ответ невалидный — возвращает
    оригинал. Никогда не блокирует pipeline.
    """
    if not text or len(contexts) < 2:
        return text
    if NOTES_CFG is None:
        return text
    client = _get_anthropic_client()
    if client is None:
        return text

    contexts_desc = NOTES_CFG["contexts_desc"]
    contexts_list = ", ".join(contexts)

    system = f"""Ты структурируешь многотематическую голосовую заметку.

На входе — один связный текст заметки, в котором пользователь последовательно
говорит о нескольких проектах. Твоя задача: разбить текст на секции
по проектам, НЕ теряя информацию.

КОНТЕКСТЫ ПОЛЬЗОВАТЕЛЯ:
{contexts_desc}

В ЭТОЙ ЗАМЕТКЕ ЗАТРОНУТЫ: {contexts_list}

ПРАВИЛА:
1. Для каждого контекста из списка создай секцию с заголовком `## <context_key>`
   (ровно тот ключ что в списке выше — marquiz, persona, skvo и т.д., не русские названия).
2. В каждую секцию помести фрагмент текста, относящийся к этому контексту.
3. Секции располагай в том порядке, как контексты упоминались в тексте.
4. НЕ меняй слова пользователя. НЕ переписывай. НЕ сокращай. НЕ добавляй.
5. НАСЛЕДОВАНИЕ временных/приоритетных маркеров:
   - Если в начале заметки есть маркер типа «завтра», «срочно», «до пятницы»,
     «в первую очередь» и т.п., и он явно относится к нескольким упомянутым
     проектам — повтори этот маркер в каждой релевантной секции.
   - Если маркер сказан ЯВНО про один конкретный пункт (например, «...проверить
     их завтра» — только про документы Наташи) — оставляй его ТОЛЬКО в той секции.
   - Если не уверен к скольки проектам относится маркер — оставь только там
     где он изначально был сказан.
6. Разговорные связки ("А ещё", "И ещё нужно", "Так ещё") оставляй в тех
   секциях где они изначально были — не теряй интонацию переключения.
7. Если какой-то контекст из списка фактически не упоминается в тексте —
   НЕ создавай для него секцию (лучше вернуть меньше секций чем пустые).

Верни ТОЛЬКО текст с секциями, без пояснений, кавычек, markdown-обёрток.
Не добавляй преамбулу перед первой секцией `##` — заголовок должен быть
первой строкой."""

    try:
        resp = client.messages.create(
            model=NOTES_CFG["classifier_model"],
            max_tokens=max(800, len(text) * 3),
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        segmented = resp.content[0].text.strip()
        if segmented.startswith("```"):
            segmented = re.sub(r"^```(?:\w+)?\s*", "", segmented)
            segmented = re.sub(r"\s*```\s*$", "", segmented)
        # Валидация: должно начинаться с `## `
        if not segmented.startswith("## "):
            print(f"  [segment] skipped — no section headers in output", flush=True)
            return text
        # Валидация длины: разбивка может немного увеличить (заголовки +
        # возможное дублирование inheritance markers), но не в разы.
        if len(segmented) < len(text) * 0.7 or len(segmented) > len(text) * 2.5:
            print(f"  [segment] skipped — length delta suspicious "
                  f"(orig {len(text)}, segmented {len(segmented)})", flush=True)
            return text
        print(f"  [segment] split into {segmented.count('## ')} section(s)", flush=True)
        return segmented
    except Exception as e:
        print(f"  [segment] failed (non-fatal): {e}", flush=True)
        return text


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
- title: 2-5 word slug in latin kebab-case (e.g. "work-deploy-issue"). When
  the note spans multiple contexts, pick a title that reflects the PRIMARY
  (first-mentioned) context, not a generic summary.
- contexts: array of ALL applicable context keys. Return MULTIPLE entries
  when the note clearly mentions distinct projects. Better to over-tag
  than under-tag: if the user talks about 3 projects, return all 3 keys.
  Order: first-mentioned first (this becomes the "primary" context).
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
        # Multi-context split: if Haiku classifier returned multiple contexts,
        # ask Haiku to structure the text into '## <context>' sections with
        # inheritance of temporal markers across sections.
        if len(meta.get("contexts", [])) > 1:
            text = segment_by_context(text, meta["contexts"])
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


def generate_rephrasings(text):
    """Generate 3 alternative phrasings of the given text, preserving meaning."""
    try:
        resp = client.chat.completions.create(
            model=CONFIG.get("predict_model", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rephrasing assistant. Given a piece of text, "
                        "produce 3 distinct alternative phrasings that preserve "
                        "the original meaning but vary in wording, tone, or "
                        "structure. Keep roughly the same length and the SAME "
                        "language as the input. Do not add or remove information. "
                        "Return JSON: {\"rephrasings\": [\"...\", \"...\", \"...\"]}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.7,
        )
        data = json.loads(resp.choices[0].message.content)
        items = data.get("rephrasings", [])
        if isinstance(items, list) and len(items) >= 1:
            return [str(v) for v in items[:3]]
    except Exception as e:
        print(f"Rephrase error: {e}", flush=True)
    return []


class PredictController(AppKit.NSObject):
    _rephrasings = []
    _pasted_len = 0

    def pickRephrasing_(self, sender):
        idx = sender.tag()
        if 0 <= idx < len(self._rephrasings):
            text = self._rephrasings[idx]
            n = self._pasted_len
            print(f"✦ rephrase: {text}", flush=True)
            def _replace():
                _delete_chars(n)
                paste_text(text + " ")
            threading.Thread(target=_replace, daemon=True).start()


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


def setup_predict():
    global _predict_controller
    _predict_controller = PredictController.alloc().init()


def show_predict_menu(original_text):
    """Generate rephrasings and show NSMenu. The original text has already
    been pasted by stop_and_transcribe; if the user picks a rephrasing we
    delete those chars and paste the replacement. On dismiss we leave the
    original in place."""
    set_hud(True, "predict")
    rephrasings = generate_rephrasings(original_text)
    set_hud(False)

    if not rephrasings:
        print("(no rephrasings — keeping original)", flush=True)
        return

    _predict_controller._rephrasings = rephrasings
    _predict_controller._pasted_len = len(original_text) + 1  # +1 for trailing space

    menu = AppKit.NSMenu.alloc().init()
    menu.setAutoenablesItems_(False)
    menu.setMinimumWidth_(300)
    menu.setAppearance_(
        AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark)
    )

    for i, reph in enumerate(rephrasings):
        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            reph, "pickRephrasing:", str(i + 1)
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
    global prev_fn_down, _fn_press_time, _shift_held, _option_held, note_mode, predict_mode
    global recording, cancelled, audio_chunks, _retry_count, auto_send

    # Mouse routing: HUD is an NSPanel that can't receive native mouse events;
    # we dispatch via CGEventTap so click-to-retry and pointer cursor still work.
    if event_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        _route_mouse_to_hud(event_type, event)
        return event

    keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
    flags_now = Quartz.CGEventGetFlags(event)

    prev_shift_held = _shift_held
    prev_option_held = _option_held
    _shift_held  = bool(flags_now & Quartz.kCGEventFlagMaskShift)
    _option_held = bool(flags_now & Quartz.kCGEventFlagMaskAlternate)

    # Shift TAP during recording → toggle note mode
    if recording and _shift_held and not prev_shift_held:
        if NOTES_CFG:
            note_mode = not note_mode
            predict_mode = False
            set_hud(True, "note" if note_mode else "recording")
            print(f"[toggle] note_mode={'on' if note_mode else 'off'}", flush=True)
        else:
            print("notes plugin not installed — shift+fn disabled", flush=True)

    # Option TAP during recording → toggle predict mode
    if recording and _option_held and not prev_option_held:
        predict_mode = not predict_mode
        note_mode = False
        set_hud(True, "predict" if predict_mode else "recording")
        print(f"[toggle] predict_mode={'on' if predict_mode else 'off'}", flush=True)

    # Esc → cancel
    if event_type == Quartz.kCGEventKeyDown and keycode == 53 and (recording or transcribing):
        threading.Thread(target=cancel_recording, daemon=True).start()
        return event

    # Enter during recording → toggle auto-send + undo the inserted Enter
    if keycode in (36, 76) and recording and event_type == Quartz.kCGEventKeyDown:
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
        if _shift_held and NOTES_CFG:
            note_mode    = True
            predict_mode = False
        elif _option_held:
            predict_mode = True
            note_mode    = False
        else:
            predict_mode = False
            note_mode    = False
        # Set recording state synchronously so fn-up always observes it.
        # Only the blocking mic initialization is offloaded to a thread.
        with _state_lock:
            if recording:
                should_start_mic = False
            else:
                recording    = True
                cancelled    = False
                auto_send    = False
                audio_chunks = []
                _retry_count = 0
                should_start_mic = True
        print(
            f"[mode] shift={_shift_held} option={_option_held} "
            f"→ note={note_mode} predict={predict_mode}",
            flush=True,
        )
        if should_start_mic:
            threading.Thread(target=_start_mic_stream, daemon=True).start()
            def _show_hud_delayed():
                time.sleep(0.20)
                if recording and not cancelled:
                    _show_recording_hud()
            threading.Thread(target=_show_hud_delayed, daemon=True).start()
    elif not is_down and prev_fn_down:
        held = time.time() - _fn_press_time
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
        print("ERROR: CGEventTap failed. Check Accessibility permission.", flush=True)
        sys.exit(1)

    src = CoreFoundation.CFMachPortCreateRunLoopSource(None, tap, 0)
    CoreFoundation.CFRunLoopAddSource(
        CoreFoundation.CFRunLoopGetMain(), src, CoreFoundation.kCFRunLoopCommonModes,
    )
    Quartz.CGEventTapEnable(tap, True)
    print("Hotkey monitor installed.", flush=True)
    return tap


def _tap_health_check(tap):
    """Poll CGEventTap health every 7s. Daemon thread -- exits with main process."""
    global _health_monitor_owns_hud
    tap_was_disabled = False
    while True:
        time.sleep(7)
        try:
            enabled = Quartz.CGEventTapIsEnabled(tap)
        except Exception:
            continue  # if tap reference becomes invalid, keep polling
        if not enabled and not tap_was_disabled:
            tap_was_disabled = True
            # Attempt re-enable per D-10
            try:
                Quartz.CGEventTapEnable(tap, True)
            except Exception:
                pass
            cancel_recording(skip_hud=True, quiet=True)
            set_hud(True, mode="error_fatal", tooltip=_tooltip("accessibility_revoked"))
            _health_monitor_owns_hud = True
            print("! Accessibility revoked -- attempting re-enable", flush=True)
        elif not enabled and tap_was_disabled:
            # Still disabled -- try re-enable again
            try:
                Quartz.CGEventTapEnable(tap, True)
            except Exception:
                pass
        elif enabled and tap_was_disabled:
            # Recovered! Clear error per D-11
            tap_was_disabled = False
            if _health_monitor_owns_hud and _hud_error_mode == "error_fatal":
                set_hud(False)
            _health_monitor_owns_hud = False
            print("Accessibility restored.", flush=True)
        # If enabled and was not disabled -- normal state, do nothing


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
        subprocess.run([editor, str(path)])
        return
    if choice != "r":
        return

    audio = _record_until_enter()
    if audio is None or len(audio) / SAMPLE_RATE < 0.3:
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


# ── Singleton enforcement ────────────────────────────────────────────────────
def _find_other_govori_pids():
    """Return PIDs of other running govori daemons (excluding self)."""
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "govori.py"], text=True
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
    print(f"! Govori is already running (PID {pid_list}).", flush=True)
    if not sys.stdin.isatty():
        print("  Another instance is active — refusing to start.", flush=True)
        sys.exit(1)
    try:
        ans = input("  Kill it and take over? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if ans in ("n", "no", "н", "нет"):
        print("  Aborted.", flush=True)
        sys.exit(1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as e:
            print(f"  Cannot stop PID {pid}: {e}", flush=True)
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
        print(f"  Failed to stop PID(s) {remaining}. Aborting.", flush=True)
        sys.exit(1)
    print("  Replaced previous instance.", flush=True)


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
    _ensure_singleton()
    print(f"Govori started. Model: {MODEL}. Hold fn to record.", flush=True)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    setup_hud()
    setup_predict()
    tap = install_monitor()
    threading.Thread(target=_tap_health_check, args=(tap,), daemon=True).start()

    # Microphone startup check (non-blocking per D-12, D-13)
    try:
        sd.query_devices(kind='input')
    except sd.PortAudioError:
        print("! No microphone detected. Plug one in before recording.", flush=True)

    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    run_loop = AppKit.NSRunLoop.mainRunLoop()
    while True:
        run_loop.runMode_beforeDate_(
            AppKit.NSDefaultRunLoopMode,
            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.5),
        )
