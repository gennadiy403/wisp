"""Interactive onboarding for Govori.

This module deliberately keeps print() calls: setup strings contain ANSI escape
sequences and are TTY-only, so routing them through loguru would degrade the
interactive setup experience and leak terminal codes into the daemon log.
"""

from __future__ import annotations

import json
import sys

import yaml

from .config import CONFIG_DIR, CONFIG_FILE, PLUGINS_DIR, SETUP_DONE_FILE, _load_yaml

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
    SETUP_DONE_FILE.touch()

    print(s["done"])
    sys.exit(0)


def _is_first_run():
    """Check if setup has never been run."""
    return not (CONFIG_DIR / ".setup_done").exists()

