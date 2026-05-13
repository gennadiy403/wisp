"""CLI routing and runtime configuration entry point."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from loguru import logger

from . import config as cfg
from .logging_setup import configure_logging
from .onboarding import _is_first_run, cli_setup


VERSION = "0.1.0"


def _usage() -> str:
    return (
        "Usage: govori [--help] [--version] [--gpt] <command>\n"
        "\n"
        "Commands:\n"
        "  setup                 run interactive setup\n"
        "  plugin <list|init|remove> ...\n"
        "  add [-p|-t] <word>... add words to vocabulary\n"
        "  notes                 open notes picker\n"
        "  note <text>           save text as a note\n"
        "\n"
        "Without a command, Govori starts the daemon."
    )


def cli_plugin(args):
    """Handle `govori plugin <subcommand>` CLI."""
    if not args:
        logger.info("Usage: govori plugin <list|add|init|remove>")
        sys.exit(1)

    sub = args[0]

    if sub == "list":
        if not cfg.PLUGINS:
            logger.info("No plugins installed.")
        for name, meta in cfg.PLUGINS.items():
            desc = meta.get("description", "")
            trigger = meta.get("trigger", "n/a")
            logger.info(f"  {name:20s} trigger={trigger:12s}  {desc}")

    elif sub == "init":
        if len(args) < 2:
            logger.info("Usage: govori plugin init <name>")
            sys.exit(1)
        name = args[1]
        dest = cfg.PLUGINS_DIR / name
        if dest.exists():
            logger.error(f"Plugin '{name}' already exists at {dest}")
            sys.exit(1)
        dest.mkdir(parents=True)
        (dest / "plugin.yaml").write_text(
            f"name: {name}\n"
            "description: My custom plugin\n"
            "trigger: shift+fn\n"
            "classifier_model: claude-haiku-4-5-20251001\n"
            "\n"
            "output_dir: ~/govori-notes/{year}/{month}\n"
            "index_file: ~/govori-notes/index/recent.jsonl\n"
            "\n"
            'whisper_prompt: ""\n',
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
        logger.info(f"Plugin scaffold created at {dest}/")
        logger.info("Edit contexts.yaml to define your contexts, then restart govori.")

    elif sub == "remove":
        if len(args) < 2:
            logger.info("Usage: govori plugin remove <name>")
            sys.exit(1)
        name = args[1]
        dest = cfg.PLUGINS_DIR / name
        if not dest.exists():
            logger.error(f"Plugin '{name}' not found.")
            sys.exit(1)
        shutil.rmtree(dest)
        logger.info(f"Removed plugin '{name}'.")

    else:
        logger.error(f"Unknown subcommand: {sub}")
        logger.info("Usage: govori plugin <list|init|remove>")
        sys.exit(1)


def cli_add(args):
    """Add words to vocabulary (terms.md/people.md), rebuild whisper_prompt, sync to VPS."""
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
            logger.info("Usage: govori add [-p|-t] <word>...")
            logger.info("  -p, --people   append to people.md (format: 'Имя — контекст')")
            logger.info("  -t, --terms    append to terms.md (default)")
            sys.exit(0)
        else:
            words.append(a)
        i += 1

    if not words:
        logger.info("Usage: govori add [-p|-t] <word>...")
        sys.exit(1)

    target_file = cfg.CONFIG_DIR / ("people.md" if target == "people" else "terms.md")
    if not target_file.exists():
        logger.error(f"Vocabulary file not found: {target_file}")
        sys.exit(1)

    def _key(s):
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
        logger.success(f"{target_file.name}: +{len(added)} ({', '.join(added)})")
    if skipped:
        logger.info(f"  skipped (already present): {', '.join(skipped)}")
    if not added:
        sys.exit(0)

    refresh = cfg.CONFIG_DIR / "refresh-prompt.sh"
    if refresh.exists():
        r = subprocess.run(["bash", str(refresh)], capture_output=True, text=True)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith(("whisper_prompt:", "language:", "preview:")):
                    logger.info(f"  {stripped}")
        else:
            logger.error(f"refresh-prompt failed: {r.stderr.strip()}")
    else:
        logger.info(f"  (refresh-prompt.sh not found at {refresh}, skipping whisper_prompt rebuild)")

    r = subprocess.run(
        [
            "rsync", "-az",
            str(cfg.CONFIG_DIR / "config.yaml"),
            str(cfg.CONFIG_DIR / "terms.md"),
            str(cfg.CONFIG_DIR / "people.md"),
            "vps:.config/govori/",
        ],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        logger.success("synced to VPS")
    else:
        logger.error(f"VPS sync failed: {r.stderr.strip()}")

    sys.exit(0)


def cli_main():
    """Route CLI subcommands before starting the daemon."""
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        logger.info(_usage())
        sys.exit(0)

    if "--version" in args or "-v" in args:
        logger.info(f"govori {VERSION}")
        sys.exit(0)

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
        from .notes_cli import cli_notes

        cli_notes(positional[1:])
        sys.exit(0)

    if positional and positional[0] == "note":
        if len(positional) < 2 and sys.stdin.isatty():
            logger.info("Usage: govori note <text>  |  echo <text> | govori note")
            sys.exit(1)
        if len(positional) >= 2:
            text = " ".join(positional[1:])
        else:
            text = sys.stdin.read()
        text = text.strip()
        if not text:
            logger.debug("Empty note")
            sys.exit(1)
        if not cfg.NOTES_CFG:
            logger.error("notes plugin not installed. Run: govori plugin init notes")
            sys.exit(1)
        from .notes import save_or_merge_note

        preview = f"{text[:120]}{'…' if len(text) > 120 else ''}"
        logger.bind(event="transcript").info(preview)
        save_or_merge_note(text, duration_sec=0)
        sys.exit(0)

    if _is_first_run():
        cli_setup()


def main():
    bench_mode = os.environ.get("BENCH_MODE") == "1"
    configure_logging(cfg.CONFIG_DIR, bench_mode=bench_mode)

    config = cfg.load_config()
    plugins = cfg.load_plugins()
    model = "gpt-4o-transcribe" if "--gpt" in sys.argv else config.model
    cfg.install_runtime_config(config, plugins, model=model)

    cli_main()

    logger.info("Govori ready.")
    if cfg.NOTES_CFG:
        n_ctx = len(cfg.NOTES_CFG["valid_contexts"])
        logger.info(f"notes plugin: {n_ctx} contexts loaded")
    else:
        logger.info("notes plugin: not installed (shift+fn disabled)")
