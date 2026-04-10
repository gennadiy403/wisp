# Wisp

Voice dictation for macOS. Hold **fn** to record, release to transcribe and paste.

No menu bar icon. No electron. Just a 32px floating dot that pulses while you talk.

## Modes

| Shortcut | Mode | What happens |
|----------|------|-------------|
| **fn** (hold) | Dictate | Speech → text → paste at cursor |
| **Option + fn** | Predict | Speech → text → 3 autocomplete suggestions |
| **Shift + fn** | Note | Speech → classify → save as markdown note |

**During recording:**
- **Enter** — toggle auto-send (press Enter after paste)
- **Shift** — toggle note mode on/off
- **Esc** — cancel

## Requirements

- macOS (uses native Cocoa APIs for HUD, hotkeys, clipboard)
- Accessibility permission (System Settings → Privacy → Accessibility)
- OpenAI API key (for Whisper transcription)
- Anthropic API key (optional, for note classification)

## Install

```bash
git clone https://github.com/YOUR_USERNAME/wisp.git
cd wisp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `~/.config/wisp/env` with your API keys:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...  # optional, for notes plugin
```

## Run

```bash
./wisp
```

Or with GPT-4o transcription model:

```bash
./wisp --gpt
```

## Configuration

Base config lives at `~/.config/wisp/config.yaml`:

```yaml
language: ru          # Whisper language
model: whisper-1      # or gpt-4o-transcribe
sample_rate: 16000
whisper_prompt: ""    # extra vocab to seed Whisper with
```

## Plugins

Wisp supports plugins for extending what happens after transcription. Plugins live in `~/.config/wisp/plugins/<name>/`.

### Notes plugin

The notes plugin classifies your voice notes using Claude and saves them as structured markdown files.

**Quick setup:**

```bash
# Scaffold a new notes plugin with default contexts
wisp plugin init notes

# Edit your contexts
nano ~/.config/wisp/plugins/notes/contexts.yaml
```

Or copy the example:

```bash
cp -r examples/notes ~/.config/wisp/plugins/notes
```

**Plugin structure:**

```
~/.config/wisp/plugins/notes/
├── plugin.yaml       # plugin settings (output dir, classifier model)
├── contexts.yaml     # your life/work contexts for classification
└── stuck.yaml        # optional: ongoing tasks to link notes to
```

**contexts.yaml** defines your personal contexts:

```yaml
- key: work
  description: My day job at Acme Corp

- key: side_project
  description: My SaaS app — features, bugs, launches

- key: personal
  description: Family, health, finance, hobbies
```

The classifier uses these descriptions to automatically tag each voice note with the right context.

**Output format:**

```markdown
---
id: 2026-04-10_1430_deploy-fix-idea
created: 2026-04-10T14:30:00+05:00
source: voice
duration_sec: 12
contexts: ["work"]
type: idea
urgency: medium
tags: ["deploy", "ci"]
related_stuck: []
---

Maybe we should add a canary step before the full rollout...
```

### Managing plugins

```bash
wisp plugin list              # show installed plugins
wisp plugin init my-plugin    # scaffold a new plugin
wisp plugin remove my-plugin  # remove a plugin
```

## How it works

1. Listens for fn key via `CGEventTap` (requires Accessibility permission)
2. Records audio through `sounddevice` at 16kHz mono
3. Encodes to OGG/Opus via `PyAV`
4. Sends to OpenAI Whisper API for transcription
5. Pastes result at cursor position via `Cmd+V` (restores clipboard after)
6. Optional: classifies note via Claude Haiku and saves as markdown

## License

MIT
