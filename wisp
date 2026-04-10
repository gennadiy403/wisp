#!/bin/bash
cd "$(dirname "$0")"
# Load secrets (ANTHROPIC_API_KEY etc.) if present
[ -f "$HOME/.config/wisp/env" ] && source "$HOME/.config/wisp/env"
source .venv/bin/activate
python wisp.py "$@"
