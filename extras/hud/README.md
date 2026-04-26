# Govori Status HUD (optional)

Tiny always-on-top indicator showing whether govori is running. Optional — govori works fine without it.

## Install

1. Install Hammerspoon: `brew install --cask hammerspoon`, launch it, grant Accessibility permission.
2. Add one line to `~/.hammerspoon/init.lua` (create if missing):

   ```lua
   dofile(os.getenv("HOME") .. "/Projects/govori/extras/hud/status_hud.lua")
   ```

3. Hammerspoon menu → *Reload Config*.

## Usage

- Green dot in the bottom-left corner = govori is running.
- `⌥⇧W` — toggle HUD on/off.
- `⌥⇧E` — cycle between `minimal` (few pixels) and `visible` (12px semi-transparent).

## Tweak

Edit defaults at the top of `status_hud.lua`: `mode`, `poll_interval`, colors, sizes, hotkeys, process pattern.

## Uninstall

Remove the `dofile` line from `~/.hammerspoon/init.lua` and reload.
