-- Govori status HUD — optional Hammerspoon add-on
-- Menubar indicator: ● when govori is running, ○ when stopped.
-- Context menu on the icon for start/stop/restart/logs.
--
-- Load from ~/.hammerspoon/init.lua:
--   dofile(os.getenv("HOME") .. "/Projects/govori/extras/hud/status_hud.lua")

local M = {}

-- Config --------------------------------------------------------------------
M.config = {
  process_pattern = "govori.py",
  poll_interval = 5,
  launchd_label = "com.user.govori",
  log_out = os.getenv("HOME") .. "/Library/Logs/govori.out.log",
  log_err = os.getenv("HOME") .. "/Library/Logs/govori.err.log",
}

-- Internal state ------------------------------------------------------------
local statusTimer, menubar = nil, nil
local running = false

local function isRunning()
  local out, ok = hs.execute("/usr/bin/pgrep -f " .. M.config.process_pattern, false)
  return ok and out and #out > 0
end

local function pidInfo()
  local out = hs.execute("/usr/bin/pgrep -f " .. M.config.process_pattern, false) or ""
  return (out:gsub("%s+$", ""))
end

-- launchd controls ----------------------------------------------------------
local function launchctl(action)
  local label = M.config.launchd_label
  local plist = os.getenv("HOME") .. "/Library/LaunchAgents/" .. label .. ".plist"
  local cmd
  if action == "start" then
    cmd = string.format("/bin/launchctl load -w %q 2>&1; /bin/launchctl kickstart -k gui/$(id -u)/%s 2>&1", plist, label)
  elseif action == "stop" then
    cmd = string.format("/bin/launchctl unload -w %q 2>&1", plist)
  elseif action == "restart" then
    cmd = string.format("/bin/launchctl kickstart -k gui/$(id -u)/%s 2>&1", label)
  end
  if not cmd then return end
  local out = hs.execute(cmd, true) or ""
  hs.alert.closeAll()
  hs.alert.show("govori " .. action .. (out ~= "" and (": " .. out:sub(1,80)) or ""))
end

local function tailLogs()
  hs.execute(string.format(
    [[/usr/bin/osascript -e 'tell app "Terminal" to do script "tail -n 100 -f %s %s"' -e 'tell app "Terminal" to activate']],
    M.config.log_out, M.config.log_err), true)
end

-- Menubar -------------------------------------------------------------------
local function refreshMenubar()
  if not menubar then return end
  menubar:setTitle(running and "●" or "○")
  menubar:setTooltip("Govori: " .. (running and ("running (pid " .. pidInfo() .. ")") or "stopped"))
  menubar:setMenu(function()
    local pid = pidInfo()
    return {
      { title = running and ("Govori running (pid " .. pid .. ")") or "Govori stopped", disabled = true },
      { title = "-" },
      { title = "Start",   fn = function() launchctl("start") end },
      { title = "Restart", fn = function() launchctl("restart") end },
      { title = "Stop",    fn = function() launchctl("stop") end },
      { title = "-" },
      { title = "Show logs", fn = tailLogs },
    }
  end)
end

local function statusTick()
  running = isRunning() and true or false
  refreshMenubar()
end

function M.start()
  if statusTimer then statusTimer:stop() end
  if menubar then menubar:delete() end
  menubar = hs.menubar.new()
  running = isRunning() and true or false
  refreshMenubar()
  statusTimer = hs.timer.doEvery(M.config.poll_interval, statusTick)
  hs.alert.show("Govori HUD loaded (" .. (running and "running" or "stopped") .. ")")
end

M.start()
return M
