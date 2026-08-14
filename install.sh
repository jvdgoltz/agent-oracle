#!/usr/bin/env bash
# Install Agent Oracle backend and frontend as launchd services.
#
# Both services start at login, restart on crash, and run with --reload so
# code changes are immediately visible. The script is idempotent: running it
# again unloads existing agents and reloads them fresh.
#
# Backend:  uvicorn with --reload  ->  http://localhost:8731
# Frontend: vite dev server        ->  http://localhost:8732
set -euo pipefail

LABEL_BACKEND="com.agent-oracle.backend"
LABEL_FRONTEND="com.agent-oracle.frontend"
LABEL_BACKUP="com.agent-oracle.backup"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.agent-oracle/logs"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve absolute paths to the runtime binaries so launchd can find them
# without a shell PATH.
UV_BIN="$(command -v uv || true)"
NODE_BIN="$(command -v node || true)"
NPM_BIN="$(command -v npm || true)"

if [[ -z "$UV_BIN" ]]; then
    echo "Error: 'uv' not found on PATH." >&2
    exit 1
fi
if [[ -z "$NODE_BIN" || -z "$NPM_BIN" ]]; then
    echo "Error: 'node' and 'npm' must be on PATH." >&2
    exit 1
fi

mkdir -p "$LAUNCH_DIR" "$LOG_DIR"

# Remove any existing agents so this script is safe to re-run.
for label in "$LABEL_BACKEND" "$LABEL_FRONTEND" "$LABEL_BACKUP"; do
    if launchctl list "$label" &>/dev/null; then
        echo "Unloading existing $label ..."
        launchctl unload "$LAUNCH_DIR/$label.plist" 2>/dev/null || true
    fi
done

#
# --- Backend plist --------------------------------------------------------
#
cat >"$LAUNCH_DIR/$LABEL_BACKEND.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL_BACKEND</string>
    <key>ProgramArguments</key>
    <array>
        <string>$UV_BIN</string>
        <string>run</string>
        <string>uvicorn</string>
        <string>agent_oracle.main:app</string>
        <string>--reload</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8731</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/backend.out.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/backend.err.log</string>
</dict>
</plist>
PLIST

#
# --- Frontend plist -------------------------------------------------------
#
cat >"$LAUNCH_DIR/$LABEL_FRONTEND.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL_FRONTEND</string>
    <key>ProgramArguments</key>
    <array>
        <string>$NODE_BIN</string>
        <string>$NPM_BIN/../lib/node_modules/npm/bin/npm-cli.js</string>
        <string>run</string>
        <string>dev</string>
        <string>--</string>
        <string>--port</string>
        <string>8732</string>
        <string>--host</string>
        <string>127.0.0.1</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR/frontend</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/frontend.out.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/frontend.err.log</string>
</dict>
</plist>
PLIST

#
# --- Backup plist ----------------------------------------------------------
#
# Runs the database backup script at 12:00 and 18:00 every day.
#
cat >"$LAUNCH_DIR/$LABEL_BACKUP.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL_BACKUP</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$PROJECT_DIR/scripts/backup-db.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <dict>
            <key>Hour</key>
            <integer>12</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>18</integer>
        </dict>
    </array>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/backup.out.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/backup.err.log</string>
</dict>
</plist>
PLIST

#
# --- Load and start both services -----------------------------------------
#
echo "Loading $LABEL_BACKEND ..."
launchctl load "$LAUNCH_DIR/$LABEL_BACKEND.plist"

echo "Loading $LABEL_FRONTEND ..."
launchctl load "$LAUNCH_DIR/$LABEL_FRONTEND.plist"

echo "Loading $LABEL_BACKUP ..."
launchctl load "$LAUNCH_DIR/$LABEL_BACKUP.plist"

echo ""
echo "Agent Oracle services installed and started."
echo "  Backend:  http://localhost:8731  (logs: $LOG_DIR/backend.{out,err}.log)"
echo "  Frontend: http://localhost:8732  (logs: $LOG_DIR/frontend.{out,err}.log)"
echo "  Backup:   12:00 and 18:00 daily (logs: $LOG_DIR/backup.{out,err}.log)"
echo ""
echo "Both services will start at login and restart on crash."
echo "Manage them with:"
echo "  launchctl list | grep agent-oracle"
echo "  launchctl unload $LAUNCH_DIR/$LABEL_BACKEND.plist"
echo "  launchctl load   $LAUNCH_DIR/$LABEL_BACKEND.plist"
