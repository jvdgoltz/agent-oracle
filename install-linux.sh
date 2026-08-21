#!/usr/bin/env bash
# Install Agent Oracle as reload-enabled systemd user services on Linux.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
LOG_DIR="$HOME/.agent-oracle/logs"
UV_BIN="$(command -v uv || true)"
NODE_BIN="$(command -v node || true)"
NPM_BIN="$(command -v npm || true)"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: install-linux.sh only supports Linux." >&2
    exit 1
fi
if ! command -v systemctl >/dev/null; then
    echo "Error: systemd is required." >&2
    exit 1
fi
if [[ -z "$UV_BIN" ]]; then
    echo "Error: 'uv' not found on PATH." >&2
    exit 1
fi
if [[ -z "$NODE_BIN" || -z "$NPM_BIN" ]]; then
    echo "Error: 'node' and 'npm' must be on PATH." >&2
    exit 1
fi
if [[ ! -d "$PROJECT_DIR/frontend/node_modules" ]]; then
    echo "Error: frontend dependencies are missing; run 'npm ci' in frontend first." >&2
    exit 1
fi

mkdir -p "$UNIT_DIR" "$LOG_DIR"

cat >"$UNIT_DIR/agent-oracle-backend.service" <<UNIT
[Unit]
Description=Agent Oracle backend
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$UV_BIN run uvicorn agent_oracle.main:app --reload --host 127.0.0.1 --port 8731
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
UNIT

cat >"$UNIT_DIR/agent-oracle-frontend.service" <<UNIT
[Unit]
Description=Agent Oracle frontend
After=agent-oracle-backend.service

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR/frontend
Environment=PATH=$(dirname "$NODE_BIN"):/usr/local/bin:/usr/bin:/bin
ExecStart=$NPM_BIN run dev -- --port 8732 --host 127.0.0.1
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
UNIT

cat >"$UNIT_DIR/agent-oracle-backup.service" <<UNIT
[Unit]
Description=Back up the Agent Oracle database

[Service]
Type=oneshot
ExecStart=/bin/bash $PROJECT_DIR/scripts/backup-db.sh
UNIT

cat >"$UNIT_DIR/agent-oracle-backup.timer" <<'UNIT'
[Unit]
Description=Back up the Agent Oracle database twice daily

[Timer]
OnCalendar=*-*-* 12,18:00:00
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now agent-oracle-backend.service agent-oracle-frontend.service
systemctl --user enable --now agent-oracle-backup.timer

echo "Agent Oracle services installed and started."
echo "  Backend:  http://localhost:8731"
echo "  Frontend: http://localhost:8732"
echo "  Logs:     journalctl --user -u 'agent-oracle-*'"
echo "Agent Oracle watches Codex, Factory, Claude Code, and Oh My Pi sessions."
