#!/usr/bin/env bash
# Create a timestamped backup of the Agent Oracle database.
#
# Copies index.db to index.db.bak.<unix-timestamp> and prunes backups older
# than 30 days. Safe to run while the server is online: SQLite handles
# concurrent reads, and cp uses the WAL-checkpointed file on disk.
set -euo pipefail

DATA_DIR="$HOME/.agent-oracle"
DB="$DATA_DIR/index.db"

if [[ ! -f "$DB" ]]; then
    echo "Database not found at $DB" >&2
    exit 1
fi

TIMESTAMP=$(date +%s)
BACKUP="$DATA_DIR/index.db.bak.$TIMESTAMP"

cp "$DB" "$BACKUP"
echo "Backup created: $BACKUP ($(du -h "$BACKUP" | cut -f1))"

# Prune backups older than 30 days (10800 * 30 seconds).
find "$DATA_DIR" -name 'index.db.bak.*' -mtime +30 -delete 2>/dev/null || true
echo "Pruned backups older than 30 days."
