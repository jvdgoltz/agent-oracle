"""Tests for the Linux service installer."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
INSTALLER = PROJECT_ROOT / "install-linux.sh"


def test_linux_installer_configures_reload_and_session_watching() -> None:
    """The Linux installer starts reload servers through the app lifespan."""
    script = INSTALLER.read_text()

    assert "uvicorn agent_oracle.main:app --reload" in script
    assert "run dev -- --port 8732 --host 127.0.0.1" in script
    assert "systemctl --user enable --now" in script
    assert "Agent Oracle watches Codex, Factory, Claude Code, and Oh My Pi sessions" in script


def test_linux_installer_creates_backend_frontend_and_backup_units() -> None:
    """The Linux installer manages both apps and the scheduled database backup."""
    script = INSTALLER.read_text()

    assert "agent-oracle-backend.service" in script
    assert "agent-oracle-frontend.service" in script
    assert "agent-oracle-backup.service" in script
    assert "agent-oracle-backup.timer" in script
    assert "OnCalendar=*-*-* 12,18:00:00" in script
