"""Tests for the launchd installer and repository MCP client configuration."""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
INSTALL_SCRIPT = PROJECT_DIR / "install.sh"
MCP_URL = "http://127.0.0.1:8731/mcp/"


def _link_command(bin_dir: Path, name: str) -> None:
    """Expose one system command in an otherwise isolated test PATH."""
    target = shutil.which(name)
    assert target is not None
    (bin_dir / name).symlink_to(target)


def _write_executable(path: Path, content: str) -> None:
    """Write an executable test command at *path*."""
    path.write_text(content)
    path.chmod(0o755)


def _run_installer(tmp_path: Path, *, include_uv: bool) -> subprocess.CompletedProcess[str]:
    """Run the installer with fake launchd commands and optional uv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("cat", "dirname", "mkdir"):
        _link_command(bin_dir, name)

    launchctl_log = tmp_path / "launchctl.log"
    _write_executable(
        bin_dir / "launchctl",
        """#!/bin/bash
echo "$*" >>"$LAUNCHCTL_LOG"
[[ "$1" != "list" ]]
""",
    )
    if include_uv:
        _write_executable(bin_dir / "uv", "#!/bin/bash\nexit 0\n")

    env = {
        "HOME": str(tmp_path / "home"),
        "LAUNCHCTL_LOG": str(launchctl_log),
        "PATH": str(bin_dir),
    }
    return subprocess.run(
        ["/bin/bash", str(INSTALL_SCRIPT)],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installer_starts_backend_without_optional_frontend_runtime(tmp_path: Path) -> None:
    """Missing Node.js and npm skip only the optional frontend service."""
    result = _run_installer(tmp_path, include_uv=True)

    assert result.returncode == 0
    assert (
        "Warning: frontend not installed; 'node' and 'npm' are not both on PATH." in result.stderr
    )
    assert f"MCP:      {MCP_URL}" in result.stdout
    assert "Reconnect or restart open MCP clients to refresh Agent Oracle tools." in result.stdout

    launch_dir = tmp_path / "home" / "Library" / "LaunchAgents"
    assert (launch_dir / "com.agent-oracle.backend.plist").is_file()
    assert (launch_dir / "com.agent-oracle.backup.plist").is_file()
    assert not (launch_dir / "com.agent-oracle.frontend.plist").exists()

    launchctl_calls = (tmp_path / "launchctl.log").read_text()
    assert "load " in launchctl_calls
    assert "com.agent-oracle.backend.plist" in launchctl_calls
    assert "com.agent-oracle.backup.plist" in launchctl_calls
    assert "load " + str(launch_dir / "com.agent-oracle.frontend.plist") not in launchctl_calls


def test_installer_fails_before_changes_when_uv_is_missing(tmp_path: Path) -> None:
    """The mandatory backend package runner remains an explicit prerequisite."""
    result = _run_installer(tmp_path, include_uv=False)

    assert result.returncode == 1
    assert result.stderr == "Error: 'uv' not found on PATH.\n"
    assert not (tmp_path / "home" / "Library" / "LaunchAgents").exists()
    assert not (tmp_path / "launchctl.log").exists()


def test_repository_mcp_configs_use_the_bound_backend_address() -> None:
    """All supported clients use the exact loopback MCP endpoint."""
    claude = json.loads((PROJECT_DIR / ".mcp.json").read_text())
    droid = json.loads((PROJECT_DIR / ".factory" / "mcp.json").read_text())
    codex = tomllib.loads((PROJECT_DIR / ".codex" / "config.toml").read_text())

    assert claude["mcpServers"]["agent-oracle"]["url"] == MCP_URL
    assert droid["mcpServers"]["agent-oracle"]["url"] == MCP_URL
    assert codex["mcp_servers"]["agent-oracle"]["url"] == MCP_URL
