"""Quick command handler for BB3K Telegram bot.

Intercepts !-prefixed messages and handles them locally without invoking Claude.
These are fast local actions that bypass the Claude Code overhead.

Commands:
    !note <text>     - Append to scratchpad
    !task <text>     - Create Asana task
    !research <topic> - Queue research topic
    !status          - VPS health dashboard
    !help            - Show available commands
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

BB3K_DIR = Path("/home/bb3k/bb3k")
SCRIPTS_DIR = BB3K_DIR / "scripts"
# Use system Python for BB3K scripts — they depend on system-installed packages
# (requests, etc.) not available in the Poetry virtualenv
SYSTEM_PYTHON = "/usr/bin/python3"


def _run_script(args: list, timeout: int = 15) -> tuple[bool, str]:
    """Run a BB3K script and return (success, output)."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BB3K_DIR),
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr[:200] if result.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)[:200]


def handle_note(text: str) -> str:
    """Append a note to scratchpad."""
    if not text:
        return "Usage: !note <text>"
    ok, output = _run_script(
        [SYSTEM_PYTHON, str(SCRIPTS_DIR / "obsidian_write.py"), "scratchpad", text]
    )
    return "Added to scratchpad." if ok else f"Failed: {output}"


def handle_task(text: str) -> str:
    """Create an Asana task."""
    if not text:
        return "Usage: !task <description>"
    ok, output = _run_script(
        [SYSTEM_PYTHON, str(SCRIPTS_DIR / "asana_api.py"), "create", text]
    )
    return f"Task created: {text}" if ok else f"Failed: {output}"


def handle_research(text: str) -> str:
    """Queue a research topic."""
    if not text:
        return "Usage: !research <topic>"
    ok, output = _run_script(
        [
            SYSTEM_PYTHON,
            str(SCRIPTS_DIR / "process_research_queue.py"),
            "--add",
            text,
            "--add-source",
            "telegram_command",
        ],
        timeout=10,
    )
    if ok:
        # Kick off processing in background
        subprocess.Popen(
            [
                SYSTEM_PYTHON,
                str(SCRIPTS_DIR / "process_research_queue.py"),
                "--max",
                "1",
            ],
            cwd=str(BB3K_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            data = json.loads(output)
            pending = data.get("pending_count", "?")
            return f"Queued for research ({pending} pending)."
        except Exception:
            return "Queued for research."
    return f"Failed: {output}"


def handle_status() -> str:
    """Show VPS status."""
    ok, output = _run_script(
        [SYSTEM_PYTHON, str(SCRIPTS_DIR / "bb3k_vps_status.py")],
        timeout=15,
    )
    return output if ok else f"Status check failed: {output}"


def handle_help() -> str:
    """Show available commands."""
    return (
        "BB3K Telegram Commands\n\n"
        "!note <text> \u2014 Add to scratchpad\n"
        "!task <text> \u2014 Create Asana task\n"
        "!research <topic> \u2014 Queue research\n"
        "!status \u2014 VPS health dashboard\n"
        "!help \u2014 This message\n\n"
        "Anything else goes to Claude Code."
    )


# Command routing table
_HANDLERS = {
    "!note": lambda args: handle_note(args),
    "!task": lambda args: handle_task(args),
    "!research": lambda args: handle_research(args),
    "!status": lambda _: handle_status(),
    "!help": lambda _: handle_help(),
}


MAX_INPUT_LENGTH = 2000  # Cap input to prevent abuse


def route_command(text: str) -> Optional[str]:
    """Route a !command. Returns response string or None if not a command."""
    if not text.startswith("!"):
        return None

    parts = text.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    if len(args) > MAX_INPUT_LENGTH:
        return f"Input too long ({len(args)} chars). Max {MAX_INPUT_LENGTH}."

    handler = _HANDLERS.get(cmd)
    if handler:
        logger.info("Quick command", command=cmd, args_length=len(args))
        return handler(args)

    return None  # Not a known command, let Claude handle it
