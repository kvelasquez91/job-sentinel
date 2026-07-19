"""
claude_cli.py — thin, resource-guarded wrapper around the local Claude Code CLI
(`claude`), used as a headless, single-shot text-completion endpoint.

Every Claude call in Job Sentinel goes through run_claude(). It shells out to the
locally-installed `claude` CLI in print mode (-p), which bills against this machine's
Claude subscription rather than the Anthropic API.

CRITICAL: the CLI uses ANTHROPIC_API_KEY (pay-per-token) whenever it is present in the
environment, and ANTHROPIC_BASE_URL forces it to the first-party API (where the
subscription OAuth token is rejected with a 401). To force subscription billing we strip
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, and ANTHROPIC_BASE_URL from the subprocess
environment before spawning `claude`.

SAFETY — added after a batch-scoring run spawned 4 concurrent `claude` processes that
each ballooned to ~18 GB (hung + un-reaped) and OOM-crashed a 16 GB machine. A normal
call uses ~260 MB and finishes in ~3 s; these guards bound the worst case:
  * Global concurrency cap (default 1): a process-wide semaphore ensures the WHOLE
    application (scorer, resume tailor, future scrapers) never runs more than
    CLAUDE_CLI_MAX_CONCURRENCY `claude` processes at once, regardless of caller/threads.
  * Per-process memory ceiling (default 1500 MB): a watchdog samples each child's RSS and
    kills it if it exceeds CLAUDE_CLI_MAX_RSS_MB.
  * Guaranteed cleanup: each child runs in its own process group and is force-killed +
    reaped on timeout, over-memory, error, or normal exit. No orphaned `claude` process
    can outlive a call.

run_claude() is a single-attempt primitive — it does not retry. Callers apply their own
retry/fallback policy (the scorer and tailor both already have one).
"""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Optional, Tuple

CLAUDE_BIN = "claude"

# Neutral working dir so the CLI does not auto-discover this project's .claude/ settings,
# hooks, or CLAUDE.md on every call (faster, isolated). With --tools "" the call never
# needs filesystem access anyway.
_NEUTRAL_CWD = tempfile.gettempdir()

# Replaces the CLI's default (coding-agent) system prompt so it does not bias the task.
_DEFAULT_SYSTEM = (
    "You are an expert assistant. Follow the user's instructions precisely "
    "and respond exactly in the requested format."
)

# --- Resource guards (overridable via environment) -------------------------------------
# Max concurrent `claude` processes across the ENTIRE process (all threads/callers).
# Must be >= llm_scoring.workers (config.yaml) or scoring threads queue here instead
# of running in parallel. 4 × ~260 MB ≈ 1 GB steady-state (worst case 4 × 1.5 GB
# watchdog ceiling = 6 GB) — sized for the 16 GB machine.
MAX_CONCURRENCY = max(1, int(os.environ.get("CLAUDE_CLI_MAX_CONCURRENCY", "4")))
# Kill any `claude` process whose resident memory exceeds this (MB). Normal call ~260 MB.
MAX_RSS_MB = float(os.environ.get("CLAUDE_CLI_MAX_RSS_MB", "1500"))
# How often the watchdog samples memory / checks the timeout (seconds).
_POLL_SECONDS = 0.5

_CLAUDE_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENCY)


class ClaudeCLIError(Exception):
    """Raised on non-zero exit, is_error result, timeout, memory kill, missing binary, or bad JSON."""


class ClaudeCLITimeout(ClaudeCLIError):
    """Raised specifically when the call is killed for exceeding its wall-clock timeout.

    A distinct subtype so callers can treat a timeout differently from a transient
    error: for a fixed prompt the latency is deterministic, so retrying at the same
    ceiling just wastes the subscription window and blocks the global semaphore.
    """


def is_cli_available() -> bool:
    """Return True if the `claude` binary is on PATH."""
    return shutil.which(CLAUDE_BIN) is not None


def _build_env() -> dict:
    """Copy the environment with Anthropic API credentials removed (force subscription auth).

    Strips ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN (pay-per-token credentials) and
    ANTHROPIC_BASE_URL. A base URL pointing at the raw API (e.g. inherited from a
    Claude Desktop shell) routes the CLI to first-party API auth, where the
    subscription OAuth token is rejected with a 401 — so it must be dropped to keep
    every call on the local subscription.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    env.pop("ANTHROPIC_BASE_URL", None)
    return env


def _proc_rss_mb(pid: int) -> float:
    """Resident set size of a pid in MB (0.0 if the process is gone or unreadable)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], stderr=subprocess.DEVNULL
        ).strip()
        return (int(out) / 1024.0) if out else 0.0
    except Exception:
        return 0.0


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Force-kill the child's whole process group (covers any helper children it spawned)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_guarded(
    cmd: list,
    input_bytes: bytes,
    env: dict,
    cwd: Optional[str],
    timeout: float,
    max_rss_mb: float,
) -> Tuple[int, bytes, bytes]:
    """
    Spawn `cmd`, feed `input_bytes` on stdin, and return (returncode, stdout, stderr).

    Enforces the safety guarantees from the module docstring:
      * acquires the global concurrency semaphore (serializes claude processes),
      * runs the child in its own process group (start_new_session=True),
      * kills the whole group if it exceeds `timeout` or `max_rss_mb`,
      * always reaps the child before returning (no orphans).

    Raises ClaudeCLIError on missing binary, timeout, or memory-ceiling kill.
    """
    _CLAUDE_SEMAPHORE.acquire()
    proc: Optional[subprocess.Popen] = None
    try:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError("claude CLI not found on PATH") from exc

        # communicate() writes stdin, reads stdout/stderr, and reaps the child. It runs
        # on a worker thread so the main thread can watch memory and the wall clock.
        io_result: dict = {}

        def _pump() -> None:
            try:
                out, err = proc.communicate(input=input_bytes)
                io_result["out"] = out
                io_result["err"] = err
            except Exception as exc:  # pragma: no cover - defensive
                io_result["exc"] = exc

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()

        start = time.monotonic()
        kill_reason: Optional[str] = None
        timed_out = False
        while True:
            pump.join(_POLL_SECONDS)
            if not pump.is_alive():
                break  # finished normally
            if (time.monotonic() - start) > timeout:
                kill_reason = f"timed out after {timeout:.0f}s"
                timed_out = True
                break
            rss = _proc_rss_mb(proc.pid)
            if rss > max_rss_mb:
                kill_reason = f"exceeded memory ceiling ({rss:.0f}MB > {max_rss_mb:.0f}MB)"
                break

        if kill_reason is not None:
            _kill_process_group(proc)
            pump.join(timeout=5)  # let communicate() unblock and reap the zombie
            exc_cls = ClaudeCLITimeout if timed_out else ClaudeCLIError
            raise exc_cls(f"claude CLI killed: {kill_reason}")

        pump.join()
        returncode = proc.returncode if proc.returncode is not None else proc.wait()
        return returncode, io_result.get("out") or b"", io_result.get("err") or b""
    finally:
        # Belt-and-suspenders: never let a `claude` process outlive this call.
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
        _CLAUDE_SEMAPHORE.release()


def run_claude(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    timeout: float = 60.0,
) -> dict:
    """
    Run a single headless, resource-guarded `claude -p` completion and return parsed results.

    Args:
        prompt:        User prompt (piped via stdin — safe for large/multi-line text).
        model:         Model id, e.g. "claude-sonnet-4-6" or "claude-opus-4-8".
        system_prompt: Replaces the CLI's default system prompt. Defaults to a neutral
                       instruction.
        timeout:       Seconds before the subprocess is killed.

    Returns:
        {"text": str, "usage": dict, "cost_usd": float, "is_error": bool,
         "session_id": Optional[str]}

    Raises:
        ClaudeCLIError on missing binary, timeout, memory kill, non-zero exit, is_error
        result, or unparseable output.
    """
    if system_prompt is None:
        system_prompt = _DEFAULT_SYSTEM

    cmd = [
        CLAUDE_BIN,
        "-p",
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system_prompt,
        "--tools", "",                  # disable all tools — pure text completion
        "--permission-mode", "dontAsk",  # never block on a permission prompt
        "--no-session-persistence",      # don't write session files for batch calls
        "--strict-mcp-config",           # don't load project MCP servers
    ]

    returncode, out_bytes, err_bytes = _run_guarded(
        cmd, prompt.encode("utf-8"), _build_env(), _NEUTRAL_CWD, timeout, MAX_RSS_MB
    )

    stdout = out_bytes.decode("utf-8", errors="replace").strip()
    stderr = err_bytes.decode("utf-8", errors="replace").strip()

    if not stdout:
        raise ClaudeCLIError(
            f"claude CLI produced no output (exit={returncode}): {stderr[:300]}"
        )

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCLIError(f"claude CLI returned non-JSON output: {stdout[:300]}") from exc

    if returncode != 0 or data.get("is_error"):
        raise ClaudeCLIError(
            f"claude CLI error (exit={returncode}): {str(data.get('result'))[:300]}"
        )

    return {
        "text": data.get("result", ""),
        "usage": data.get("usage", {}) or {},
        "cost_usd": float(data.get("total_cost_usd", 0.0) or 0.0),
        "is_error": bool(data.get("is_error", False)),
        "session_id": data.get("session_id"),
    }
