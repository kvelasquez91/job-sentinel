import json
import os
import sys
import time
import threading
from unittest import mock

import pytest

import claude_cli

PY = sys.executable


# ---------------------------------------------------------------------------
# run_claude() unit tests — mock the guarded runner, verify parsing + key strip
# ---------------------------------------------------------------------------

def _envelope(result="ok", is_error=False, cost=0.0):
    return json.dumps({
        "is_error": is_error,
        "result": result,
        "total_cost_usd": cost,
        "session_id": "sid",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }).encode()


def test_run_claude_parses_success_envelope():
    with mock.patch.object(claude_cli, "_run_guarded",
                           return_value=(0, _envelope('{"fit_score": 88}', cost=0.012), b"")) as m:
        out = claude_cli.run_claude("score this", model="claude-sonnet-4-6")
    assert out["text"] == '{"fit_score": 88}'
    assert out["cost_usd"] == 0.012
    assert out["is_error"] is False
    cmd = m.call_args.args[0]
    assert "--model" in cmd and "claude-sonnet-4-6" in cmd
    assert "--output-format" in cmd and "json" in cmd


def test_run_claude_strips_api_key_from_subprocess_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    # A stray base URL (e.g. from a Claude Desktop shell) forces the CLI to the
    # pay-per-use API where the subscription OAuth token is invalid -> 401. Strip
    # it so run_claude always routes to the subscription endpoint.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    with mock.patch.object(claude_cli, "_run_guarded", return_value=(0, _envelope(), b"")) as m:
        claude_cli.run_claude("hi", model="claude-sonnet-4-6")
    # _run_guarded(cmd, input_bytes, env, cwd, timeout, max_rss_mb)
    env_arg = m.call_args.args[2]
    assert "ANTHROPIC_API_KEY" not in env_arg
    assert "ANTHROPIC_AUTH_TOKEN" not in env_arg
    assert "ANTHROPIC_BASE_URL" not in env_arg


def test_run_claude_raises_on_is_error():
    with mock.patch.object(claude_cli, "_run_guarded",
                           return_value=(1, _envelope("bad model", is_error=True), b"")):
        with pytest.raises(claude_cli.ClaudeCLIError):
            claude_cli.run_claude("hi", model="nope")


def test_run_claude_raises_on_empty_output():
    with mock.patch.object(claude_cli, "_run_guarded", return_value=(0, b"", b"boom")):
        with pytest.raises(claude_cli.ClaudeCLIError):
            claude_cli.run_claude("hi", model="claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# _run_guarded() safety tests — real subprocesses (bounded + self-capped)
# ---------------------------------------------------------------------------

def test_guarded_normal_passthrough():
    rc, out, err = claude_cli._run_guarded(
        [PY, "-c", "import sys; sys.stdout.write('hello')"],
        b"", os.environ.copy(), None, timeout=10, max_rss_mb=1500,
    )
    assert rc == 0
    assert out == b"hello"


def test_guarded_kills_on_timeout():
    t0 = time.monotonic()
    with pytest.raises(claude_cli.ClaudeCLIError) as ei:
        claude_cli._run_guarded(
            [PY, "-c", "import time; time.sleep(30)"],
            b"", os.environ.copy(), None, timeout=1, max_rss_mb=1500,
        )
    assert "timed out" in str(ei.value).lower()
    # Timeouts are a distinct subtype so callers can choose not to retry them.
    assert isinstance(ei.value, claude_cli.ClaudeCLITimeout)
    assert time.monotonic() - t0 < 10  # killed promptly, not after 30s


def test_guarded_kills_on_memory_ceiling():
    # Child grows ~50MB/0.1s but SELF-CAPS at 30 iters (<=1.5GB) so even a broken watchdog
    # can never exhaust the machine. With the watchdog working it dies in ~1s at >250MB.
    hog = (
        "import time\n"
        "x=[]\n"
        "for _ in range(30):\n"
        "    x.append(bytearray(50*1024*1024))\n"
        "    time.sleep(0.1)\n"
    )
    t0 = time.monotonic()
    with pytest.raises(claude_cli.ClaudeCLIError) as ei:
        claude_cli._run_guarded(
            [PY, "-c", hog], b"", os.environ.copy(), None, timeout=20, max_rss_mb=250,
        )
    assert "memory" in str(ei.value).lower()
    # A memory kill is a ClaudeCLIError but NOT a timeout — callers still retry it.
    assert not isinstance(ei.value, claude_cli.ClaudeCLITimeout)
    assert time.monotonic() - t0 < 15


def test_guarded_serializes_concurrency(monkeypatch):
    # The global semaphore must prevent more guarded runs than it has slots.
    # Pin a 1-slot semaphore (the production default is larger) so the test
    # exercises the mechanism: two 0.6s sleeps forced one-after-another take >= ~1.2s.
    monkeypatch.setattr(claude_cli, "_CLAUDE_SEMAPHORE", threading.BoundedSemaphore(1))

    def call():
        claude_cli._run_guarded(
            [PY, "-c", "import time; time.sleep(0.6)"],
            b"", os.environ.copy(), None, timeout=10, max_rss_mb=1500,
        )

    t0 = time.monotonic()
    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.monotonic() - t0 >= 1.1
