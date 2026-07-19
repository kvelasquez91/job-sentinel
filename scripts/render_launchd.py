"""Render the launchd agent plists from their templates + config.yaml.

Substitutions: __REPO_DIR__ (repo root), __HOME__, __CLAUDE_DIR__ (directory
of the `claude` CLI), __PROFILE_KEY__ (config profile.key), and the daily
agent's StartCalendarInterval from `schedule.daily_times` (a list of 24h
"HH:MM" strings; default keeps the historical 02:30 / 13:00). Custom run
times therefore live in untracked config.yaml — never in the tracked
templates.

Writes com.jobsentinel.<key>.{daily,dashboard}.plist into --out-dir (default
~/Library/LaunchAgents), creates the per-profile log directory, and REFUSES
to overwrite a plist that belongs to a different checkout (it doesn't
mention this repo's path) — that plist's label is withheld from stdout so
the caller never bootstraps it. Prints one rendered label per line; SETUP.md
§8 pipes those into launchctl bootout/bootstrap. This script itself never
calls launchctl.
"""
import argparse
import os
import re
import shutil
import sys

import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = ("daily", "dashboard")
DEFAULT_DAILY_TIMES = ["02:30", "13:00"]
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def start_calendar_interval_xml(times) -> str:
    """Build the <array> of Hour/Minute dicts for StartCalendarInterval."""
    items = []
    for t in times:
        m = _TIME_RE.match(str(t).strip())
        if not m:
            raise ValueError(
                f"schedule.daily_times entry {t!r} is not a 24h 'HH:MM' time")
        items.append(
            "        <dict>\n"
            "            <key>Hour</key>\n"
            f"            <integer>{int(m.group(1))}</integer>\n"
            "            <key>Minute</key>\n"
            f"            <integer>{int(m.group(2))}</integer>\n"
            "        </dict>"
        )
    return "<array>\n" + "\n".join(items) + "\n    </array>"


def render(config: dict, repo_dir: str, home: str, claude_dir: str) -> dict:
    """Pure render: {label: plist-xml} for both agents from the templates."""
    profile_key = str(((config.get("profile") or {}).get("key")) or "default")
    times = ((config.get("schedule") or {}).get("daily_times")
             or DEFAULT_DAILY_TIMES)
    interval = start_calendar_interval_xml(times)

    out = {}
    for agent in AGENTS:
        template = os.path.join(repo_dir, f"com.jobsentinel.{agent}.plist.template")
        with open(template, "r", encoding="utf-8") as fh:
            text = fh.read()
        text = (text.replace("__REPO_DIR__", repo_dir)
                    .replace("__HOME__", home)
                    .replace("__CLAUDE_DIR__", claude_dir)
                    .replace("__PROFILE_KEY__", profile_key)
                    .replace("__START_CALENDAR_INTERVAL__", interval))
        out[f"com.jobsentinel.{profile_key}.{agent}"] = text
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",
                        default=os.path.expanduser("~/Library/LaunchAgents"))
    parser.add_argument("--home", default=os.path.expanduser("~"))
    parser.add_argument("--config",
                        default=os.path.join(REPO_DIR, "config.yaml"))
    parser.add_argument("--claude-dir", default=None,
                        help="dir of the claude CLI (default: from PATH)")
    args = parser.parse_args(argv)

    claude_dir = args.claude_dir
    if claude_dir is None:
        claude_path = shutil.which("claude")
        if not claude_path:
            print("claude CLI not found on PATH — install Claude Code first "
                  "(SETUP.md §3)", file=sys.stderr)
            return 1
        claude_dir = os.path.dirname(claude_path)

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        config = {}

    plists = render(config, repo_dir=REPO_DIR, home=args.home,
                    claude_dir=claude_dir)

    profile_key = str(((config.get("profile") or {}).get("key")) or "default")
    os.makedirs(os.path.join(args.home, "Library", "Logs", "job-sentinel",
                             profile_key), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for label, content in plists.items():
        dest = os.path.join(args.out_dir, f"{label}.plist")
        if os.path.exists(dest):
            with open(dest, "r", encoding="utf-8") as fh:
                existing = fh.read()
            if REPO_DIR not in existing:
                print(f"skipping {label} — {dest} belongs to a different "
                      "checkout; remove it or pick a different profile.key",
                      file=sys.stderr)
                continue
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
