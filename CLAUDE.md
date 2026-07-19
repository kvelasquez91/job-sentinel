# Job Sentinel — project instructions

Job Sentinel is a personal job-search engine: scrapers (LinkedIn guest pages,
Greenhouse/Lever/Ashby/Workday/Eightfold/SmartRecruiters/SuccessFactors, WTTJ,
HN "Who is hiring?") → deterministic keyword score → LLM fit score via the
local `claude` CLI on the owner's own subscription → blended score → local web
dashboard with application tracking and optional resume tailoring.

**First, every session:** if `config.yaml` is missing, or `llm_scoring.
resume_summary` in it is unset/commented, this clone is not personalized yet.
Offer to run the setup interview — the `jobsentinel-setup` skill in
`.claude/skills/` — before doing anything else. Setup writes ONLY untracked
files (`config.yaml`, `data/experience_inventory.md`, `.env`, launchd/cron
schedules), so `git pull` updates never conflict.

Ground rules:
- Run everything from the repo root. Tests: `python -m pytest -q` (bare
  `pytest` fails collection — there is no root conftest path hook).
- Never commit `config.yaml`, `.env`, or your personal files under `data/`
  (`experience_inventory.md`, `jobs.db`, etc.) — those are gitignored; the
  `.example` templates in `data/` stay tracked.
- All LLM work goes through the local `claude` CLI (subscription billing).
  Never introduce API-key code paths; the CLI wrapper strips `ANTHROPIC_*`.
- Policy lives in config (`policy:` section — see CUSTOMIZING.md), not code.
  Personalization must never require editing tracked files.
- To update the install: use the `jobsentinel-update` skill.
- This repo is a generated mirror — see CONTRIBUTING.md before proposing PRs.
