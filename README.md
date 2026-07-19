# Job Sentinel

Automated job-search sourcing engine. It scrapes job boards on a schedule,
scores each posting with a fast keyword pass and a deeper LLM fit pass (run
through your own Claude subscription — no API key needed), and surfaces the
results in a local web dashboard for review, application tracking, and
optional automatic resume tailoring.

## Quick Start

```bash
git clone <this repository> job-sentinel && cd job-sentinel
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Then open Claude Code in the directory and say **"set me up"** — the bundled
setup skill interviews you (profession, titles, salary bars, location,
sources, schedule) and writes your configuration. Prefer doing it by hand?
See SETUP.md and CUSTOMIZING.md.

See **SETUP.md** for the full bootstrap runbook (Python environment, Claude
Code login, first run, dashboard, optional automation).

## How it works

1. **Scrape** — pluggable scrapers pull postings from LinkedIn, Greenhouse,
   Lever, Ashby, Workday, Eightfold, SmartRecruiters, SAP SuccessFactors,
   Welcome to the Jungle, and Hacker News "Who is hiring?" threads.
2. **Keyword score** (0–100) — fast, deterministic, computed at save time
   from your configured title/skill keywords, seniority signal, compensation
   data, remote confidence, and priority-company bonus.
3. **LLM score** (0–100) — a 5-dimension fit rubric run through the local
   `claude` CLI: role match, seniority, remote/location, domain fit, and
   compensation. Capped per run to protect your Claude usage window, with a
   backfill queue for anything not yet reached.
4. **Blend** — final score = **0.4 × keyword + 0.6 × LLM** (weights live in
   `config.yaml` under `llm_scoring`). Jobs the LLM hasn't scored yet keep
   their keyword-only score until the backfill drains them. Match threshold:
   **60+** (`scoring.alert_threshold` in `config.yaml`).
5. **Dashboard** — review, filter, track applications, and optionally
   auto-tailor your resume per job.

Everything above is generic machinery — the actual titles, keywords,
geography, and compensation bars are policy values you set. See
**Personalizing** below.

## Dashboard

`python main.py --dashboard` → http://127.0.0.1:8500. Beyond the score/status
filters and job cards:

- **Comp filter chips** — posted-salary-only, or configurable dollar-threshold
  tiers (posted or LLM-estimated)
- **Application tracking panel** — per-job stage (Applied → Phone Screen →
  Interview → Offer → Rejected/Ghosted), free-text notes, and an offer
  worksheet (base/bonus/equity)
- **Market comps per job** — expanding a job shows the p25–p75 and median of
  comparable postings (matching seniority tier, local vs. remote/national)
  from your own job corpus
- **📈 Market analytics panel** — weekly inflow + high-match volume, live
  comp distribution, and per-source yield, each as a bar chart
- **👍/👎 feedback** — mark a job "more/less like this" to calibrate future
  LLM scoring

## Project Structure

```
job-sentinel/
├── main.py              # Entry point, CLI, orchestration
├── config.yaml          # All configuration
├── CLAUDE.md            # Project instructions for Claude Code (setup interview, ground rules)
├── CUSTOMIZING.md       # config.yaml schema reference — every policy/profile/dashboard key
├── profile_policy.py    # Policy LOADER — all owner values live in config.yaml's policy: section
├── local_area.py        # Shared commuter-area location matching, used by the scorer + scrapers
├── salary_rules.py      # Shared salary-plausibility rules (cap + range sanitizing), used by every scraper and engine/llm_scorer.py
├── filter_match.py      # Filter Match: employer-screening survival estimate (pure functions)
├── filter_judge.py      # Filter Match v2 judge: prompts + response parsing against your experience inventory
├── requirements.txt
├── .claude/
│   └── skills/           # jobsentinel-setup (the "set me up" interview) + jobsentinel-update
├── scrapers/
│   ├── base.py          # JobPosting dataclass, BaseScraper ABC
│   ├── linkedin.py      # LinkedIn public search scraper (429 retry + circuit breaker)
│   ├── greenhouse.py    # Greenhouse + Lever + Ashby JSON API scrapers
│   ├── workday.py       # Workday CxS tenant scraper
│   ├── eightfold.py     # Eightfold career-site scraper
│   ├── smartrecruiters.py # SmartRecruiters API scraper
│   ├── successfactors.py  # SAP SuccessFactors Career Site Builder scraper
│   ├── wttj.py          # Welcome to the Jungle (Algolia) scraper
│   └── hn_whoishiring.py # Hacker News "Who is hiring?" scraper via the Algolia API
├── engine/
│   ├── scorer.py        # Keyword scoring and explanation logic
│   ├── llm_scorer.py    # LLM fit scoring via the claude CLI (blend, caps, backfill)
│   └── company_intel.py # Company health enrichment (yfinance, news, layoffs)
├── claude_cli.py        # Resource-guarded wrapper around the local claude CLI
├── alerts/
│   ├── notifier.py      # JSON run-report writer
│   └── error_monitor.py # Run-history recording + errors.log
├── dashboard/
│   ├── app.py           # FastAPI REST API
│   └── static/
│       └── index.html   # Single-page dashboard
├── resume_tailor/
│   ├── config.py        # Paths, OAuth scopes, LLM model, ATS targets
│   ├── google_api.py    # Google Docs + Drive API wrapper (OAuth 2.0)
│   ├── jd_extractor.py  # 4-tier job description extraction from URLs
│   ├── tailor_engine.py # 4-step LLM chain + Google Docs edit application
│   ├── pipeline.py      # Synchronous 10-step tailor pipeline shared by the dashboard's ✂ button and main.py's auto-tailor
│   └── ats_checker.py   # ATS compliance checks (score 0-100)
├── scripts/
│   └── setup_google_auth.py  # One-time OAuth flow; writes token.json
└── data/
    ├── jobs.db               # SQLite database (auto-created)
    ├── reports/              # JSON reports per run
    └── tailored_resumes/     # Exported .docx files
```

## Commands

| Command | Description |
|---|---|
| `python main.py` | Full run: scrape, keyword-score, save, LLM-score, enrich companies |
| `python main.py --dry-run` | Scrape and keyword-score, print results to terminal, no DB write |
| `python main.py --scrape-only` | Scrape, keyword-score, and save — skip LLM scoring and company enrichment |
| `python main.py --skip-llm` | Full run minus the LLM scoring pass (keyword-only scores) |
| `python main.py --skip-company-intel` | Full run minus company intelligence enrichment |
| `python main.py --skip-auto-tailor` | Full run minus the post-run auto-tailor pass |
| `python main.py --dashboard` | Start web dashboard on http://127.0.0.1:8500 |
| `python main.py --rescore-all` | LLM-score every job where llm_score IS NULL (resumable) |
| `python main.py --rescore-force` | Clear all LLM scores and re-score from scratch (heavy Claude usage) |
| `python main.py --enrich-companies` | Refresh company intel for every company in the DB, then exit |
| `python main.py --dismiss-job <ID-or-URL>` | Hide a job from the dashboard |
| `python main.py --log-level DEBUG` | Verbose logging |

## Personalizing

This repo ships with generic, functional defaults — no target titles, no
target cities, no keyword lists, no compensation bars, and a neutral scoring
rubric. Every one of those is a value in `config.yaml`'s `policy:` section
(plus `profile.key` and `dashboard:`); nothing about who you are or what
you're searching for lives in a tracked file. The fastest way to fill it in
is the interview: open Claude Code in the repo and say **"set me up"** — it
asks about your profession, titles, salary bars, location, sources, and
schedule, then writes `config.yaml`, your experience inventory, `.env`, and
your run schedule for you. Everything it writes is untracked, so `git pull`
updates never conflict with it.

Prefer to edit `config.yaml` by hand, or want to know exactly what each key
does and what it falls back to when left out? See **CUSTOMIZING.md** — the
full schema reference, including which keys have "feature goes dormant when
empty" semantics you should know about before leaving them blank. See
**SETUP.md** first for the one-time bootstrap steps.

## A note on LinkedIn

The LinkedIn scraper reads LinkedIn's public guest endpoints — no login, no
credentials, conservative rate limits with backoff and a circuit breaker.
LinkedIn's User Agreement prohibits automated scraping; this project ships
the scraper for personal, single-user job searching, and you use it at your
own judgment. If that trade-off isn't for you, leave `search_queries` empty —
every other source is an official JSON API.

## Updating

`git pull` — your personalization lives entirely in untracked files
(config.yaml, .env, data/), so updates never conflict. Then reinstall deps
and re-run tests, or just ask Claude to run the `jobsentinel-update` skill.
See UPDATING.md.

## Contributing & License

This repository is a generated mirror of a private working repo — see
CONTRIBUTING.md for how issues and PRs are handled (short version: issues
welcome; PRs land by re-expression with co-author credit). MIT licensed
(LICENSE).
