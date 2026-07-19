# Job Sentinel — Setup

A bootstrap runbook to get Job Sentinel running from a fresh checkout. Follow the
sections in order — each one assumes the previous is done.

## 1. Requirements

- **macOS or Linux.** Windows is not supported — the code uses `fcntl` and process
  groups. WSL works fine if you're on Windows.
- **Python 3.12+.** Check with `python3 --version`. `main.py` refuses to start on
  anything older with a clear error message.
- **A Claude subscription with Claude Code installed.** This is how Job Sentinel
  does its LLM work — see section 3.

## 2. Install

```bash
git clone <this repository> job-sentinel && cd job-sentinel
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cp config.example.yaml config.yaml
```

## 3. Claude Code

Install Claude Code per https://claude.com/claude-code, then confirm you're logged
in:

```bash
claude auth status
```

This must show you as logged in before anything LLM-related will work.

**Note on billing:** Job Sentinel bills against your Claude subscription, not an
API key — API keys deliberately do **not** work here (the wrapper strips
`ANTHROPIC_*` env vars from the child process before calling the `claude` CLI).

Resume tailoring (optional, see section 9) uses an Opus-class model, which is
available on Max-tier plans. If you're on a Pro plan, set in your `.env`:

```
TAILOR_EDIT_MODEL=claude-sonnet-5
```

Quality caveat: Sonnet does a noticeably less careful job on the tailoring edit
loop than Opus, so expect to review its output more closely.

## 4. Personalize now, before the first run

`config.yaml` exists from step 2 (a copy of `config.example.yaml` with
generic, functional defaults — no target titles, no target cities, no
compensation bars). Before running anything, open Claude Code in this
directory and say:

> set me up

That's the `jobsentinel-setup` skill — an interview covering your
profession, titles, salary bars, location, sources, and schedule. It writes
`config.yaml`, your experience inventory, `.env`, and your run schedule; all
of it lands in untracked, gitignored files, so nothing it writes ever
conflicts with a `git pull`. Prefer doing it by hand instead? See
CUSTOMIZING.md for the full key-by-key schema.

Everything from here on assumes that's done — your search terms, filters,
and identity are in `config.yaml` instead of the generic defaults.

## 5. First run

Confirm `linkedin_max_pages: 1` in `config.yaml` (the example default) before
your very first run. Your database is empty, so the scraper will detail-fetch
every card it sees — LinkedIn rate-limits hard if you point it at more than one
page on an empty DB.

```bash
python main.py --scrape-only
```

Do **not** use `--dry-run` for this first run — it persists nothing, so the dry
run itself scrapes without the dedup set, and everything it fetches gets fetched
again by your first real run. That's the heaviest possible
footprint against LinkedIn, not a lighter one.

Once that completes, check the dashboard (section 7) to see what came in, then
raise `linkedin_max_pages` to `3` in `config.yaml` for normal runs.

## 6. Full run

```bash
python main.py
```

This scrapes, LLM-scores, and saves. It will fail with a clear error until
`llm_scoring.resume_summary` is set in `config.yaml` — that's intentional, not
a bug; the `jobsentinel-setup` skill (§4) writes it for you, or see the
`llm_scoring:` block's inline comment in `config.example.yaml` to write it
by hand.

## 7. Dashboard

```bash
python main.py --dashboard
```

Then open http://127.0.0.1:8500 — or whatever `dashboard.port` you set in
`config.yaml`; if 8500 is already taken on your machine (another app, or a
second Job Sentinel install), just pick a free port there. The dashboard is
unauthenticated by design and binds to localhost only, per `config.yaml`.

Run all commands from the repo root — config paths are resolved relative to it.

## 8. Automation (macOS)

Two launchd agents ship as templates: one for the twice-daily scrape/score run,
one to keep the dashboard always running. Agent labels and log paths are
suffixed with your `profile.key` (from `config.yaml`), so a second checkout or
profile on the same machine gets its own agents instead of silently re-pointing
an existing install's — the render script refuses to overwrite an agent that
belongs to a different checkout. (`profile.key` must be a plain slug — letters,
digits, dots, dashes, underscores; the setup interview writes one.) Run times
come from `schedule.daily_times` in `config.yaml` (24h "HH:MM" strings;
absent = `["02:30", "13:00"]`), so custom times never require editing a
tracked file. Instantiate and load both with:

```bash
./venv/bin/python scripts/render_launchd.py | while read -r label; do
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/"$label.plist"
done
```

Run this from the repo root (the script reads `config.yaml` there; it renders
the plists and prints their labels, and skipped/foreign agents are reported on
stderr and never bootstrapped). The dashboard agent uses `RunAtLoad` +
`KeepAlive`, so once loaded you never need to start `--dashboard` manually
again — it just stays up. To change run times later, edit
`schedule.daily_times` and re-run the snippet. Upgrading from agents installed
before labels were per-profile? See the one-time migration note in UPDATING.md.

**Linux alternative** (no launchd — use cron):

```
30 2,13 * * * cd <repo> && ./venv/bin/python main.py >> logs/cron.log 2>&1
```

## 9. Optional: resume tailoring (Google)

Skip this section entirely if you don't want auto-tailored resumes — everything
else in Job Sentinel works without Google.

1. Create your own Google Cloud project and enable the **Google Docs API** and
   **Google Drive API**.
2. Create an OAuth **Desktop App** client and download its credentials JSON.
3. Save it as `resume_tailor/config/client_secret.json` (the directory doesn't
   exist in a fresh copy — create it first):

   ```bash
   mkdir -p resume_tailor/config
   ```
4. Put your master resume in a Google Doc, then add to `.env`:

   ```
   MASTER_RESUME_DOC_ID=<doc id>
   TAILOR_USER_NAME=<your name>
   ```

5. Run the one-time OAuth setup:

   ```bash
   python scripts/setup_google_auth.py
   ```

   This opens a browser for authorization and verifies it can see your master
   resume doc.

6. Once a manual tailor produces good output, flip on auto-tailoring in
   `config.yaml`:

   ```yaml
   auto_tailor:
     enabled: true
   ```

## 10. Memory tuning

On 8 GB machines, lower the concurrency defaults to avoid memory pressure
during LLM scoring:

- `config.yaml`: `llm_scoring.workers: 2`
- `.env`: `CLAUDE_CLI_MAX_CONCURRENCY=2`

---

That's it — you're running. For deeper personalization (search terms, filters,
scoring weights, experience inventory), say "set me up" for the guided
interview, or see `CUSTOMIZING.md` for the full config schema.
