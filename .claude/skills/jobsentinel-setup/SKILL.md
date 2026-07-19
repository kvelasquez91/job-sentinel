---
name: jobsentinel-setup
description: Use when config.yaml is missing or unpersonalized, or the user asks to set up / personalize Job Sentinel — an interview that writes config.yaml, the experience inventory, .env, and the run schedule. Writes ONLY untracked files.
---

# Job Sentinel setup interview

You are personalizing Job Sentinel for its new owner. Ask ONE question at a
time; apply answers by writing config files, never by editing tracked code.
Everything you write is untracked/gitignored — updates can never conflict.

Before starting: `cp config.example.yaml config.yaml` if config.yaml does not
exist. Open config.example.yaml alongside — every key you will set is
documented inline there; CUSTOMIZING.md holds the full schema.

## 1. Profession & targets
Ask: profession/role family, 3-6 target job titles, seniority sought.
Write: `profile.target_titles`, `search_queries` (phrase them the way a
recruiter or LinkedIn search would), `policy.title_gates.baseline` (lowercase
title substrings that define the role family — this gates all seven ATS
scrapers: greenhouse, lever, ashby, workday, eightfold, smartrecruiters,
successfactors [WTTJ and HN "Who is hiring?" have their own separate
`policy.wttj.*` / `policy.hn.*` config, set in step 4]; when empty, those
seven scrapers pass ONLY titles matching your `target_titles`, so set at
least one of the two), `policy.title_gates.domain`
(domain terms for title relevance), `policy.keywords.primary` (5-8 core skill
terms, 10 pts each) and `.secondary` (5-8 supporting terms, 5 pts each),
`policy.prefilter.*` patterns (regexes for roles to CAP without an LLM call —
derive from what the owner is NOT: e.g. an engineer's non_target_titles caps
recruiting/legal/medical titles; target_keywords is the words that rescue a
title, like "\\b(engineer|developer|architect)\\b"), and
`policy.prefilter.domain_signal_pattern` (regex of domain terms; empty
disables the domain hard-cap).
Then rewrite the four rubric blocks under `policy.rubric.*`: UNCOMMENT each
`*_block` key before writing into it — a key left present but set to `""`
still breaks the prompt (splices a blank dimension in), where absent/
commented is what keeps the safe generic default (see CUSTOMIZING.md
"Rubric blocks"). Keep the exact structural shape of the defaults (numbered
header question, 4-6 score bands, an IMPORTANT clarification; dimension 4
may add a HARD CAP sentence tied to domain_signal_pattern's terms) but make
the content the owner's profession. NEVER change the dimension names/ranges
or the JSON key `ai_domain_fit` — the response parser depends on them.

## 2. Location
Ask: remote-only, or a commuter area? If commuter: cities + state.
Write: remote-only → `local_locations: []` (everything local goes dormant —
supported and tested). Commuter → `local_locations` (city list),
`policy.geography.local_state_pattern` (e.g. "texas|tx\\b" — REQUIRED for any
local matching), `linkedin_local_geo` (exact LinkedIn location string),
the three `rubric_local_*` prose values, and optionally `local_target_titles`
+ `local_search_queries` for broadened local coverage.

## 3. Compensation
Ask: remote/national floor, local floor (if commuter), stretch/relocation bar.
Write: the seven `policy.comp.*` keys (target, local_full, local_partial,
remote_partial, cap_low, cap_mid, relocation_exception) and
`dashboard.comp_tiers` (three ascending chip values). Sanity: cap_low <
cap_mid < target; local_partial < local_full.

## 4. Sources
Ask: which companies/boards matter? For each ATS: find tenant IDs using the
verification probes documented inline in config.example.yaml (Workday POST
probe, SuccessFactors curl probe, etc.) — never guess a tenant slug; verify
each one. Write: `workday_tenants`, `eightfold_tenants`,
`smartrecruiters_companies`, `successfactors_tenants`, `greenhouse_slugs`,
`lever_slugs`, `ashby_slugs`, `policy.companies.high_paying` / `.priority`,
`policy.wttj.*`, `policy.hn.role_pattern` (+ `remote_only`), LinkedIn
`search_queries`. LinkedIn note (transparency): it scrapes public guest
endpoints; LinkedIn's ToS prohibits scraping — rate limits are conservative
and no login is used, but the owner should know that is the mechanism.
Keep `linkedin_max_pages: 1` until after the first run.

## 5. Identity & background
Ask for a resume or background summary. Write: `llm_scoring.resume_summary`
(10-20 lines: background, skills, target roles, salary floor, location
policy — scoring fails loudly until this exists); copy
`data/experience_inventory.example.md` → `data/experience_inventory.md` and
fill it from the owner's REAL background. The three section headers
`HARD FACTS`, `NOT CLAIMED`, `BASELINE PRODUCT CRAFT` are load-bearing
literals — never rename them. NOT CLAIMED is binding: the judge and tailor
treat anything listed there as absent; only claims the owner can defend in an
interview belong elsewhere. Set `profile.key` (any short slug; it partitions
the database). Optional resume tailoring: SETUP.md §9 (Google OAuth), then
`.env` MASTER_RESUME_DOC_ID + TAILOR_USER_NAME and `policy.tailor.*`
(master_title_line must be byte-exact from their master doc).

## 6. Schedule
Ask: how often to run? Write the times to `schedule.daily_times` in
config.yaml (24h "HH:MM" strings; omit the key for the 02:30/13:00 default)
— NEVER edit the tracked plist templates. macOS: then run SETUP.md §8's
snippet (scripts/render_launchd.py + launchctl; agent labels and log paths
get suffixed with profile.key, so a second checkout/profile never overwrites
an existing install's agents, and foreign-checkout plists are refused; the
dashboard agent has no schedule — it runs continuously via
RunAtLoad+KeepAlive). Offer to run it. Linux: emit the cron line from
SETUP.md §8 with their times.

## 7. Verify
Run `python -m pytest -q` → must be green. First run: confirm
`linkedin_max_pages: 1`, then `python main.py --scrape-only` (NEVER
`--dry-run` first — it skips DB dedup and is the heaviest LinkedIn
footprint). Then `python main.py --dashboard` → http://127.0.0.1:<dashboard.
port from config.yaml, 8500 by default — pick a free port there if 8500 is
taken> — walk the owner through the chips/filters they configured. After a good first
run, raise `linkedin_max_pages` to 3.
