# Customizing Job Sentinel

`config.yaml` is the **only** personalization surface. Nothing about who
you are, what you're searching for, or where you live belongs in a tracked
file — every owner-specific value (titles, keywords, comp bars, geography,
title gates, rubric prose, prefilter patterns, tailor anchors, dashboard
labels) is a config key, read once at import time by `profile_policy.py`. A
key you leave absent — or an absent `config.yaml` entirely — falls back to a
generic, functional default baked into `profile_policy.py`'s `_DEFAULTS`
table. The easiest way to set all of this is to open Claude Code in the repo
and say **"set me up"**; this page is the schema reference for doing it by
hand, or for understanding what the interview wrote.

## Empty-value semantics

Config values aren't just "unset = generic". Several keys have specific,
tested "feature dormant" behavior when left empty — know these before you
assume a blank list is harmless:

- **`policy.title_gates.baseline` empty** — the shared ATS scraper gate,
  applied by all seven ATS scrapers (Greenhouse, Lever, Ashby, Workday,
  Eightfold, SmartRecruiters, SuccessFactors), no longer passes every title
  through unfiltered; instead those scrapers pass *only* jobs whose title
  token-matches your `profile.target_titles`. If **both**
  `policy.title_gates.baseline` and `target_titles` are empty, every ATS
  source is dormant — nothing from those scrapers survives the filter. (WTTJ
  and HN "Who is hiring?" are separate sources with their own keyword/pattern
  config — see the next two bullets — not part of this shared gate.)
- **`policy.wttj.title_keywords` empty** (with no `wttj_queries` override and
  an empty `policy.wttj.default_queries`) — Welcome to the Jungle is dormant.
- **`policy.hn.role_pattern` empty** — Hacker News "Who is hiring?" is
  dormant; no comment can match a `None` pattern.
- **A `policy.prefilter.*` pattern empty** — that specific rule is off, and
  jobs it would have capped instead defer entirely to the LLM judge.
- **`policy.keywords.primary` / `.secondary` empty** — no keyword-score
  contribution from that tier (fine if you rely on the LLM judge alone).
- **`local_locations` empty, OR `policy.geography.local_state_pattern`
  empty** — the entire local-commuter-area layer goes dormant: no job is
  ever treated as "local" by the scorer, scrapers, or LLM rubric, regardless
  of what the other of the pair is set to. Both must be non-empty together
  for local matching to engage.
- **`policy.rubric.*_block` keys must stay ABSENT, not empty-string, to use
  the generic default rubric.** This one is the opposite of the others:
  `""` is a real value, not "unset". If you set
  `policy.rubric.role_match_block: ""`, `profile_policy.py` returns that
  empty string as-is — it does **not** fall back to the generic block — and
  the LLM system prompt gets a blank dimension spliced in. That's a
  misconfiguration, not a supported "disable this dimension" path, which is
  why `config.example.yaml` deliberately leaves these four keys **commented
  out** rather than blanked. Un-comment and rewrite a block only when you're
  ready to replace it with your own prose (see "Rubric blocks" below).

## Key reference

Every key below lives under `policy:` in `config.yaml` unless noted (`profile.key`
and the `dashboard.*` keys are top-level siblings of `policy:`). Type shown
is the Python type after loading; "Consumer" is the module(s) that read the
resolved value.

| Key | Type | Default | Consumer | Meaning |
|---|---|---|---|---|
| `profile.key` | str | `"default"` | main.py, dashboard/app.py, engine/llm_scorer.py | DB partition key — scopes jobs/runs/API calls to one owner (the `profile` column) |
| `policy.owner_pins.rubric_sha` | str | `""` | profile_policy.py, tests (CI guard) | Optional fork-maintainer pin: sha256 of your assembled fit-scoring system template; non-empty makes the byte-identity guard test assert it, empty skips that guard. See "Owner pins" below |
| `policy.owner_pins.summary_mentions_relocation` | bool | `false` | profile_policy.py, tests (CI guard) | Optional fork-maintainer opt-in: `true` makes the owner-lint assert your `resume_summary` names your relocation bar; `false` skips it. See "Owner pins" below |
| `policy.geography.local_state_pattern` | str (regex) | `""` | local_area.py, scrapers/linkedin.py, engine/llm_scorer.py | State/region regex a `local_locations` city must be directly followed by to count as local; empty disables local matching |
| `policy.geography.linkedin_local_geo` | str | `""` | scrapers/linkedin.py | Exact LinkedIn geo-search string used by the local search layer |
| `policy.geography.rubric_local_area_prose` | str | `"the candidate's local commuter area (see profile)"` | engine/llm_scorer.py | Long-form local-area phrase spliced into the `remote_location` rubric dimension |
| `policy.geography.rubric_local_area_short` | str | `"the candidate's local commuter area"` | engine/llm_scorer.py | Short-form local-area phrase used in the comp-calibration rubric prose |
| `policy.geography.rubric_local_anchor` | str | `"local-market senior role"` | engine/llm_scorer.py | Anchor phrase describing a full-local-comp role, used in the comp estimation prose |
| `policy.comp.target` | int (USD/yr) | `220000` | engine/scorer.py, engine/llm_scorer.py | Remote/national comp bar (full points) |
| `policy.comp.local_full` | int (USD/yr) | `150000` | engine/scorer.py, engine/llm_scorer.py | Local-market full-points comp bar |
| `policy.comp.local_partial` | int (USD/yr) | `120000` | engine/scorer.py, engine/llm_scorer.py | Local-market partial-credit comp bar |
| `policy.comp.remote_partial` | int (USD/yr) | `180000` | engine/scorer.py, engine/llm_scorer.py | Remote partial-credit comp bar |
| `policy.comp.cap_low` | int (USD/yr) | `170000` | engine/scorer.py, engine/llm_scorer.py | Lower boundary of the LLM `comp_match` hard-cap bands |
| `policy.comp.cap_mid` | int (USD/yr) | `200000` | engine/scorer.py, engine/llm_scorer.py | Middle boundary of the LLM `comp_match` hard-cap bands |
| `policy.comp.relocation_exception` | int (USD/yr) | `300000` | engine/scorer.py, engine/llm_scorer.py | Posted top-of-band that exempts an onsite/hybrid US role from the remote-location penalty |
| `policy.keywords.primary` | list[str] | `[]` | engine/scorer.py, engine/llm_scorer.py | Core skill terms, worth 10 keyword-score points each; empty contributes none |
| `policy.keywords.secondary` | list[str] | `[]` | engine/scorer.py, engine/llm_scorer.py | Supporting skill terms, worth 5 keyword-score points each |
| `policy.companies.high_paying` | list[str] | `[]` | engine/scorer.py, engine/llm_scorer.py | Companies flagged for a high-paying scoring bonus |
| `policy.companies.priority` | list[str] | `[]` | engine/scorer.py, engine/llm_scorer.py | Companies flagged for a priority-company scoring bonus |
| `policy.title_gates.baseline` | list[str] | `[]` | scrapers/greenhouse.py (also handles lever + ashby boards; shared by smartrecruiters.py, successfactors.py, workday.py, eightfold.py) | Lowercase title substrings gating all seven ATS scrapers (greenhouse, lever, ashby, workday, eightfold, smartrecruiters, successfactors); see "Empty-value semantics" above |
| `policy.title_gates.domain` | list[str] | `[]` | scrapers/greenhouse.py | Domain terms (e.g. AI/ML) checked in a title for a stricter domain-relevance signal |
| `policy.wttj.title_keywords` | list[str] | `[]` | scrapers/wttj.py | Title substrings gating Welcome to the Jungle results; empty passes no titles |
| `policy.wttj.default_queries` | list[str] | `[]` | scrapers/wttj.py | Default WTTJ Algolia search queries used when `wttj_queries` is unset |
| `policy.hn.role_pattern` | str (regex) | `""` | scrapers/hn_whoishiring.py | Regex a HN "Who is hiring?" comment must match to be treated as a candidate role; empty = source dormant |
| `policy.hn.remote_only` | bool | `true` | scrapers/hn_whoishiring.py | Require the literal word "remote" in the comment text |
| `policy.rubric.role_match_block` | str | generic `role_match` prose | engine/llm_scorer.py | Full text of rubric dimension 1 (`role_match`, 0-30); leave ABSENT for the default |
| `policy.rubric.seniority_match_block` | str | generic `seniority_match` prose | engine/llm_scorer.py | Full text of rubric dimension 2 (`seniority_match`, 0-20); leave ABSENT for the default |
| `policy.rubric.remote_location_block` | str | generic `remote_location` prose | engine/llm_scorer.py | Full text of rubric dimension 3 (`remote_location`, 0-20); leave ABSENT for the default |
| `policy.rubric.domain_fit_block` | str | generic `ai_domain_fit` prose | engine/llm_scorer.py | Full text of rubric dimension 4 (`ai_domain_fit`, 0-20); leave ABSENT for the default |
| `policy.prefilter.non_target_titles` | str (regex) | `""` | engine/llm_scorer.py | Titles to hard-cap without an LLM call; empty disables the rule (defers to the LLM) |
| `policy.prefilter.sales_bd_titles` | str (regex) | `""` | engine/llm_scorer.py | Sales/BD titles to hard-cap; empty disables |
| `policy.prefilter.solutions_cs_titles` | str (regex) | `""` | engine/llm_scorer.py | Solutions/customer-success titles to hard-cap; empty disables |
| `policy.prefilter.eng_title_keywords` | str (regex) | `""` | engine/llm_scorer.py | Engineering-title signal (relevant when your own targets ARE engineering roles); empty disables |
| `policy.prefilter.non_target_adjacent_titles` | str (regex) | `""` | engine/llm_scorer.py | Adjacent-but-not-target titles to soft-cap; empty disables |
| `policy.prefilter.target_keywords` | str (regex) | `""` | engine/llm_scorer.py | Words that rescue a title from one of the caps above; empty disables |
| `policy.prefilter.domain_signal_pattern` | str (regex) | `""` | engine/llm_scorer.py | Domain terms whose total absence from the description hard-caps `ai_domain_fit` at 8/20; empty disables the hard cap |
| `policy.tailor.master_title_line` | str | `""` | resume_tailor/ | Byte-exact title line at the top of your master résumé Google Doc; `apply_edits` anchors on it |
| `policy.tailor.skill_subcategory_labels` | dict | `{}` | resume_tailor/ | Bold skill-block labels in your master doc, mapped for the tailoring/reorder logic |
| `policy.tailor.extra_ats_headers` | list[str] | `[]` | resume_tailor/ | Your own résumé section headers the ATS checker should treat as standard (not flagged) |
| `dashboard.page_title` | str | `"Job Sentinel — Opportunities"` | dashboard/app.py | Dashboard page `<title>` / header text |
| `dashboard.comp_tiers` | list[int] | `[150000, 200000, 250000]` | dashboard/app.py | Three ascending comp filter-chip dollar breakpoints; must match the analytics histogram's buckets |
| `dashboard.profiles` | dict | `{}` | dashboard/app.py | `profile.key` → `{label, subtitle}` display strings for the dashboard's profile selector |

(`local_locations`, `profile.target_titles`, `search_queries`, ATS tenant
lists, and the other top-level `config.yaml` keys outside `policy:` /
`profile.key` / `dashboard:` are documented inline in `config.example.yaml`
— this table covers the `_DEFAULTS`-backed policy schema specifically.)

## Rubric blocks

Write your own rubric blocks by **uncommenting** the relevant
`policy.rubric.*_block` key and filling it in place (the `jobsentinel-setup`
skill does this in its step 1) — a key left present but set to an empty
string still breaks the prompt (splices a blank dimension in), where leaving
it absent/commented is what engages the safe generic default; see
"Empty-value semantics" above.

The four `policy.rubric.*_block` values are spliced **verbatim** into the
LLM system prompt that scores every job — they're not wrapped, truncated, or
reformatted, so whatever you write is exactly what the model sees as that
dimension's scoring instructions. A few structural rules apply if you
rewrite one:

- Keep the dimension's name and JSON key intact. The response parser reads
  a fixed set of JSON keys back from the model; renaming
  `ai_domain_fit`, `role_match`, `seniority_match`, `remote_location`, or
  `comp_match` breaks scoring. `ai_domain_fit` in particular is a **frozen**
  key name — you can rewrite everything about what that dimension measures
  (it doesn't have to be about AI at all), but the JSON key it reports under
  must stay `ai_domain_fit`.
- Keep each dimension's 0-N range intact (`role_match` 0-30,
  `seniority_match` 0-20, `remote_location` 0-20, `ai_domain_fit` 0-20) —
  `fit_score` is the sum of all five dimensions and downstream code assumes
  these ranges.
- You can use these `%%TOKEN%%` placeholders inside any rubric block; they
  resolve from your `policy.geography.*` and `policy.comp.*` values before
  the prompt is sent: `%%LOCAL_AREA_PROSE%%`, `%%LOCAL_AREA_SHORT%%`,
  `%%LOCAL_ANCHOR%%`, `%%EXCEPTION_COMP%%`, `%%EXCEPTION_COMP_K%%`,
  `%%COMP_TARGET_K%%`, `%%COMP_TARGET_MINUS1_K%%`, `%%COMP_CAP_LOW_K%%`,
  `%%LOCAL_FULL_K%%`, `%%LOCAL_FULL_MINUS1_K%%`, `%%LOCAL_PARTIAL_K%%`.

**Warning:** changing rubric prose changes what the LLM is asked, which
changes its answers. A prompt-text edit re-rolls the LLM score on every job
that subsequently gets re-judged (the local `claude` CLI has no
temperature/seed control, so re-judging is inherently non-deterministic on
top of that). Treat rubric edits as a deliberate, occasional decision — not
something to tweak casually while chasing a single job's score, since you'll
be re-scoring your whole backlog's fit dimension, not just the one job.

## Owner pins (optional)

Two keys under `policy.owner_pins` let a **fork maintainer** freeze parts of
their own configuration in CI. Both default to off, and a personalized clone
should leave them off — the suite (`python -m pytest -q`) stays green whether
or not you set them. They exist only for a maintainer who wants their own
tuning guarded against silent drift.

| Key | What it does | When a maintainer sets it |
|---|---|---|
| `policy.owner_pins.rubric_sha` | sha256 of your **assembled** fit-scoring system template (rubric blocks + comp/geography tokens, as rendered). Non-empty makes the byte-identity guard test assert the live template hashes to it; empty (default) skips that guard. | Once your rubric wording is tuned and you want CI to catch any later edit that would silently re-roll the non-deterministic LLM judge or break the `claude` CLI prompt-cache prefix. |
| `policy.owner_pins.summary_mentions_relocation` | `true` makes the owner-lint assert your `llm_scoring.resume_summary` literally contains your relocation bar (e.g. `$300K`); `false` (default) skips the lint. | If your search strategy relies on the relocation exception and you want a guard that your summary keeps naming that bar. |

**Why they're off by default.** Personalizing your rubric (overriding a
`policy.rubric.*_block`) changes the template bytes, so a hardcoded owner sha
would turn red for you against a value you can never match; and a
`resume_summary` that doesn't happen to mention a relocation figure is normal.
Gating both behind opt-in keys keeps "follow the setup interview → `pytest`
is green" true for every personalized fork.

**Setting `rubric_sha`.** Get the current sha, then paste it into config:

```
python -c "import hashlib; from engine.llm_scorer import _FIT_SYSTEM_TEMPLATE; \
print(hashlib.sha256(_FIT_SYSTEM_TEMPLATE.encode()).hexdigest())"
```

On any deliberate change that moves the template (rubric prose, or a
`policy.comp.*` / `policy.geography.*` value the rubric splices), update the
pin in the **same commit** and record the re-judge decision in the commit
message — a surgical, ids-scoped re-judge, or an explicit
accept-drift-on-future-rows note.
