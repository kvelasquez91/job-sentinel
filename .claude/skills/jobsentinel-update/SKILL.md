---
name: jobsentinel-update
description: Use when the user asks to update Job Sentinel to the latest upstream version — pull, sync dependencies, test, and surface new config options.
---

# Job Sentinel update

1. `git status` — if tracked files are modified, stop: stash or branch first
   (personalization lives in untracked files, so a clean tree is normal).
2. `git pull`.
3. `pip install -r requirements.txt` — ALWAYS, even if it looks unnecessary:
   an update may add dependencies, and an unsynced venv turns the test suite
   red with import errors that look like a broken update. (pip adds but never
   removes; if dependencies ever get genuinely wedged, recreate:
   `rm -rf venv && python3 -m venv venv && ./venv/bin/pip install -r
   requirements.txt`.)
4. `python -m pytest -q` — must be green before using the update.
5. Config-drift check: compare the key tree of the user's `config.yaml`
   against the freshly pulled `config.example.yaml`. Report keys the example
   defines that the user's config lacks — quote the example's inline doc and
   default — and say explicitly that absent keys already fall back to safe
   defaults in code (nothing is broken). OFFER to add any the user wants to
   set; never add silently (writing a default pins it, opting them out of
   future default improvements).
6. If the pull conflicted (only possible if they edited tracked files):
   `git merge --abort`, re-apply their change on a branch, retry — or move
   the customization into config where it belongs (see CUSTOMIZING.md).
