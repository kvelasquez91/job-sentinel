# Updating

Your personalization lives entirely in untracked, gitignored files
(`config.yaml`, `.env`, `data/`), so pulling new commits never conflicts with
it. The steps below are what the bundled `jobsentinel-update` skill runs for
you — ask Claude Code to "update Job Sentinel" and it does all of this;
this page is the human-readable version for doing it by hand.

1. **Pull.** `git pull`. Bring your working tree up to date with `main`.

2. **Reinstall dependencies — always.** `pip install -r requirements.txt`,
   even if nothing looks like it changed. An update can add a dependency, and
   a venv that's out of sync turns the test suite red with import errors that
   look like a broken update, not a stale environment.

3. **Run the tests.** `python -m pytest -q` must be green before you trust
   the update. If your venv gets genuinely wedged (odd version-resolution
   errors that `pip install` alone doesn't clear), recreate it as an escape
   hatch:

   ```bash
   rm -rf venv && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
   ```

4. **Check for config drift.** Compare the key tree of your `config.yaml`
   against the freshly pulled `config.example.yaml`. New keys the example
   defines that your config doesn't have are worth a look — but nothing is
   broken by leaving them out: every key not in your `config.yaml` already
   falls back to a safe, documented default in code (see CUSTOMIZING.md).
   Add only the ones you actually want to change; writing a default in
   explicitly just pins you to it and opts you out of future default
   improvements.

5. **If `git pull` conflicts** — only possible if you edited a *tracked*
   file (i.e. not `config.yaml`/`.env`/`data/`) — `git merge --abort`, then
   either re-apply your change on a branch, or better, move the
   customization into `config.yaml` where it belongs, since that's what the
   `policy:` schema in CUSTOMIZING.md exists for.

One more thing worth knowing: `origin` is read-only for you — you don't have
push access to the upstream repository, and updates only ever flow one way
(pull, never push). If you want your own remote backup of your fork plus
your local changes, fork the repository and add your fork as a second
remote; `origin` stays the pristine upstream you pull from.
