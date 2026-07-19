"""/api/ui-config serves config-driven UI values; index.html hardcodes nothing."""
import asyncio
import importlib

# `import dashboard.app as app_mod` would bind the FastAPI instance, not the
# module (dashboard/__init__.py does `from .app import app`, shadowing the
# submodule). Fetch the real module the same way tests/test_analytics.py does.
app_mod = importlib.import_module("dashboard.app")


def test_ui_config_shape(monkeypatch):
    monkeypatch.setattr(app_mod, "DASHBOARD_PAGE_TITLE", "T")
    monkeypatch.setattr(app_mod, "PROFILE_KEY", "testuser")
    monkeypatch.setattr(app_mod, "DASHBOARD_COMP_TIERS", [1, 2, 3])
    monkeypatch.setattr(app_mod, "DASHBOARD_PROFILES", {"testuser": {"label": "L", "subtitle": "S"}})
    monkeypatch.setattr(app_mod, "_UI_LOCAL_PATTERN", None)
    cfg = asyncio.run(app_mod.get_ui_config())
    assert cfg == {"page_title": "T", "default_profile": "testuser",
                   "profiles": {"testuser": {"label": "L", "subtitle": "S"}},
                   "comp_tiers": [1, 2, 3], "local_pattern": None}


def test_comp_bucket_labels():
    assert app_mod._comp_bucket_labels([150_000, 200_000, 250_000]) == [
        "<150k", "150-200k", "200-250k", "250k+"]
    assert app_mod._comp_bucket_labels([150_000, 200_000, 220_000]) == [
        "<150k", "150-200k", "200-220k", "220k+"]
