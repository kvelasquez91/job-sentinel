"""launchd templates + scripts/render_launchd.py: per-profile labels/log
paths (so multiple checkouts can't hijack each other's agents) and a
config-driven StartCalendarInterval (schedule.daily_times), so custom run
times never require editing a tracked template. SETUP.md §8 runs the render
script; these tests exercise the same code path."""
import pathlib
import plistlib

import pytest

import scripts.render_launchd as rl

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATES = {
    "daily": _ROOT / "com.jobsentinel.daily.plist.template",
    "dashboard": _ROOT / "com.jobsentinel.dashboard.plist.template",
}


def test_labels_are_profile_key_suffixed():
    # A shared, unsuffixed label is what lets SETUP §8 silently re-point a
    # different install's agents at this checkout; the templates must carry
    # the __PROFILE_KEY__ placeholder instead.
    for name, path in _TEMPLATES.items():
        text = path.read_text()
        assert "__PROFILE_KEY__" in text, f"{path.name} lacks __PROFILE_KEY__"
        assert f"<string>com.jobsentinel.{name}</string>" not in text, (
            f"{path.name} still carries the shared, collision-prone label")


def test_daily_template_takes_schedule_from_config_not_tracked_edits():
    # Hardcoded Hour/Minute integers forced owners to edit a TRACKED file for
    # custom times — breaking "setup writes only untracked files".
    text = _TEMPLATES["daily"].read_text()
    assert "__START_CALENDAR_INTERVAL__" in text
    assert "<key>Hour</key>" not in text


def _rendered(config):
    return rl.render(config, repo_dir=str(_ROOT), home="/home/user",
                     claude_dir="/usr/local/bin")


def test_render_default_times_match_old_template_schedule():
    plists = _rendered({})
    assert set(plists) == {"com.jobsentinel.default.daily",
                           "com.jobsentinel.default.dashboard"}
    daily = plistlib.loads(plists["com.jobsentinel.default.daily"].encode())
    assert daily["StartCalendarInterval"] == [
        {"Hour": 2, "Minute": 30}, {"Hour": 13, "Minute": 0}]


def test_render_reads_profile_key_and_schedule_from_config():
    cfg = {"profile": {"key": "alpha"},
           "schedule": {"daily_times": ["07:15", "17:45"]}}
    plists = _rendered(cfg)
    daily = plistlib.loads(plists["com.jobsentinel.alpha.daily"].encode())
    assert daily["Label"] == "com.jobsentinel.alpha.daily"
    assert daily["StartCalendarInterval"] == [
        {"Hour": 7, "Minute": 15}, {"Hour": 17, "Minute": 45}]
    for k in ("StandardOutPath", "StandardErrorPath"):
        assert daily[k].startswith("/home/user/Library/Logs/job-sentinel/alpha/")
    assert "__" not in plists["com.jobsentinel.alpha.daily"]


def test_render_dashboard_agent_is_always_on_with_no_schedule():
    plists = _rendered({"schedule": {"daily_times": ["07:15"]}})
    dash = plistlib.loads(
        plists["com.jobsentinel.default.dashboard"].encode())
    assert "StartCalendarInterval" not in dash
    assert dash["RunAtLoad"] is True and dash["KeepAlive"] is True
    assert "__" not in plists["com.jobsentinel.default.dashboard"]


@pytest.mark.parametrize("bad", ["25:00", "07:60", "0730", "7", "", "aa:bb"])
def test_render_rejects_malformed_times(bad):
    with pytest.raises(ValueError):
        _rendered({"schedule": {"daily_times": [bad]}})


def test_main_writes_plists_creates_log_dir_and_prints_labels(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text(
        "profile:\n  key: beta\nschedule:\n  daily_times: ['06:00']\n")
    out_dir = tmp_path / "agents"
    rc = rl.main(["--out-dir", str(out_dir), "--home", str(tmp_path),
                  "--config", str(tmp_path / "config.yaml"),
                  "--claude-dir", "/usr/local/bin"])
    assert rc == 0
    labels = capsys.readouterr().out.split()
    assert labels == ["com.jobsentinel.beta.daily",
                      "com.jobsentinel.beta.dashboard"]
    for label in labels:
        data = plistlib.loads((out_dir / f"{label}.plist").read_bytes())
        assert data["Label"] == label
    assert (tmp_path / "Library/Logs/job-sentinel/beta").is_dir()


def test_main_guard_refuses_foreign_checkout_plist(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("profile:\n  key: beta\n")
    out_dir = tmp_path / "agents"
    out_dir.mkdir()
    foreign = out_dir / "com.jobsentinel.beta.daily.plist"
    foreign.write_text("<plist><string>/some/other/checkout</string></plist>")

    rc = rl.main(["--out-dir", str(out_dir), "--home", str(tmp_path),
                  "--config", str(tmp_path / "config.yaml"),
                  "--claude-dir", "/usr/local/bin"])
    assert rc == 0
    captured = capsys.readouterr()
    # foreign plist untouched, its label withheld from stdout (so SETUP §8's
    # launchctl loop never bootstraps it), and a stderr note explains why
    assert foreign.read_text().startswith("<plist><string>/some/other")
    assert "com.jobsentinel.beta.daily" not in captured.out.split()
    assert "com.jobsentinel.beta.dashboard" in captured.out.split()
    assert "different checkout" in captured.err
