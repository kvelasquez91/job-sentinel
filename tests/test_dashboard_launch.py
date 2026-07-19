"""Dashboard-mode launch wiring: uvicorn.run must receive access_log=False
(launchd's StandardOutPath/StandardErrorPath files never rotate, and uvicorn
access lines are what would grow them unboundedly under a polling UI) plus
the host/port from config.yaml. uvicorn.run is stubbed — nothing binds."""
import sys

import uvicorn

import main


def test_dashboard_mode_disables_uvicorn_access_log(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dashboard:\n  port: 8765\n  host: \"127.0.0.1\"\n")

    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    # main() os.chdir()s to the config's parent; monkeypatch.chdir restores CWD after.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--dashboard", "--config", str(cfg)]
    )

    main.main()

    assert captured["app"] == "dashboard.app:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["reload"] is False
    assert captured["access_log"] is False
