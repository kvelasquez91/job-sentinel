"""launchd templates: per-profile labels and log paths so multiple checkouts
(e.g. a production install plus a second clone or profile) can never render
agents that overwrite or hijack each other. Rendering here mirrors the sed
substitutions documented in SETUP.md §8."""
import pathlib
import plistlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATES = {
    "daily": _ROOT / "com.jobsentinel.daily.plist.template",
    "dashboard": _ROOT / "com.jobsentinel.dashboard.plist.template",
}


def _render(text, key="testkey"):
    return (text.replace("__REPO_DIR__", "/tmp/checkout")
                .replace("__HOME__", "/home/user")
                .replace("__CLAUDE_DIR__", "/usr/local/bin")
                .replace("__PROFILE_KEY__", key))


def test_labels_are_profile_key_suffixed():
    # A shared, unsuffixed label is what lets SETUP §8 silently re-point a
    # different install's agents at this checkout; the templates must carry
    # the __PROFILE_KEY__ placeholder instead.
    for name, path in _TEMPLATES.items():
        text = path.read_text()
        assert "__PROFILE_KEY__" in text, f"{path.name} lacks __PROFILE_KEY__"
        assert f"<string>com.jobsentinel.{name}</string>" not in text, (
            f"{path.name} still carries the shared, collision-prone label")


def test_rendered_plists_parse_with_per_profile_label_and_logs():
    for name, path in _TEMPLATES.items():
        rendered = _render(path.read_text())
        assert "__" not in rendered, "unsubstituted placeholder left behind"
        data = plistlib.loads(rendered.encode())
        assert data["Label"] == f"com.jobsentinel.testkey.{name}"
        for k in ("StandardOutPath", "StandardErrorPath"):
            assert data[k].startswith(
                "/home/user/Library/Logs/job-sentinel/testkey/")
        assert data["WorkingDirectory"] == "/tmp/checkout"
