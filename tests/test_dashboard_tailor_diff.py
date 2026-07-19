"""dashboard tailor-diff endpoints: GET (pure read) + POST sync (backfill)."""
import asyncio
import importlib
import json
import sqlite3

import pytest
from fastapi import HTTPException

import main as main_mod

app_mod = importlib.import_module("dashboard.app")

MASTER = ("SENIOR AI PRODUCT MANAGER | ENTERPRISE AUTOMATION\n"
          "Old summary line.\nShared body line.\n")
EDITS = {
    "title_line_replacement": "Principal Product Manager",
    "summary_replacement": "New GenAI summary.",
    "skills_reorder": {}, "experience_edits": [], "rewritten_bullets": [],
    "master_text": MASTER,
}


def _db(tmp_path):
    db = tmp_path / "jobs.db"
    main_mod.init_database(str(db)).close()
    return db


def _seed(db, *, edits=EDITS, final_text=None, source=None, verdicts=None,
          doc_url="https://docs.google.com/document/d/abc123XYZ_-/edit"):
    conn = sqlite3.connect(str(db))
    with conn:
        cur = conn.execute(
            "INSERT INTO tailor_history (job_id, created_at, google_doc_url, "
            "edits_json, final_text, final_text_source, edit_verdicts_json, "
            "warnings_json) VALUES (1, '2026-07-15', ?, ?, ?, ?, ?, ?)",
            (doc_url,
             json.dumps(edits) if isinstance(edits, dict) else edits,
             final_text, source,
             json.dumps(verdicts) if verdicts is not None else None,
             json.dumps({"layout": [], "must_have": []})))
        hid = cur.lastrowid
    conn.close()
    return hid


def test_get_404_for_unknown_row(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_db(tmp_path)))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_mod.get_tailor_diff(999))
    assert exc.value.status_code == 404


def test_get_reports_missing_final_text(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db)  # backlog row: no final_text
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    assert asyncio.run(app_mod.get_tailor_diff(hid)) == {"final_text_missing": True}


def test_get_never_writes(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    asyncio.run(app_mod.get_tailor_diff(hid))
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT final_text, final_text_source FROM "
                       "tailor_history WHERE id = ?", (hid,)).fetchone()
    conn.close()
    assert row == (None, None)  # GET on a backlog row changed nothing


def test_get_returns_diff_payload_for_stored_row(tmp_path, monkeypatch):
    db = _db(tmp_path)
    final = ("PRINCIPAL PRODUCT MANAGER\nNew GenAI summary.\n"
             "Shared body line.\n")
    stored_verdicts = [{"section": "title", "label": "Title",
                        "verdict": "landed", "before": "x", "after": "y"}]
    hid = _seed(db, final_text=final, source="pipeline",
                verdicts=stored_verdicts)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    out = asyncio.run(app_mod.get_tailor_diff(hid))
    assert out["final_text_source"] == "pipeline"
    assert out["verdicts"] == stored_verdicts     # stored, not recomputed
    assert any(b["type"] == "change" for b in out["blocks"])
    assert set(out["warnings"]) == {"layout", "must_have"}


def test_get_diff_unavailable_on_malformed_or_missing_master(tmp_path, monkeypatch):
    db = _db(tmp_path)
    bad = _seed(db, edits="not json{", final_text="whatever", source="pipeline")
    no_master = _seed(db, edits={"title_line_replacement": "T"},
                      final_text="whatever", source="pipeline")
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    assert "diff_unavailable" in asyncio.run(app_mod.get_tailor_diff(bad))
    assert "diff_unavailable" in asyncio.run(app_mod.get_tailor_diff(no_master))


def test_get_malformed_verdicts_and_warnings_degrade_to_defaults(tmp_path, monkeypatch):
    """Defense-in-depth behind the json_valid CHECKs: if malformed JSON ever
    reaches these columns (constraint dropped, hand-edit with checks off),
    the payload degrades to defaults instead of raising."""
    db = _db(tmp_path)
    hid = _seed(db, final_text="PRINCIPAL PRODUCT MANAGER\nShared body line.\n",
                source="pipeline")
    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            "UPDATE tailor_history SET edit_verdicts_json = 'not[json', "
            "warnings_json = '{broken' WHERE id = ?", (hid,))
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    conn.close()
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    out = asyncio.run(app_mod.get_tailor_diff(hid))
    assert "diff_unavailable" not in out          # edits_json is fine
    assert out["verdicts"] == []                   # defaulted, not crashed
    assert out["warnings"] == {}                   # defaulted, not crashed
    assert any(b["type"] == "change" for b in out["blocks"])


def test_get_never_writes_on_full_payload_branch(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db, final_text="some final text", source="pipeline", verdicts=[])
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    before = dict(conn.execute(
        "SELECT * FROM tailor_history WHERE id = ?", (hid,)).fetchone())
    conn.close()
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    asyncio.run(app_mod.get_tailor_diff(hid))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    after = dict(conn.execute(
        "SELECT * FROM tailor_history WHERE id = ?", (hid,)).fetchone())
    conn.close()
    assert after == before  # every column identical — GET wrote nothing


class _FakeGoogleClient:
    """Counts fetches; returns a doc whose text lands the title edit."""
    fetch_count = 0

    def authenticate(self):
        pass

    def read_document(self, doc_id):
        type(self).fetch_count += 1
        return {"doc": doc_id}

    def extract_plain_text(self, doc):
        return "PRINCIPAL PRODUCT MANAGER\nOld summary line.\nShared body line.\n"


def _patch_tailor_deps(monkeypatch):
    monkeypatch.setattr(app_mod, "_TAILOR_AVAILABLE", True)
    _FakeGoogleClient.fetch_count = 0
    monkeypatch.setattr(app_mod, "GoogleAPIClient", _FakeGoogleClient,
                        raising=False)
    monkeypatch.setattr(app_mod, "_SKILL_SUBCATEGORY_LABELS",
                        {"AI Platforms": "Al Platforms:"}, raising=False)
    monkeypatch.setattr(app_mod, "_MASTER_TITLE_LINE",
                        "SENIOR AI PRODUCT MANAGER | ENTERPRISE AUTOMATION",
                        raising=False)


def test_sync_503_when_tailor_unavailable(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    monkeypatch.setattr(app_mod, "_TAILOR_AVAILABLE", False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_mod.sync_tailor_diff(hid))
    assert exc.value.status_code == 503


def test_sync_fills_row_and_is_idempotent(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db)
    # A real jobs row makes the never-badges assertion below non-vacuous:
    # an UPDATE-shaped regression would hit this row and fail the pin.
    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute("INSERT INTO jobs (id, title, company, url, profile) "
                     "VALUES (1, 't', 'c', 'https://x/1', 'testuser')")
    conn.close()
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    _patch_tailor_deps(monkeypatch)

    out1 = asyncio.run(app_mod.sync_tailor_diff(hid))
    assert out1["final_text_source"] == "live_fetch"
    by = {v["section"]: v["verdict"] for v in out1["verdicts"]}
    assert by["title"] == "landed"        # uppercased replacement in doc
    assert by["summary"] == "not_landed"  # doc kept the old summary

    out2 = asyncio.run(app_mod.sync_tailor_diff(hid))  # second call
    assert _FakeGoogleClient.fetch_count == 1          # no second fetch
    assert out2["final_text_source"] == "live_fetch"

    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT tailor_issue_json FROM jobs WHERE id = 1").fetchone()
    conn.close()
    assert row is not None and row[0] is None  # sync must never badge a job row


def test_sync_conditional_write_never_overwrites(tmp_path):
    """First-writer-wins at the SQL layer, independent of the endpoint's
    short-circuit: the conditional UPDATE is a no-op on a non-NULL row."""
    db = _db(tmp_path)
    hid = _seed(db, final_text="already synced text", source="live_fetch",
                verdicts=[])
    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute(
            "UPDATE tailor_history SET final_text = ?, "
            "final_text_source = 'live_fetch', edit_verdicts_json = '[]' "
            "WHERE id = ? AND final_text IS NULL",
            ("intruder text", hid))
        kept = conn.execute("SELECT final_text FROM tailor_history "
                            "WHERE id = ?", (hid,)).fetchone()[0]
    conn.close()
    assert kept == "already synced text"


def test_sync_404_when_no_doc_url(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db, doc_url=None)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    _patch_tailor_deps(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_mod.sync_tailor_diff(hid))
    assert exc.value.status_code == 404


def test_sync_fetch_failure_writes_nothing(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    _patch_tailor_deps(monkeypatch)

    class Boom(_FakeGoogleClient):
        def read_document(self, doc_id):
            raise RuntimeError("doc deleted")

    monkeypatch.setattr(app_mod, "GoogleAPIClient", Boom, raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_mod.sync_tailor_diff(hid))
    assert exc.value.status_code == 502
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT final_text FROM tailor_history WHERE id = ?",
                       (hid,)).fetchone()
    conn.close()
    assert row[0] is None


def test_sync_malformed_edits_json_no_fetch_no_write(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db, edits="not json{")  # backlog row, malformed edits_json
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    _patch_tailor_deps(monkeypatch)
    out = asyncio.run(app_mod.sync_tailor_diff(hid))
    assert "diff_unavailable" in out
    assert _FakeGoogleClient.fetch_count == 0      # never fetched
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT final_text, final_text_source FROM "
                       "tailor_history WHERE id = ?", (hid,)).fetchone()
    conn.close()
    assert row == (None, None)                     # never wrote


def test_sync_no_master_text_skips_fetch_and_write(tmp_path, monkeypatch):
    db = _db(tmp_path)
    hid = _seed(db, edits={"title_line_replacement": "T"})  # valid JSON, no master_text
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    _patch_tailor_deps(monkeypatch)
    out = asyncio.run(app_mod.sync_tailor_diff(hid))
    assert "diff_unavailable" in out
    assert _FakeGoogleClient.fetch_count == 0
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT final_text FROM tailor_history WHERE id = ?",
                       (hid,)).fetchone()
    conn.close()
    assert row[0] is None
