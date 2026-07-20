"""run_tailor_pipeline: dashboard-independent tailoring core."""
import dataclasses
import json
import sqlite3
from types import SimpleNamespace

import pytest

import filter_match as fm
import main as main_mod
import resume_tailor.pipeline as pipe
import resume_tailor.tailor_engine as te
from claude_cli import ClaudeCLIError, ClaudeCLITimeout
from policy_fixtures import patch_pipeline_title_line


@dataclasses.dataclass
class FakeEdits:
    keyword_count: int = 28
    title_line: str = "Senior PM, AI"


class FakeClient:
    def authenticate(self): pass
    def copy_document(self, master_id, title): return "doc-123"
    def read_document(self, doc_id): return {"body": {}}
    def extract_plain_text(self, doc): return "resume text with GenAI keywords"
    def export_as_pdf(self, doc_id): return b"%PDF"
    def move_to_folder(self, doc_id, folder): pass
    def export_as_docx(self, doc_id, path):
        with open(path, "wb") as f:
            f.write(b"docx")
    def get_document_url(self, doc_id): return f"https://docs.google.com/{doc_id}"


def test_pipeline_returns_result_and_writes_history(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, score, status, profile) "
        "VALUES ('Senior PM AI', 'Acme', 'https://x/1', 92, 'new', 'testuser')")
    conn.commit()
    # tailor_history is now created by main.init_database itself (schema
    # consolidation) — no manual CREATE TABLE needed here.
    conn.close()

    monkeypatch.setattr(pipe, "extract_jd", lambda url: SimpleNamespace(
        raw_text="JD text", company="Acme", title="Senior PM AI"))
    monkeypatch.setattr(pipe, "extract_keywords", lambda text: SimpleNamespace(
        exact_job_title="Senior PM, AI", priority_keywords=["genai", "llm"]))
    monkeypatch.setattr(pipe, "GoogleAPIClient", FakeClient)
    monkeypatch.setattr(pipe, "gap_analysis", lambda master, jd, **kw: {"gaps": []})
    monkeypatch.setattr(pipe, "generate_edits", lambda m, j, g, c, **kw: FakeEdits())
    monkeypatch.setattr(pipe, "apply_edits", lambda doc_id, edits, client: None)
    monkeypatch.setattr(pipe, "build_line_map", lambda pdf: object())   # master map stub
    monkeypatch.setattr(pipe, "enforce_layout",
                        lambda doc_id, edits, doc, master_map, client, **kw: (b"%PDF-final", []))
    monkeypatch.setattr(pipe, "ats_check_all", lambda *a, **k: SimpleNamespace(
        score=88, passed=True, issues=[], warnings=[]))
    monkeypatch.setattr(pipe, "DOCX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "master_resume.txt"))
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))
    monkeypatch.setattr(pipe, "jd_get_token_usage", lambda: {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01})
    monkeypatch.setattr(pipe, "engine_get_token_usage", lambda: {"input_tokens": 20, "output_tokens": 9, "cost_usd": 0.04})
    monkeypatch.setattr(pipe, "jd_reset_token_usage", lambda: None)
    monkeypatch.setattr(pipe, "engine_reset_token_usage", lambda: None)

    steps = []
    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=steps.append)

    assert result["ats_score"] == 88
    assert result["google_doc_url"] == "https://docs.google.com/doc-123"
    # keywords_matched is measured on the FINAL doc text ("resume text with GenAI
    # keywords") via the shared whole-word matcher — "genai" matches, "llm" does
    # not — NOT the pre-revert edits.keyword_count (28).
    assert result["keywords_matched"] == 1
    assert any("Step 1" in s for s in steps)

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    job = conn.execute("SELECT tailored_resume_url, tailored_at FROM jobs WHERE id = 1").fetchone()
    hist = conn.execute("SELECT job_id, ats_score, company FROM tailor_history").fetchone()
    conn.close()
    assert job["tailored_resume_url"] == "https://docs.google.com/doc-123"
    assert job["tailored_at"] is not None
    assert (hist["job_id"], hist["ats_score"], hist["company"]) == (1, 88, "Acme")


def test_pipeline_passes_credited_must_haves_to_layout_guard(tmp_path, monkeypatch):
    # The guard must know which stored must-have terms the judge credited so
    # its repairs preserve them and its reverts don't silently destroy them
    # (the 2026-07-17 incident: a revert dropped two credited terms and the
    # recompute landed 7 points under the judged ceiling).
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    filter_json = json.dumps({"must_haves": [
        {"term": "people management", "aliases": ["team management"],
         "verdict": "explicit"},
        {"term": "quantum computing", "aliases": [], "verdict": "absent"},
    ]})
    conn.execute(
        "INSERT INTO jobs (title, company, url, score, status, profile, filter_json) "
        "VALUES ('Senior PM AI', 'Acme', 'https://x/1', 92, 'new', 'testuser', ?)",
        (filter_json,))
    conn.commit()
    conn.close()

    monkeypatch.setattr(pipe, "extract_jd", lambda url: SimpleNamespace(
        raw_text="JD text", company="Acme", title="Senior PM AI"))
    monkeypatch.setattr(pipe, "extract_keywords", lambda text: SimpleNamespace(
        exact_job_title="Senior PM, AI", priority_keywords=["genai"]))
    monkeypatch.setattr(pipe, "GoogleAPIClient", FakeClient)
    monkeypatch.setattr(pipe, "gap_analysis", lambda master, jd, **kw: {"gaps": []})
    monkeypatch.setattr(pipe, "generate_edits", lambda m, j, g, c, **kw: FakeEdits())
    monkeypatch.setattr(pipe, "apply_edits", lambda doc_id, edits, client: None)
    monkeypatch.setattr(pipe, "build_line_map", lambda pdf: object())
    captured = {}
    def fake_guard(doc_id, edits, doc, master_map, client, credited_items=None):
        captured["credited_items"] = credited_items
        return (b"%PDF-final", [])
    monkeypatch.setattr(pipe, "enforce_layout", fake_guard)
    monkeypatch.setattr(pipe, "ats_check_all", lambda *a, **k: SimpleNamespace(
        score=88, passed=True, issues=[], warnings=[]))
    monkeypatch.setattr(pipe, "DOCX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "master_resume.txt"))
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))
    for name in ("jd_get_token_usage", "engine_get_token_usage"):
        monkeypatch.setattr(pipe, name, lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    for name in ("jd_reset_token_usage", "engine_reset_token_usage"):
        monkeypatch.setattr(pipe, name, lambda: None)

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    # Only the judge-credited item reaches the guard — absent verdicts are
    # prompt-level targets, never layout-guard contracts.
    assert [i["term"] for i in captured["credited_items"]] == ["people management"]


def test_pipeline_surfaces_layout_guard_warnings(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, score, status, profile) "
        "VALUES ('Senior PM AI', 'Acme', 'https://x/1', 92, 'new', 'testuser')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(pipe, "extract_jd", lambda url: SimpleNamespace(
        raw_text="JD text", company="Acme", title="Senior PM AI"))
    monkeypatch.setattr(pipe, "extract_keywords", lambda text: SimpleNamespace(
        exact_job_title="Senior PM, AI", priority_keywords=["genai"]))
    monkeypatch.setattr(pipe, "GoogleAPIClient", FakeClient)
    monkeypatch.setattr(pipe, "gap_analysis", lambda master, jd, **kw: {"gaps": []})
    monkeypatch.setattr(pipe, "generate_edits", lambda m, j, g, c, **kw: FakeEdits())
    monkeypatch.setattr(pipe, "apply_edits", lambda doc_id, edits, client: None)
    monkeypatch.setattr(pipe, "build_line_map", lambda pdf: object())
    monkeypatch.setattr(pipe, "enforce_layout",
                        lambda doc_id, edits, doc, master_map, client, **kw:
                        (b"%PDF-final", ["Layout guard reverted 'summary' to master text (grew)"]))
    captured = {}
    def fake_ats(doc, kws, title, pdf_bytes=None, master_word_count=None):
        captured["pdf_bytes"] = pdf_bytes
        captured["master_word_count"] = master_word_count
        return SimpleNamespace(score=90, passed=True, issues=[], warnings=["ats-warning"])
    monkeypatch.setattr(pipe, "ats_check_all", fake_ats)
    monkeypatch.setattr(pipe, "DOCX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "master_resume.txt"))
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))
    for name in ("jd_get_token_usage", "engine_get_token_usage"):
        monkeypatch.setattr(pipe, name, lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    for name in ("jd_reset_token_usage", "engine_reset_token_usage"):
        monkeypatch.setattr(pipe, name, lambda: None)

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    assert captured["pdf_bytes"] == b"%PDF-final"        # guard PDF reused, no re-export
    assert captured["master_word_count"] == len("resume text with GenAI keywords".split())
    assert "ats-warning" in result["warnings"]
    assert any("reverted" in w.lower() for w in result["warnings"])


def _seed_pipeline_db(tmp_path, filter_json=None):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (id, title, company, url, score, status, profile, "
        "filter_score, filter_score_master, filter_source, filter_knockout, filter_json) "
        "VALUES (1, 'Senior PM AI', 'Acme', 'https://x/1', 92, 'new', 'testuser', "
        "?, ?, ?, ?, ?)",
        (40, 40, "master", 0, filter_json) if filter_json else (None,) * 5,
    )
    conn.commit()
    conn.close()
    return db


def _patch_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(pipe, "extract_jd", lambda url: SimpleNamespace(
        raw_text="JD text", company="Acme", title="Senior PM AI"))
    monkeypatch.setattr(pipe, "extract_keywords", lambda text: SimpleNamespace(
        exact_job_title="Senior PM, AI", priority_keywords=["genai", "llm"]))
    monkeypatch.setattr(pipe, "GoogleAPIClient", FakeClient)
    monkeypatch.setattr(pipe, "gap_analysis", lambda master, jd, **kw: {"gaps": []})
    monkeypatch.setattr(pipe, "generate_edits", lambda m, j, g, c, **kw: FakeEdits())
    monkeypatch.setattr(pipe, "apply_edits", lambda doc_id, edits, client: None)
    monkeypatch.setattr(pipe, "build_line_map", lambda pdf: object())
    monkeypatch.setattr(pipe, "enforce_layout",
                        lambda doc_id, edits, doc, master_map, client, **kw: (b"%PDF-final", []))
    monkeypatch.setattr(pipe, "ats_check_all", lambda *a, **k: SimpleNamespace(
        score=88, passed=True, issues=[], warnings=[]))
    monkeypatch.setattr(pipe, "DOCX_OUTPUT_DIR", str(tmp_path))
    for name in ("jd_get_token_usage", "engine_get_token_usage"):
        monkeypatch.setattr(pipe, name, lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    for name in ("jd_reset_token_usage", "engine_reset_token_usage"):
        monkeypatch.setattr(pipe, name, lambda: None)
    # Hermetic: never read the developer's real experience inventory.
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))


def test_pipeline_writes_master_resume_cache(tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    cache = tmp_path / "master_resume.txt"
    monkeypatch.setattr(fm, "MASTER_RESUME_CACHE", str(cache))
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(cache))

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    assert cache.read_text() == "resume text with GenAI keywords"


def test_pipeline_survives_non_oserror_cache_write_failure(tmp_path, monkeypatch):
    # The master-cache write is best-effort: a NON-OSError (e.g. a bad encode)
    # must be swallowed, not abort an otherwise-successful expensive run. Guards
    # against narrowing the handler back to `except OSError`.
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    cache = tmp_path / "master_resume.txt"
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(cache))

    real_open = open
    def exploding_open(file, *args, **kwargs):
        if str(file) == str(cache):
            raise RuntimeError("boom")   # non-OSError raised by the cache write
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr("builtins.open", exploding_open)

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    # Pipeline ran to completion despite the cache write blowing up.
    assert result["ats_score"] == 88
    assert result["google_doc_url"] == "https://docs.google.com/doc-123"
    assert not cache.exists()   # the write never succeeded


def test_pipeline_raises_on_cli_failure_and_never_marks_success(tmp_path, monkeypatch):
    """2026-07-15 final review: a Claude CLI failure mid-pipeline must abort the
    run, not complete as a mislabeled success (tailored_at set on an untailored
    copy, job silently removed from the auto-tailor gate). Step 2 runs the REAL
    extract_keywords against a dead CLI; everything around the LLM is stubbed."""
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    monkeypatch.setattr(pipe, "extract_keywords", te.extract_keywords)

    def dead_cli(*a, **k):
        # Timeout subclass: fail-fast (no retry sleeps), still a ClaudeCLIError.
        raise ClaudeCLITimeout("claude CLI killed: timed out after 120s")
    monkeypatch.setattr(te, "run_claude", dead_cli)

    with pytest.raises(ClaudeCLIError):
        pipe.run_tailor_pipeline(
            {"id": 1, "url": "https://x/1", "company": "Acme", "title": "PM"},
            db_path=str(db), progress=lambda m: None)

    conn = sqlite3.connect(str(db))
    tailored_at = conn.execute(
        "SELECT tailored_at FROM jobs WHERE id = 1").fetchone()[0]
    history_rows = conn.execute(
        "SELECT COUNT(*) FROM tailor_history").fetchone()[0]
    conn.close()
    assert tailored_at is None    # never recorded as a successful tailor
    assert history_rows == 0


def test_post_tailor_recompute_uses_stored_denominator(tmp_path, monkeypatch):
    # Stored extraction: 2 must-haves; final doc text ("resume text with GenAI
    # keywords") contains "GenAI" (alias of Generative AI) but not "Kubernetes".
    stored = fm.build_filter_json(
        [{"term": "Generative AI", "aliases": ["GenAI"], "present": False},
         {"term": "Kubernetes", "aliases": ["K8s"], "present": False}],
        ["Senior PM, AI"], "none",
        [{"requirement": "US work authorization", "verdict": "met"}],
        "close",
    )
    db = _seed_pipeline_db(tmp_path, filter_json=stored)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "cache.txt"))

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT filter_score, filter_score_master, filter_source, filter_json "
        "FROM jobs WHERE id = 1").fetchone()
    conn.close()
    # 1/2 present → 37.5 coverage; title 'close' (stored alignment, title not in
    # doc text) → 5; knockouts all met → 15. round(57.5) = 58.
    assert row["filter_score"] == 58
    assert row["filter_score_master"] == 40          # frozen
    assert row["filter_source"] == "tailored"
    detail = json.loads(row["filter_json"])
    assert [m["term"] for m in detail["must_haves"]] == ["Generative AI", "Kubernetes"]
    assert [m["present"] for m in detail["must_haves"]] == [True, False]


def test_post_tailor_recompute_skips_without_filter_json(tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path)   # filter fields all NULL
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "cache.txt"))

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT filter_score, filter_source FROM jobs WHERE id = 1").fetchone()
    conn.close()
    assert row == (None, None)


def _v2_blob(term="AI enablement", aliases=("AI tooling adoption",),
             verdict="explicit"):
    return json.dumps({
        "version": 2,
        "must_haves": [{"term": term, "aliases": list(aliases),
                        "verdict": verdict, "evidence": "e"}],
        "title_variants": [], "title_alignment": "none",
        "title_claim": "none", "knockouts": [], "uncapped_score": 90,
    })


def test_load_stored_must_haves_v2_v1_sentinel_null_and_errors(tmp_path):
    db = _seed_pipeline_db(tmp_path, filter_json=_v2_blob())
    got = pipe._load_stored_must_haves(str(db), 1)
    assert got[0]["term"] == "AI enablement"
    assert got[0]["verdict"] == "explicit"

    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute("UPDATE jobs SET filter_json = ? WHERE id = 1",
                     (json.dumps({"must_haves": [
                         {"term": "X", "aliases": [], "present": True}]}),))
    conn.close()
    got = pipe._load_stored_must_haves(str(db), 1)
    assert got == [{"term": "X", "aliases": [], "present": True}]

    for bad in ("{}", "not json", None):
        conn = sqlite3.connect(str(db))
        with conn:
            conn.execute("UPDATE jobs SET filter_json = ? WHERE id = 1", (bad,))
        conn.close()
        assert pipe._load_stored_must_haves(str(db), 1) is None

    assert pipe._load_stored_must_haves(str(tmp_path / "missing.db"), 1) is None


def test_pipeline_passes_stored_must_haves_and_inventory_to_engine(
        tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path, filter_json=_v2_blob())
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    monkeypatch.setattr(pipe, "load_inventory",
                        lambda: ("## HARD FACTS", "sha", "inventory"))
    seen = {}

    def fake_gap(master, jd, **kw):
        seen["gap"] = kw
        return {"gaps": []}

    def fake_edits(m, j, g, c, **kw):
        seen["edits"] = kw
        return FakeEdits()

    monkeypatch.setattr(pipe, "gap_analysis", fake_gap)
    monkeypatch.setattr(pipe, "generate_edits", fake_edits)

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "PM"},
        db_path=str(db), progress=lambda m: None)

    assert seen["gap"]["inventory_text"] == "## HARD FACTS"
    assert seen["gap"]["must_haves"][0]["verdict"] == "explicit"
    assert seen["edits"]["inventory_text"] == "## HARD FACTS"
    assert seen["edits"]["must_haves"][0]["term"] == "AI enablement"


def test_pipeline_does_not_pass_resume_fallback_as_inventory(
        tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    monkeypatch.setattr(pipe, "load_inventory",
                        lambda: ("resume text", "sha", "resume_fallback"))
    seen = {}

    def fake_gap(master, jd, **kw):
        seen["gap"] = kw
        return {"gaps": []}

    monkeypatch.setattr(pipe, "gap_analysis", fake_gap)

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "PM"},
        db_path=str(db), progress=lambda m: None)

    assert seen["gap"]["inventory_text"] == ""
    assert seen["gap"]["must_haves"] is None


def test_pipeline_warns_when_credited_term_missing_from_final_text(
        tmp_path, monkeypatch):
    # FakeClient's final text is "resume text with GenAI keywords" —
    # "agentification" is credited but absent, "Generative AI" hits via alias.
    blob = json.dumps({
        "version": 2,
        "must_haves": [
            {"term": "agentification", "aliases": [],
             "verdict": "evidenced", "evidence": "e"},
            {"term": "Generative AI", "aliases": ["GenAI"],
             "verdict": "explicit", "evidence": "e"},
        ],
        "title_variants": [], "title_alignment": "none",
        "title_claim": "none", "knockouts": [], "uncapped_score": 90,
    })
    db = _seed_pipeline_db(tmp_path, filter_json=blob)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "PM"},
        db_path=str(db), progress=lambda m: None)

    hits = [w for w in result["warnings"] if "agentification" in w]
    assert len(hits) == 1
    assert not any("Generative AI" in w for w in result["warnings"])


def test_pipeline_no_warning_when_all_credited_terms_present(
        tmp_path, monkeypatch):
    blob = _v2_blob(term="Generative AI", aliases=("GenAI",))
    db = _seed_pipeline_db(tmp_path, filter_json=blob)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "PM"},
        db_path=str(db), progress=lambda m: None)

    assert not any("must-have" in w for w in result["warnings"])


@dataclasses.dataclass
class RichEdits:
    """Edits with real content so verdicts have something to check."""
    title_line_replacement: str = "PRINCIPAL PRODUCT MANAGER"
    summary_replacement: str = "New summary about GenAI."
    skills_reorder: dict = dataclasses.field(default_factory=dict)
    experience_edits: list = dataclasses.field(default_factory=list)
    rewritten_bullets: list = dataclasses.field(default_factory=list)
    keyword_count: int = 5


def _setup_pipeline_mocks(monkeypatch, tmp_path, edits_factory,
                          final_text="resume text with GenAI keywords",
                          line_map_fails=False,
                          enforce_layout_fails=False):
    """The standard mock block, parameterized. Mirrors
    test_pipeline_returns_result_and_writes_history."""
    class Client(FakeClient):
        def extract_plain_text(self, doc):
            return final_text

    monkeypatch.setattr(pipe, "extract_jd", lambda url: SimpleNamespace(
        raw_text="JD text", company="Acme", title="Senior PM AI"))
    monkeypatch.setattr(pipe, "extract_keywords", lambda text: SimpleNamespace(
        exact_job_title="Senior PM, AI", priority_keywords=["genai"]))
    monkeypatch.setattr(pipe, "GoogleAPIClient", Client)
    monkeypatch.setattr(pipe, "gap_analysis", lambda master, jd, **kw: {"gaps": []})
    monkeypatch.setattr(pipe, "generate_edits",
                        lambda m, j, g, c, **kw: edits_factory())
    monkeypatch.setattr(pipe, "apply_edits", lambda doc_id, edits, client: None)
    if line_map_fails:
        def _boom(pdf):
            raise RuntimeError("no line map")
        monkeypatch.setattr(pipe, "build_line_map", _boom)
    else:
        monkeypatch.setattr(pipe, "build_line_map", lambda pdf: object())
    if enforce_layout_fails:
        def _guard_boom(doc_id, edits, doc, master_map, client, **kw):
            raise RuntimeError("guard boom")
        monkeypatch.setattr(pipe, "enforce_layout", _guard_boom)
    else:
        monkeypatch.setattr(pipe, "enforce_layout",
                            lambda doc_id, edits, doc, master_map, client, **kw: (b"%PDF", []))
    monkeypatch.setattr(pipe, "ats_check_all", lambda *a, **k: SimpleNamespace(
        score=88, passed=True, issues=[], warnings=[]))
    monkeypatch.setattr(pipe, "DOCX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE",
                        str(tmp_path / "master_resume.txt"))
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))
    monkeypatch.setattr(pipe, "jd_get_token_usage",
                        lambda: {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0})
    monkeypatch.setattr(pipe, "engine_get_token_usage",
                        lambda: {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0})
    monkeypatch.setattr(pipe, "jd_reset_token_usage", lambda: None)
    monkeypatch.setattr(pipe, "engine_reset_token_usage", lambda: None)


def _seed_job(db):
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, score, status, profile) "
        "VALUES ('Senior PM AI', 'Acme', 'https://x/1', 92, 'new', 'testuser')")
    conn.commit()
    conn.close()


def test_pipeline_persists_final_text_warnings_and_verdicts(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    _seed_job(db)
    # Title landed: final text contains the uppercased replacement.
    final = "PRINCIPAL PRODUCT MANAGER\nNew summary about GenAI.\nbody"
    _setup_pipeline_mocks(monkeypatch, tmp_path, RichEdits, final_text=final)

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db))

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    hist = conn.execute(
        "SELECT final_text, final_text_source, warnings_json, "
        "edit_verdicts_json FROM tailor_history").fetchone()
    job = conn.execute("SELECT tailor_issue_json FROM jobs WHERE id = 1").fetchone()
    conn.close()

    assert hist["final_text"] == final
    assert hist["final_text_source"] == "pipeline"
    warnings = json.loads(hist["warnings_json"])
    assert set(warnings) == {"layout", "must_have"}
    assert warnings["layout"] == []       # clean guard run → no layout warnings
    assert warnings["must_have"] == []    # nothing credited → no must-have gaps
    verdicts = json.loads(hist["edit_verdicts_json"])
    by = {v["section"]: v["verdict"] for v in verdicts}
    assert by["title"] == "landed" and by["summary"] == "landed"
    assert job["tailor_issue_json"] is None  # clean tailor → no badge


def test_pipeline_writes_issue_facts_on_problem_tailor(tmp_path, monkeypatch):
    # The title's "modified" verdict needs a non-empty master title line (the
    # before-text); it's "" in a neutral tree, which would score it not_landed.
    patch_pipeline_title_line(monkeypatch)
    db = tmp_path / "jobs.db"
    _seed_job(db)
    # Pre-set stale facts to prove the pipeline OVERWRITES per tailor.
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE jobs SET tailor_issue_json = '{\"reverted\": 9}' "
                     "WHERE id = 1")
    conn.close()
    # Final text contains NEITHER replacements NOR originals → modified…
    # …but line_map_fails=True makes layout_unverified trigger the facts.
    _setup_pipeline_mocks(monkeypatch, tmp_path, RichEdits,
                          final_text="unrelated resume body text",
                          line_map_fails=True)

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db))

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    job = conn.execute("SELECT tailor_issue_json FROM jobs WHERE id = 1").fetchone()
    conn.close()
    facts = json.loads(job["tailor_issue_json"])
    assert facts["layout_unverified"] is True
    assert facts["reverted"] == 1        # summary not_landed counts here
    assert facts["modified"] == 1        # title: neither side present
    assert result["issue_facts"] == facts


def test_pipeline_clears_stale_issue_facts_on_clean_retailor(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    _seed_job(db)
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE jobs SET tailor_issue_json = '{\"reverted\": 9}' "
                     "WHERE id = 1")
    conn.close()
    final = "PRINCIPAL PRODUCT MANAGER\nNew summary about GenAI.\nbody"
    _setup_pipeline_mocks(monkeypatch, tmp_path, RichEdits, final_text=final)

    pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db))

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    job = conn.execute("SELECT tailor_issue_json FROM jobs WHERE id = 1").fetchone()
    conn.close()
    assert job["tailor_issue_json"] is None


def test_pipeline_survives_verdict_computation_failure(tmp_path, monkeypatch):
    """Verdicts/facts are best-effort like every persistence step below them:
    a tailor that already spent both LLM calls and finished the doc work must
    never fail because the diagnostics crashed."""
    db = tmp_path / "jobs.db"
    _seed_job(db)
    final = "PRINCIPAL PRODUCT MANAGER\nNew summary about GenAI.\nbody"
    _setup_pipeline_mocks(monkeypatch, tmp_path, RichEdits, final_text=final)

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(pipe.tailor_diff, "compute_edit_verdicts", boom)

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db))

    assert result["google_doc_url"] == "https://docs.google.com/doc-123"
    assert result["issue_facts"] is None

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    hist = conn.execute(
        "SELECT final_text, edit_verdicts_json FROM tailor_history").fetchone()
    job = conn.execute("SELECT tailor_issue_json FROM jobs WHERE id = 1").fetchone()
    conn.close()
    assert hist["final_text"] == final    # ground truth still persisted
    assert hist["edit_verdicts_json"] == "[]"
    assert job["tailor_issue_json"] is None


def test_pipeline_flags_layout_unverified_when_guard_crashes(tmp_path, monkeypatch):
    """Pins the guard_crashed disjunct end-to-end: master map exists, but
    enforce_layout raises → layout_unverified facts on an otherwise-clean
    tailor (all edits landed, no missing terms)."""
    db = tmp_path / "jobs.db"
    _seed_job(db)
    final = "PRINCIPAL PRODUCT MANAGER\nNew summary about GenAI.\nbody"
    _setup_pipeline_mocks(monkeypatch, tmp_path, RichEdits, final_text=final,
                          enforce_layout_fails=True)

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db))

    assert result["google_doc_url"] == "https://docs.google.com/doc-123"

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    job = conn.execute("SELECT tailor_issue_json FROM jobs WHERE id = 1").fetchone()
    hist = conn.execute("SELECT warnings_json FROM tailor_history").fetchone()
    conn.close()
    facts = json.loads(job["tailor_issue_json"])
    assert facts["layout_unverified"] is True
    assert facts["reverted"] == 0         # edits all landed — layout is the only issue
    assert result["issue_facts"] == facts

    warnings = json.loads(hist["warnings_json"])
    assert "guard boom" in warnings["layout"][0]
    assert warnings["must_have"] == []


def test_pipeline_sets_engine_log_context(monkeypatch):
    """Each run stamps its job id into tailor_engine's thread-local log
    context BEFORE any LLM work, so concurrent workers' engine log lines
    ([extract_keywords], [layout_reshape], guard rounds) are attributable."""
    seen = {}

    def probe(url):
        seen["ctx"] = te.get_log_context()
        raise RuntimeError("stop after context is set")

    monkeypatch.setattr(pipe, "extract_jd", probe)
    monkeypatch.setattr(pipe, "_load_stored_must_haves", lambda db, jid: None)
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))
    monkeypatch.setattr(pipe, "jd_reset_token_usage", lambda: None)
    monkeypatch.setattr(pipe, "engine_reset_token_usage", lambda: None)

    with pytest.raises(RuntimeError):
        pipe.run_tailor_pipeline({"id": 15635, "url": "https://x/1"},
                                 db_path="/nonexistent")
    assert seen["ctx"] == 15635


# ---------------------------------------------------------------------------
# Stored-description fallback (2026-07-20): 9 LIVE Workday jobs failed Step 1
# during the overnight window although jobs.description already held their
# full sanitized JDs from the last scrape. Step 1 now falls back to that
# stored text when URL extraction fails — but ONLY for the generic
# JDExtractionError: JDPostingGoneError must keep propagating so auto-tailor's
# suspect/confirm dead-posting machinery still sees it (a provably-delisted
# job must never burn a full Opus tailor against stale text).
# ---------------------------------------------------------------------------
from resume_tailor.jd_extractor import JDExtractionError, JDPostingGoneError


def _set_description(db, text):
    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute("UPDATE jobs SET description = ? WHERE id = 1", (text,))
    conn.close()


_SUBSTANTIAL_JD = (
    "Own the AI roadmap. Partner with engineering and design. "
    "Ship LLM-powered features. Define metrics and drive adoption. " * 4
).strip()


def test_pipeline_falls_back_to_stored_description(tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    _set_description(db, _SUBSTANTIAL_JD)

    def _board_down(url):
        raise JDExtractionError("All extraction tiers failed")
    monkeypatch.setattr(pipe, "extract_jd", _board_down)

    seen = {}
    def fake_extract_keywords(text):
        seen["jd_text"] = text
        return SimpleNamespace(exact_job_title="Senior PM, AI",
                               priority_keywords=["genai"])
    monkeypatch.setattr(pipe, "extract_keywords", fake_extract_keywords)

    result = pipe.run_tailor_pipeline(
        {"id": 1, "url": "https://x/1", "company": "Acme", "title": "Senior PM AI"},
        db_path=str(db), progress=lambda m: None)

    # The tailor ran to completion against the stored text...
    assert result["google_doc_url"] == "https://docs.google.com/doc-123"
    assert seen["jd_text"] == _SUBSTANTIAL_JD
    # ...and the staleness risk is surfaced, not silent.
    assert any("stored description" in w.lower() for w in result["warnings"])


def test_pipeline_posting_gone_never_falls_back(tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    _set_description(db, _SUBSTANTIAL_JD)   # present, but must not be used

    def _gone(url):
        raise JDPostingGoneError("Posting gone (HTTP 404)", status=404)
    monkeypatch.setattr(pipe, "extract_jd", _gone)

    with pytest.raises(JDPostingGoneError):
        pipe.run_tailor_pipeline(
            {"id": 1, "url": "https://x/1", "company": "Acme",
             "title": "Senior PM AI"},
            db_path=str(db), progress=lambda m: None)


def test_pipeline_reraises_without_substantial_stored_description(
        tmp_path, monkeypatch):
    db = _seed_pipeline_db(tmp_path)
    _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    _set_description(db, "Apply now!")   # nav-snippet junk, not a real JD

    def _board_down(url):
        raise JDExtractionError("All extraction tiers failed")
    monkeypatch.setattr(pipe, "extract_jd", _board_down)

    with pytest.raises(JDExtractionError):
        pipe.run_tailor_pipeline(
            {"id": 1, "url": "https://x/1", "company": "Acme",
             "title": "Senior PM AI"},
            db_path=str(db), progress=lambda m: None)


def test_load_stored_description_substantial_short_missing(tmp_path):
    db = _seed_pipeline_db(tmp_path)
    # NULL description → None
    assert pipe._load_stored_description(str(db), 1) is None
    # At/below the 200-char bar (Tier 2's own minimum) → None
    _set_description(db, "x" * 200)
    assert pipe._load_stored_description(str(db), 1) is None
    # Substantial → returned stripped
    _set_description(db, "  " + _SUBSTANTIAL_JD + "\n")
    assert pipe._load_stored_description(str(db), 1) == _SUBSTANTIAL_JD
    # Missing row / missing DB → None, never raises
    assert pipe._load_stored_description(str(db), 999) is None
    assert pipe._load_stored_description(str(tmp_path / "missing.db"), 1) is None
