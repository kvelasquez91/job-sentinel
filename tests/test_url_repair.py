"""repair_legacy_job_urls: rewrite URLs stored by pre-2026-07-20 scrapers.

Two scraper bugs shipped broken job URLs (fixed the same day):
  - smartrecruiters stored the API self-link
    (api.smartrecruiters.com/v1/companies/X/postings/Y) instead of the
    public page (jobs.smartrecruiters.com/X/Y);
  - workday omitted the career-site segment (/en-US/job/... instead of
    /{site_path}/job/...), which Workday answers with "page not found".

The fixed scrapers emit the correct formats, but ingest dedups by URL — so
without this repair every existing workday/smartrecruiters row re-ingests
as a DUPLICATE on the first post-upgrade run. The owner DB was hand-fixed;
this migration is what public-repo users rely on."""
import main as main_mod

TENANTS = [
    {"company": "Adobe", "tenant_url": "adobe.wd5.myworkdayjobs.com",
     "company_slug": "adobe", "site_path": "external_experienced"},
    {"company": "Mastercard", "tenant_url": "mastercard.wd1.myworkdayjobs.com",
     "company_slug": "mastercard", "site_path": "CorporateCareers"},
]


def _db(tmp_path):
    return main_mod.init_database(str(tmp_path / "jobs.db"))


def _insert(conn, url, source):
    conn.execute(
        "INSERT INTO jobs (title, company, url, source, profile, status) "
        "VALUES ('T', 'Co', ?, ?, 'p', 'new')", (url, source))
    conn.commit()


def _url_of(conn, like):
    return conn.execute(
        "SELECT url FROM jobs WHERE url LIKE ?", (like,)).fetchone()


def test_smartrecruiters_api_urls_rewritten_to_public_page(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, "https://api.smartrecruiters.com/v1/companies/BoschGroup"
                  "/postings/744000138262910", "smartrecruiters")
    main_mod.repair_legacy_job_urls(conn, TENANTS)
    assert _url_of(conn, "%smartrecruiters%")[0] == (
        "https://jobs.smartrecruiters.com/BoschGroup/744000138262910")


def test_workday_siteless_urls_rewritten_only_for_configured_tenants(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, "https://adobe.wd5.myworkdayjobs.com/en-US/job/San-Jose"
                  "/PM_R1", "workday")
    _insert(conn, "https://mastercard.wd1.myworkdayjobs.com/en-US/job/OFallon"
                  "/VP_R-2", "workday")
    # Tenant absent from config: no site token known -> must stay untouched.
    _insert(conn, "https://micron.wd1.myworkdayjobs.com/en-US/job/Boise"
                  "/PM_R3", "workday")
    main_mod.repair_legacy_job_urls(conn, TENANTS)
    assert _url_of(conn, "%adobe%")[0] == (
        "https://adobe.wd5.myworkdayjobs.com/external_experienced"
        "/job/San-Jose/PM_R1")
    assert _url_of(conn, "%mastercard%")[0] == (
        "https://mastercard.wd1.myworkdayjobs.com/CorporateCareers"
        "/job/OFallon/VP_R-2")
    assert _url_of(conn, "%micron%")[0] == (
        "https://micron.wd1.myworkdayjobs.com/en-US/job/Boise/PM_R3")


def test_collision_with_existing_fixed_row_leaves_legacy_row(tmp_path):
    """url is UNIQUE: if the fixed-format URL already exists as its own row,
    the legacy row stays as-is — a migration must never delete or clobber
    rows that may carry user state (status, notes)."""
    conn = _db(tmp_path)
    _insert(conn, "https://adobe.wd5.myworkdayjobs.com/external_experienced"
                  "/job/San-Jose/PM_R1", "workday")
    _insert(conn, "https://adobe.wd5.myworkdayjobs.com/en-US/job/San-Jose"
                  "/PM_R1", "workday")
    main_mod.repair_legacy_job_urls(conn, TENANTS)  # must not raise
    urls = {r[0] for r in conn.execute("SELECT url FROM jobs")}
    assert urls == {
        "https://adobe.wd5.myworkdayjobs.com/external_experienced"
        "/job/San-Jose/PM_R1",
        "https://adobe.wd5.myworkdayjobs.com/en-US/job/San-Jose/PM_R1",
    }


def test_repair_is_idempotent(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, "https://api.smartrecruiters.com/v1/companies/X/postings/9",
            "smartrecruiters")
    _insert(conn, "https://adobe.wd5.myworkdayjobs.com/en-US/job/SJ/PM_R1",
            "workday")
    main_mod.repair_legacy_job_urls(conn, TENANTS)
    first = sorted(r[0] for r in conn.execute("SELECT url FROM jobs"))
    main_mod.repair_legacy_job_urls(conn, TENANTS)
    second = sorted(r[0] for r in conn.execute("SELECT url FROM jobs"))
    assert first == second


def test_no_tenants_config_still_repairs_smartrecruiters(tmp_path):
    conn = _db(tmp_path)
    _insert(conn, "https://api.smartrecruiters.com/v1/companies/X/postings/9",
            "smartrecruiters")
    main_mod.repair_legacy_job_urls(conn, [])
    assert _url_of(conn, "%smartrecruiters%")[0] == (
        "https://jobs.smartrecruiters.com/X/9")
