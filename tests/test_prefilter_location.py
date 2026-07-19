"""Location gate: non-remote, non-local-area jobs are pre-filtered before
any LLM call (cap 10 <= 15 means _score_one_job hard-skips the CLI)."""
import engine.llm_scorer as llm_scorer
from engine.llm_scorer import prefilter_job
from local_area import build_local_area_regex
from policy_fixtures import patch_pm_prefilter, patch_relocation_exception

_PM_TITLE = "Senior Product Manager, AI Platforms"
_DESC = "Lead our AI product roadmap. " * 30  # no remote words

# Explicit fixture cities AND an explicit state pattern so local-area tests
# are config-agnostic: the share package ships config.yaml with
# `local_locations: []` (and an empty local_state_pattern), which makes the
# module-level _LOCAL_AREA_RE None at import time. Patching the compiled
# matcher keeps the gate mechanics tested regardless of the owner's config —
# an explicit state_pattern is required here (not just explicit cities),
# since a blank ambient pattern disables matching outright (Task 3).
_FIXTURE_LOCAL_RE = build_local_area_regex(
    ("Springfield", "Riverton", "Fairview"), state_pattern=r"illinois|il\b")


def _cap(location, description=_DESC, title=_PM_TITLE):
    cap, _reason = prefilter_job(title, location, description)
    return cap


def test_named_city_state_with_no_remote_signal_is_gated():
    assert _cap("Tarrytown, NY") == 10
    assert _cap("Aventura, FL") == 10


def test_non_us_location_with_no_remote_signal_is_gated():
    assert _cap("London") == 10


def test_remote_in_location_passes():
    assert _cap("New York, NY (Remote)") is None
    assert _cap("Remote") is None


def test_remote_in_title_passes():
    """Regression (job 7603): '(Remote)' in the title is an explicit employer
    remote signal — the location gate must not cap on the city alone."""
    assert _cap("New York, NY", title=_PM_TITLE + " (Remote)") is None


def test_remote_in_description_passes():
    assert _cap("Tarrytown, NY", description=_DESC + " This role is remote.") is None


def test_hybrid_passes_to_knockout_net():
    # Hybrid may be local-hybrid or negotiable — the knockout gate
    # (stage 2) is the authoritative net for hidden onsite requirements.
    assert _cap("Charlotte, NC (Hybrid)") is None


def test_local_area_passes(monkeypatch):
    monkeypatch.setattr(llm_scorer, "_LOCAL_AREA_RE", _FIXTURE_LOCAL_RE)
    assert _cap("Springfield, IL") is None
    assert _cap("Riverton, IL") is None


def test_broad_or_empty_locations_pass():
    assert _cap("United States") is None
    assert _cap("") is None


def test_non_local_area_city_is_gated(monkeypatch):
    monkeypatch.setattr(llm_scorer, "_LOCAL_AREA_RE", _FIXTURE_LOCAL_RE)
    assert _cap("Chicago, IL") == 10


def test_more_severe_title_rules_still_win(monkeypatch):
    # A nurse posting in Tarrytown reports the non-PM reason (cap 5), not
    # the location reason — title rules run first.
    patch_pm_prefilter(monkeypatch)
    cap, reason = prefilter_job("Registered Nurse", "Tarrytown, NY", _DESC)
    assert cap == 5
    assert "non-PM" in reason


# ---------------------------------------------------------------------------
# $300K location exception (2026-07-13 spec): posted top-of-band >= $300K
# exempts a US onsite job from the location gate. Salary params are the
# CALLER's job (DB fields with regex fallback) — prefilter_job itself never
# reads dollars out of the description.
# ---------------------------------------------------------------------------


def test_high_comp_us_city_passes(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    cap, _ = prefilter_job(_PM_TITLE, "San Francisco, CA", _DESC,
                           salary_min=305_000, salary_max=385_000)
    assert cap is None


def test_high_comp_top_of_band_only_passes(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    # $225K-$468K: floor under $300K but the band reaches it — qualifies.
    cap, _ = prefilter_job(_PM_TITLE, "Sunnyvale, CA", _DESC,
                           salary_min=225_000, salary_max=468_000)
    assert cap is None


def test_high_comp_non_us_still_gated():
    cap, _ = prefilter_job(_PM_TITLE, "London", _DESC,
                           salary_min=350_000, salary_max=400_000)
    assert cap == 10


def test_high_comp_unlisted_foreign_city_still_gated():
    # Regression (jobs 4154/4316): Ottawa is absent from NON_US_LOCATIONS but
    # ", ON" trips _CITY_STATE, so the gate fires — the $300K exception must
    # not rescue a location with no positive US evidence.
    cap, _ = prefilter_job(_PM_TITLE, "Ottawa, ON, CA", _DESC,
                           salary_min=350_000, salary_max=400_000)
    assert cap == 10
    cap, _ = prefilter_job(_PM_TITLE, "Montreal, QC", _DESC,
                           salary_min=350_000, salary_max=400_000)
    assert cap == 10


def test_low_comp_us_city_still_gated(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    cap, _ = prefilter_job(_PM_TITLE, "San Francisco, CA", _DESC,
                           salary_min=180_000, salary_max=250_000)
    assert cap == 10


def test_high_comp_does_not_rescue_title_rules(monkeypatch):
    # A $400K nurse posting still dies on the non-PM title rule (step 1).
    patch_pm_prefilter(monkeypatch)
    cap, reason = prefilter_job("Registered Nurse", "Tarrytown, NY", _DESC,
                                salary_min=400_000, salary_max=500_000)
    assert cap == 5
    assert "non-PM" in reason


def test_prefilter_reads_params_not_description_dollars():
    # Design contract: the regex fallback lives at the CALL SITES, applied
    # per-field only when a DB field is NULL (the id-12747 lesson — junk
    # in-description figures must not override posted fields). With no
    # salary params, in-text dollars change nothing.
    rich = _DESC + " Total compensation $450,000 - $600,000 plus equity."
    cap, _ = prefilter_job(_PM_TITLE, "Austin, TX", rich)
    assert cap == 10
