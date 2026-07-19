"""The $300K location exception is documented on every prompt surface so the
scoring LLM stays in sync with salary_rules' LOCATION_EXCEPTION_MIN_COMP."""
import os

import pytest
import yaml

from engine.llm_scorer import _FIT_SYSTEM_TEMPLATE
from salary_rules import LOCATION_EXCEPTION_MIN_COMP

_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def test_rubric_dimension_3_has_high_comp_band():
    import profile_policy
    configured_block = (
        (profile_policy._load_config().get("policy") or {}).get("rubric") or {}
    ).get("remote_location_block")
    if configured_block is not None and "%%EXCEPTION_COMP%%" not in configured_block:
        pytest.skip("custom remote_location_block opts out of the relocation-exception band")
    assert f"${LOCATION_EXCEPTION_MIN_COMP:,}" in _FIT_SYSTEM_TEMPLATE


def test_config_resume_summary_targets_high_comp_onsite():
    """Owner-lint: opt in with policy.owner_pins.summary_mentions_relocation to
    assert your resume_summary literally names your relocation bar; the neutral
    default leaves it off, so a personalized fork whose summary doesn't mention
    that figure isn't flagged red."""
    import profile_policy
    if not profile_policy.OWNER_SUMMARY_LINT:
        pytest.skip("owner-lint not enabled (policy.owner_pins.summary_mentions_relocation)")
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)
    rs = (cfg.get("llm_scoring") or {}).get("resume_summary") or ""
    if not rs.strip():
        pytest.skip("no personalized resume_summary configured (shared/fresh checkout)")
    assert f"${LOCATION_EXCEPTION_MIN_COMP // 1000}K" in rs
