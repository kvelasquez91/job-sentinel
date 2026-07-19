"""profile_policy is a pure loader: config.yaml policy values with generic defaults."""
import os
import re

import pytest

import profile_policy
import local_area

_PII_GATE = os.path.join(os.path.dirname(__file__), "..", "scripts", "pii_gate.sh")


def _gate_terms():
    """Parse scripts/pii_gate.sh's TIER1/TIER2 regex lines into a flat list of
    literal words/phrases, dropping regex-only fragments (the phone pattern,
    the doc-id hash) that can never appear verbatim in prose. A term survives
    the filter only if it is nothing but lowercase letters, digits, and
    spaces -- every real identity/location term in the gate is shaped like
    that; the phone pattern and the doc id are not (regex metacharacters /
    mixed-case hash characters)."""
    with open(_PII_GATE) as f:
        src = f.read()
    terms = []
    for tier in ("TIER1", "TIER2"):
        m = re.search(rf"^{tier}='([^']*)'", src, re.M)
        assert m, f"{tier} not found in pii_gate.sh"
        for term in m.group(1).split("|"):
            if re.fullmatch(r"[a-z0-9 ]+", term):
                terms.append(term)
    assert terms, "no forbidden terms parsed from pii_gate.sh"
    return terms


def test_local_cities_read_from_config():
    import yaml, os
    _CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    assert [c.lower() for c in profile_policy.LOCAL_CITIES] == [
        str(c).lower() for c in (cfg.get("local_locations") or [])
    ]


def test_local_area_aliases_profile_policy():
    assert local_area.LOCAL_COMMUTER_CITIES is profile_policy.LOCAL_CITIES


def test_defaults_table_comp_values():
    """Pin the DEFAULTS (not the resolved constants — those read ambient config)."""
    d = profile_policy._DEFAULTS
    assert d["policy.comp.target"] == 220_000
    assert d["policy.comp.local_full"] == 150_000
    assert d["policy.comp.relocation_exception"] == 300_000


def test_defaults_are_generic():
    """No owner residue in the defaults table: empty lists/patterns, neutral prose.

    These structural assertions run unconditionally in any tree. The
    forbidden-terms sweep below is read at RUNTIME from scripts/pii_gate.sh
    rather than spelled out here (this file lives under tests/, the exact
    tree the public-repo owner-residue grep sweeps, so writing the owner's
    name/city literally would reintroduce the residue this guard exists to
    catch) -- and that script is dropped before the public mirror is cut, so
    this half of the test only ever runs in the owner tree. There it is
    STRONGER than a hardcoded handful: every tier-1 identity term and every
    tier-2 location term the gate knows about, not just four."""
    d = profile_policy._DEFAULTS
    assert d["profile.key"] == "default"
    assert d["policy.keywords.primary"] == []
    assert d["policy.title_gates.baseline"] == []
    assert d["policy.hn.role_pattern"] == ""
    assert d["policy.geography.local_state_pattern"] == ""

    if not os.path.exists(_PII_GATE):
        pytest.skip("scripts/pii_gate.sh absent (dropped before the public "
                     "mirror is cut) -- the runtime forbidden-terms sweep "
                     "only applies in the owner tree")

    joined = " ".join(str(v) for v in d.values()).lower()
    for term in _gate_terms():
        assert term not in joined, term


def test_resolver_prefers_config_over_default():
    assert profile_policy._resolve(
        {"policy": {"comp": {"target": 111}}}, "policy.comp.target") == 111
    assert profile_policy._resolve({}, "policy.comp.target") == 220_000


def test_empty_pattern_compiles_to_none():
    assert profile_policy._compile_or_none("") is None
    assert isinstance(profile_policy._compile_or_none(r"\bx\b"), re.Pattern)
