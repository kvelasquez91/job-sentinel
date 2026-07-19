"""Static pins for the Filter-column threshold filter (All/80+/60+/40+).

The owner approved the semantics 2026-07-17 (per the 2026-07-17
filter-score-filter design, private repo notes): the threshold gates on the
DISPLAYED value — Math.round(filter_score) — with KO rows
(filter_knockout === 1) and unscored rows (filter_score == null: gate
chips and un-judged '—') always hidden while a threshold is active.
DOM structure is asserted via BeautifulSoup (bs4 is already a project
dependency); JS logic via regex pins, the pattern
test_dashboard_default_sort.py established (bs4 can't parse script
bodies).
"""
import os
import re

from bs4 import BeautifulSoup

_INDEX = os.path.join(os.path.dirname(__file__), "..",
                      "dashboard", "static", "index.html")


def _html():
    with open(_INDEX, encoding="utf-8") as f:
        return f.read()


# --- DOM structure (bs4) ---

def test_filter_score_button_group_structure():
    soup = BeautifulSoup(_html(), "html.parser")
    group = soup.find(id="filter-score-filters")
    assert group is not None, "#filter-score-filters group missing from filter bar"
    buttons = group.find_all("button")
    labels = [b.get_text(strip=True) for b in buttons]
    assert labels == ["All", "80+", "60+", "40+"], labels
    for b in buttons:
        assert "setFilterScoreFilter(this" in b.get("onclick", ""), (
            f"button {b.get_text(strip=True)!r} not wired to setFilterScoreFilter")
    actives = [b.get_text(strip=True) for b in buttons
               if "active" in (b.get("class") or [])]
    assert actives == ["All"], "exactly the All button starts active"


def test_filter_group_sits_right_after_score_group():
    soup = BeautifulSoup(_html(), "html.parser")
    bar = soup.find(id="filter-bar")
    assert bar is not None
    group_ids = [d.get("id") for d in bar.find_all("div", class_="filter-group")]
    assert group_ids.index("filter-score-filters") == group_ids.index("score-filters") + 1
    label_texts = [s.get_text(strip=True)
                   for s in bar.find_all("span", class_="filter-label")]
    assert label_texts[:2] == ["Score", "Filter"], label_texts


# --- JS logic (regex pins) ---

def test_state_default_is_all():
    m = re.search(r"^\s*filterScoreFilter:\s*'(\w*)',", _html(), re.MULTILINE)
    assert m, "state.filterScoreFilter default not found"
    assert m.group(1) == "", "default must be '' (All)"


def test_apply_filters_gates_on_displayed_value():
    html = _html()
    assert "if (state.filterScoreFilter !== '')" in html, (
        "applyFilters must have a filterScoreFilter clause")
    pat = re.compile(
        r"j\.filter_knockout !== 1 && j\.filter_score != null\s*"
        r"&& Math\.round\(j\.filter_score\) >= parseInt\(state\.filterScoreFilter\)")
    assert pat.search(html), (
        "clause must keep all three guards — KO, null, rounded >= threshold — "
        "so no threshold ever surfaces a KO/gated/un-judged row")


def test_setter_scopes_active_sweep_to_own_group():
    m = re.search(r"function setFilterScoreFilter\(btn, value\) \{(.*?)\n\}",
                  _html(), re.DOTALL)
    assert m, "setFilterScoreFilter function missing"
    body = m.group(1)
    assert "#filter-score-filters .filter-btn" in body, (
        "active-class sweep must be scoped to #filter-score-filters "
        "(must not deactivate the Score group's buttons)")
    assert "state.filterScoreFilter = value;" in body
    assert "applyFilters();" in body
