"""Static pins for the Added-view grouping control (Run / Day).

The owner approved the semantics 2026-07-17 (per the 2026-07-17
grouping-control design, private repo notes): state.groupBy ('run' default)
switches both the 'added' sort comparator and the divider block between
run grouping (today's behavior, unchanged) and local-calendar-day
grouping of created_at (formatAddedDate's normalization; -1 sentinel
groups legacy rows at the oldest end); within a day, best match first;
choosing a grouping from any other sort jumps to the Added view. DOM
structure is asserted via BeautifulSoup; JS logic via regex pins
(the test_dashboard_default_sort.py pattern — bs4 can't parse script
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

def test_group_by_control_structure():
    soup = BeautifulSoup(_html(), "html.parser")
    group = soup.find(id="group-by-filters")
    assert group is not None, "#group-by-filters control missing from filter bar"
    buttons = group.find_all("button")
    labels = [b.get_text(strip=True) for b in buttons]
    assert labels == ["Run", "Day"], labels
    for b in buttons:
        assert "setGroupBy(this" in b.get("onclick", ""), (
            f"button {b.get_text(strip=True)!r} not wired to setGroupBy")
    actives = [b.get_text(strip=True) for b in buttons
               if "active" in (b.get("class") or [])]
    assert actives == ["Run"], "Run must be the default-active grouping"


def test_group_control_sits_between_dismissed_and_search():
    soup = BeautifulSoup(_html(), "html.parser")
    bar = soup.find(id="filter-bar")
    assert bar is not None
    ids = [el.get("id") for el in bar.find_all(True) if el.get("id")]
    assert "group-by-filters" in ids, "#group-by-filters missing from the bar"
    assert (ids.index("show-dismissed-btn") < ids.index("group-by-filters")
            < ids.index("search-input")), (
        "Group control must sit between the Dismissed button and the search input")


# --- JS logic (regex pins) ---

def test_state_default_groups_by_run():
    m = re.search(r"^\s*groupBy:\s*'(\w+)',", _html(), re.MULTILINE)
    assert m, "state.groupBy default not found"
    assert m.group(1) == "run", "default grouping must be 'run' (today's behavior)"


def test_added_day_key_normalizes_and_caches():
    m = re.search(r"function addedDayKey\(job\) \{(.*?)\n\}", _html(), re.DOTALL)
    assert m, "addedDayKey function missing"
    body = m.group(1)
    assert "replace(' ', 'T') + 'Z'" in body, (
        "must normalize SQLite UTC timestamps exactly like formatAddedDate")
    assert "_dayKey" in body, "must cache the key on the job object"
    assert "-1" in body, "missing/unparseable created_at needs the -1 sentinel"


def test_added_sort_branches_on_group_mode():
    m = re.search(r"case 'added': \{(.*?)\n      \}", _html(), re.DOTALL)
    assert m, "'added' sort case missing"
    body = m.group(1)
    assert "state.groupBy === 'day'" in body, "sort must branch on day grouping"
    assert body.count("addedDayKey(") >= 2, (
        "day branch must compare addedDayKey of both jobs")
    assert body.count("effectiveScore(b) - effectiveScore(a)") == 2, (
        "both branches tiebreak by effective score (best match first)")
    assert "a.run_id ?? -1" in body, "run branch must remain intact"


def test_divider_block_picks_key_and_label_by_mode():
    html = _html()
    assert re.search(r"const byDay = state\.groupBy === 'day'", html), (
        "renderTable must compute the grouping mode")
    assert re.search(
        r"groupKeyOf = j => byDay \? addedDayKey\(j\) : \(j\.run_id \?\? 'none'\)",
        html), "divider key extractor must switch by mode"
    assert re.search(
        r"byDay \? dayGroupLabel\(job, groupCounts\[groupKey\]\)"
        r"\s*:\s*sessionLabel\(job\.run_id, groupCounts\[groupKey\]\)",
        html), "divider label must switch by mode"
    assert "function dayGroupLabel(job, count)" in html, "dayGroupLabel missing"


def test_set_group_by_scopes_sweep_and_jumps_to_added():
    m = re.search(r"function setGroupBy\(btn, value\) \{(.*?)\n\}", _html(), re.DOTALL)
    assert m, "setGroupBy function missing"
    body = m.group(1)
    assert "#group-by-filters .filter-btn" in body, (
        "active-class sweep must be scoped to #group-by-filters "
        "(must not deactivate other button groups)")
    assert "state.groupBy = value;" in body
    assert "state.sortBy !== 'added'" in body and "state.sortBy = 'added'" in body, (
        "choosing a grouping from another sort must jump to the Added view")
    assert "state.sortDir = 'desc';" in body
    assert "updateSortArrows();" in body
    assert "applyFilters();" in body
