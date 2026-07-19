"""The runs-table above-threshold metric must count blended, KO-gated scores."""
import main as main_mod


def test_counts_blended_scores_with_knockout_gate(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    rows = [
        ("A", "uA", 70, 0, 1),   # above threshold, no KO -> counts
        ("B", "uB", 70, 1, 1),   # KO gates 70 down to 40 -> excluded
        ("C", "uC", 59, 0, 1),   # below threshold -> excluded
        ("D", "uD", 90, 0, 2),   # different run -> excluded
    ]
    for title, url, score, ko, run in rows:
        conn.execute(
            "INSERT INTO jobs (title, company, url, score, filter_knockout, "
            "run_id, profile) VALUES (?, 'C', ?, ?, ?, ?, 'testuser')",
            (title, url, score, ko, run))
    conn.commit()
    assert main_mod.count_effective_above_threshold(conn, 1, 60) == 1
