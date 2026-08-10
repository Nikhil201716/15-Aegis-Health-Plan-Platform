"""
build_warehouse.py
---------------------
DuckDB warehouse for Aegis, with the conformed member-month grain the
semantic layer depends on.

The grain declaration is the load-bearing part. `vw_member_month` is
exactly one row per member per enrolled month, carrying that month's
premium and that month's claims. Every ratio metric is defined against it,
which prevents the most common and least visible error in payer
analytics: a numerator aggregated at claim grain divided by a denominator
aggregated at member grain. That produces a loss ratio wrong by whatever
the average claims-per-member happens to be, and it looks entirely
plausible.

Loads are CREATE OR REPLACE and idempotency is asserted rather than
assumed - running the build twice must not change a single figure.

Output: database/aegis.duckdb, reports/warehouse_summary.json
"""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "platform" / "ingest"))
from domain import N_MONTHS  # noqa: E402

DATA = ROOT / "data"
DB_DIR = ROOT / "database"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "aegis.duckdb"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# One row per WHAT? Written down so it is reviewable.
GRAIN = {
    "dim_member": "one row per member",
    "dim_provider": "one row per provider (NPI)",
    "fct_enrollment_span": "one row per member (their single enrolment span)",
    "fct_claim_line": "one row per adjudicated claim LINE",
    "vw_member_month": "one row per member per enrolled month - the conformed "
                       "grain every ratio metric is defined against",
    "vw_provider_coding": "one row per provider per month, professional claims only",
}


def build(con):
    con.execute(f"""
        CREATE OR REPLACE TABLE dim_member AS
        SELECT * FROM read_csv_auto('{(DATA / "members.csv").as_posix()}');

        CREATE OR REPLACE TABLE dim_provider AS
        SELECT * FROM read_csv_auto('{(DATA / "providers.csv").as_posix()}');

        CREATE OR REPLACE TABLE fct_enrollment_span AS
        SELECT * FROM read_csv_auto('{(DATA / "enrollment_spans.csv").as_posix()}');

        CREATE OR REPLACE TABLE fct_claim_line AS
        SELECT * FROM read_csv_auto(
            '{(DATA / "claim_lines.csv").as_posix()}',
            types={{'em_code': 'VARCHAR'}});
    """)

    # A month spine, so a member-month with ZERO claims still exists. This
    # is the difference between "average cost of members who claimed" and
    # PMPM, and conflating them overstates cost by roughly the share of
    # members who used nothing that month.
    con.execute(f"""
        CREATE OR REPLACE TABLE dim_month AS
        SELECT range AS month_index FROM range(0, {N_MONTHS});
    """)

    con.execute("""
        CREATE OR REPLACE VIEW vw_member_month AS
        WITH exposure AS (
            SELECT s.member_id, m.month_index
            FROM fct_enrollment_span s
            JOIN dim_month m
              ON m.month_index >= s.start_month AND m.month_index < s.end_month
        ),
        claims AS (
            SELECT member_id, month_index,
                   SUM(paid_amount)                                   AS paid_amount,
                   SUM(allowed_amount)                                AS allowed_amount,
                   COUNT(*)                                           AS total_lines,
                   SUM(CASE WHEN status = 'denied' THEN 1 ELSE 0 END) AS denied_lines
            FROM fct_claim_line GROUP BY 1, 2
        )
        SELECT e.member_id, e.month_index,
               printf('%04d-%02d', 2024 + (e.month_index / 12)::INT,
                      (e.month_index % 12) + 1)              AS month,
               d.region, d.plan, d.product, d.channel, d.age_band, d.sex,
               d.chronic_count, d.premium_burden,
               d.monthly_premium                             AS premium_collected,
               COALESCE(c.paid_amount, 0)                    AS paid_amount,
               COALESCE(c.allowed_amount, 0)                 AS allowed_amount,
               COALESCE(c.total_lines, 0)                    AS total_lines,
               COALESCE(c.denied_lines, 0)                   AS denied_lines
        FROM exposure e
        JOIN dim_member d USING (member_id)
        LEFT JOIN claims c ON c.member_id = e.member_id
                          AND c.month_index = e.month_index;
    """)

    # Provider coding intensity, the surface the fraud study works on.
    con.execute("""
        CREATE OR REPLACE VIEW vw_provider_coding AS
        SELECT c.provider_id, p.specialty, p.region, c.month_index,
               COUNT(*)                                     AS n_professional,
               AVG(CASE c.em_code WHEN '99212' THEN 0 WHEN '99213' THEN 1
                                  WHEN '99214' THEN 2 WHEN '99215' THEN 3 END)
                                                            AS mean_intensity,
               SUM(c.encounter_minutes)                     AS total_minutes,
               SUM(c.allowed_amount)                        AS allowed_amount
        FROM fct_claim_line c
        JOIN dim_provider p USING (provider_id)
        WHERE c.service_category = 'professional' AND c.em_code IS NOT NULL
        GROUP BY 1, 2, 3, 4;
    """)


def main():
    con = duckdb.connect(str(DB_PATH))
    build(con)

    # idempotency: build twice, assert nothing moved
    before = con.execute("SELECT COUNT(*), ROUND(SUM(paid_amount), 2) "
                         "FROM vw_member_month").fetchone()
    build(con)
    after = con.execute("SELECT COUNT(*), ROUND(SUM(paid_amount), 2) "
                        "FROM vw_member_month").fetchone()
    idempotent = before == after

    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in sorted(GRAIN)}

    # The check that catches a fan-out: member-months must equal the sum of
    # enrolment durations. If a join duplicated rows this diverges, and every
    # ratio metric silently gains a denominator it should not have.
    expected_mm = con.execute(
        "SELECT SUM(end_month - start_month) FROM fct_enrollment_span "
        "WHERE end_month > start_month").fetchone()[0]
    actual_mm = counts["vw_member_month"]

    zero_claim_share = con.execute(
        "SELECT ROUND(AVG(CASE WHEN total_lines = 0 THEN 1.0 ELSE 0 END), 4) "
        "FROM vw_member_month").fetchone()[0]

    summary = {
        "grain": GRAIN,
        "row_counts": counts,
        "idempotent_rebuild": bool(idempotent),
        "member_month_fanout_check": {
            "expected_from_spans": int(expected_mm),
            "actual_in_view": int(actual_mm),
            "matches": bool(int(expected_mm) == int(actual_mm)),
            "why": ("if these diverge a join has fanned out and every ratio "
                    "metric has gained denominator rows that do not exist"),
        },
        "member_months_with_zero_claims": zero_claim_share,
        "note": ("the month spine keeps zero-claim member-months in the "
                 "denominator; dropping them turns PMPM into 'average cost "
                 "among members who claimed', which is a different and much "
                 "larger number"),
    }
    with open(REPORTS / "warehouse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    con.close()

    print("WAREHOUSE")
    for t in sorted(GRAIN):
        print(f"  {t:<24} {counts[t]:>10,}   {GRAIN[t]}")
    fc = summary["member_month_fanout_check"]
    print(f"\n  fan-out check   expected {fc['expected_from_spans']:,} "
          f"member-months, got {fc['actual_in_view']:,} -> {fc['matches']}")
    print(f"  idempotent      {idempotent}")
    print(f"  zero-claim member-months {zero_claim_share:.1%} "
          f"(kept in the denominator on purpose)")


if __name__ == "__main__":
    main()
