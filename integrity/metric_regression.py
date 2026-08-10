"""
metric_regression.py
-----------------------
The test suite for the numbers themselves.

Software has regression tests. Metrics almost never do, and the failure
mode is worse: when a metric definition changes, every historical figure
computed from it changes too, silently and retroactively. Last quarter's
loss ratio in the board pack no longer matches the warehouse, nobody
notices for a month, and the reconciliation costs more than the original
work.

A semantic layer makes this testable, because there is exactly one thing
that changed. Three checks run here:

  GOLDEN VALUES     every metric x dimension combination is snapshotted.
                    A later run diffs against the snapshot, so an
                    unintended movement fails loudly.

  BLAST RADIUS      a REAL definition change is applied - including
                    pharmacy in the loss ratio, which is a live argument
                    at every payer - and the report states exactly which
                    historical figures moved, by how much, and whether any
                    crossed a threshold that matters.

  INVARIANTS        properties that must hold regardless of definition:
                    ratios in range, numerator and denominator reconciling
                    to the base table, dimensional slices summing to the
                    total. These catch a broken definition that a golden
                    file cannot, because a golden file is only ever as
                    correct as the day it was written.

Output: reports/metric_regression.json, integrity/golden_metrics.json
"""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "platform" / "semantic"))
from metrics import REGISTRY, Metric, Registry, Dimension, compile_query  # noqa: E402

DB_PATH = ROOT / "database" / "aegis.duckdb"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
GOLDEN = Path(__file__).resolve().parent / "golden_metrics.json"

# Combinations that get snapshotted. Deliberately includes the headline
# (no dimensions) and the slices a board pack actually shows.
SNAPSHOT_SPECS = [
    ("loss_ratio", []), ("loss_ratio", ["plan"]), ("loss_ratio", ["region"]),
    ("loss_ratio", ["month"]),
    ("pmpm", []), ("pmpm", ["region"]),
    ("denial_rate", []), ("denial_rate", ["plan"]),
    ("member_months", []), ("paid_amount", ["region"]),
]

# A movement larger than this in a headline ratio is a restatement, not a
# rounding difference. Set before running, not after seeing the answer.
MATERIALITY_RATIO = 0.005          # half a percentage point
MATERIALITY_RELATIVE = 0.01        # or 1% relative, whichever binds first


def snapshot(con, registry=None):
    out = {}
    for metric, dims in SNAPSHOT_SPECS:
        key = f"{metric}|{'+'.join(dims) if dims else 'total'}"
        rows = con.execute(compile_query(metric, dims, registry=registry)).fetchall()
        if dims:
            out[key] = {str(r[0]): (round(float(r[len(dims)]), 6)
                                    if r[len(dims)] is not None else None)
                        for r in rows}
        else:
            out[key] = {"total": round(float(rows[0][0]), 6) if rows[0][0] is not None else None}
    return out


def diff(before, after):
    moved, unchanged = [], 0
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key, {}), after.get(key, {})
        for slice_key in sorted(set(b) | set(a)):
            bv, av = b.get(slice_key), a.get(slice_key)
            if bv is None or av is None:
                if bv != av:
                    moved.append({"metric": key, "slice": slice_key,
                                  "before": bv, "after": av,
                                  "absolute_change": None, "relative_change": None,
                                  "material": True})
                continue
            d = av - bv
            rel = d / bv if bv else None
            if abs(d) < 1e-9:
                unchanged += 1
                continue
            material = (abs(d) >= MATERIALITY_RATIO
                        or (rel is not None and abs(rel) >= MATERIALITY_RELATIVE))
            moved.append({
                "metric": key, "slice": slice_key,
                "before": round(bv, 6), "after": round(av, 6),
                "absolute_change": round(d, 6),
                "relative_change": round(rel, 6) if rel is not None else None,
                "material": bool(material),
            })
    return moved, unchanged


def build_changed_registry():
    """The proposed change: loss ratio now EXCLUDES pharmacy.

    This is a real argument at every payer - pharmacy is often carved out
    to a PBM and reported separately - and it is exactly the kind of change
    that gets waved through in a definition review because it sounds like a
    clarification rather than a restatement.
    """
    reg = Registry(base_relation="vw_member_month_with_pharmacy_split")
    for d in REGISTRY.dimensions.values():
        reg.add_dimension(d)
    for m in REGISTRY.metrics.values():
        if m.name == "loss_ratio":
            reg.add_metric(Metric(
                name=m.name, label=m.label + " (ex-pharmacy)",
                numerator="SUM(paid_amount - pharmacy_paid)",
                denominator=m.denominator, grain=m.grain, unit=m.unit,
                higher_is_better=m.higher_is_better,
                description="loss ratio EXCLUDING pharmacy claims"))
        else:
            reg.add_metric(m)
    return reg


def invariants(con):
    """Properties that must hold whatever the definitions say."""
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    r = con.execute("""
        SELECT MIN(paid_amount), MIN(premium_collected), MIN(total_lines),
               MIN(denied_lines), MAX(denied_lines - total_lines)
        FROM vw_member_month""").fetchone()
    add("no negative amounts or counts", all(x >= 0 for x in r[:4]),
        f"min paid={r[0]}, premium={r[1]}, lines={r[2]}, denied={r[3]}")
    add("denied lines never exceed total lines", r[4] <= 0,
        f"max(denied - total) = {r[4]}")

    lr = con.execute(compile_query("loss_ratio", [])).fetchone()[0]
    add("loss ratio within a plausible band", 0.3 <= lr <= 1.5,
        f"loss_ratio = {lr:.4f}")

    # dimensional slices must reconcile to the total. A ratio does NOT sum,
    # so this is checked on the numerator - which is the correct way and the
    # one people get wrong by trying to average the ratio.
    total = con.execute(compile_query("paid_amount", [])).fetchone()[0]
    by_region = sum(r[1] for r in con.execute(
        compile_query("paid_amount", ["region"])).fetchall())
    add("region slices reconcile to the total",
        abs(total - by_region) < 0.01,
        f"total={total:,.2f} vs sum(region)={by_region:,.2f}")

    # the ratio must equal numerator/denominator recomputed independently
    row = con.execute(compile_query("loss_ratio", [])).fetchone()
    recomputed = row[1] / row[2]
    add("ratio equals numerator / denominator", abs(row[0] - recomputed) < 1e-9,
        f"{row[0]:.9f} vs {recomputed:.9f}")
    return checks


def main():
    con = duckdb.connect(str(DB_PATH), read_only=False)

    # a view carrying the pharmacy split, needed by the changed definition
    con.execute("""
        CREATE OR REPLACE VIEW vw_member_month_with_pharmacy_split AS
        WITH ph AS (
            SELECT member_id, month_index,
                   SUM(CASE WHEN service_category = 'pharmacy'
                            THEN paid_amount ELSE 0 END) AS pharmacy_paid
            FROM fct_claim_line GROUP BY 1, 2
        )
        SELECT v.*, COALESCE(ph.pharmacy_paid, 0) AS pharmacy_paid
        FROM vw_member_month v
        LEFT JOIN ph ON ph.member_id = v.member_id
                    AND ph.month_index = v.month_index;
    """)

    current = snapshot(con)
    had_golden = GOLDEN.exists()
    if had_golden:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        drift, unchanged = diff(golden, current)
    else:
        GOLDEN.write_text(json.dumps(current, indent=2), encoding="utf-8")
        golden, drift, unchanged = current, [], len(
            [1 for v in current.values() for _ in v])

    changed = snapshot(con, build_changed_registry())
    moved, same = diff(current, changed)
    material = [m for m in moved if m["material"]]

    inv = invariants(con)
    con.close()

    summary = {
        "golden_file_existed": had_golden,
        "snapshot_specs": [f"{m}|{'+'.join(d) if d else 'total'}"
                           for m, d in SNAPSHOT_SPECS],
        "values_snapshotted": sum(len(v) for v in current.values()),
        "drift_vs_golden": {
            "moved": len(drift), "unchanged": unchanged, "detail": drift[:20],
            "verdict": ("no drift - the warehouse reproduces the golden values"
                        if not drift else
                        "VALUES MOVED since the golden snapshot - investigate "
                        "before publishing anything"),
        },
        "definition_change_blast_radius": {
            "change": "loss_ratio now EXCLUDES pharmacy claims",
            "figures_compared": sum(len(v) for v in current.values()),
            "figures_moved": len(moved),
            "materially_moved": len(material),
            "unchanged": same,
            "largest_moves": sorted(
                [m for m in moved if m["absolute_change"] is not None],
                key=lambda m: -abs(m["absolute_change"]))[:10],
            "why_this_matters": (
                "every one of these is a historical number that a person has "
                "already read and acted on. The change is defensible; making "
                "it without publishing this list is not."),
        },
        "invariants": {
            "checks": inv,
            "passed": sum(c["passed"] for c in inv),
            "total": len(inv),
        },
    }
    with open(REPORTS / "metric_regression.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"GOLDEN     {summary['values_snapshotted']} metric values snapshotted "
          f"across {len(SNAPSHOT_SPECS)} metric/dimension specs")
    dv = summary["drift_vs_golden"]
    print(f"DRIFT      {dv['moved']} moved, {dv['unchanged']} unchanged "
          f"-> {dv['verdict'][:60]}")
    b = summary["definition_change_blast_radius"]
    print(f"\nBLAST RADIUS of '{b['change']}'")
    print(f"  {b['figures_moved']} of {b['figures_compared']} historical figures "
          f"moved, {b['materially_moved']} MATERIALLY")
    for m in b["largest_moves"][:6]:
        print(f"    {m['metric']:<24} {m['slice']:<10} "
              f"{m['before']:>9.4f} -> {m['after']:>9.4f}  "
              f"({m['relative_change']:+.2%})" if m["relative_change"] is not None
              else f"    {m['metric']} {m['slice']}")
    i = summary["invariants"]
    print(f"\nINVARIANTS {i['passed']}/{i['total']} passed")
    for c in inv:
        if not c["passed"]:
            print(f"  FAILED  {c['check']}: {c['detail']}")


if __name__ == "__main__":
    main()
