"""
validation.py
----------------
The QA discipline a regulated payer actually has to satisfy, plus the one
measurement that tells you whether the test suite is worth anything.

Three parts:

  TRACEABILITY   every requirement maps to a specification, a test, and
                 the evidence that test produced. An auditor's first
                 question is not "do you have tests" but "show me the test
                 that proves requirement REQ-004 and the run that passed
                 it". A requirement with no test is reported as a gap
                 rather than quietly omitted.

  AUDIT TRAIL    an append-only, hash-chained record of every automated
                 decision. Chaining matters: it makes silent edits
                 detectable, which is the difference between a log and an
                 audit trail. The chain is deliberately tampered with at
                 the end to prove the verifier notices.

  INJECTED BUGS  ten real defects planted in the analytics code, with the
                 suite run against each. A green suite proves the tests
                 pass; it says nothing about whether they would fail if
                 the code were wrong, and that is the only property worth
                 measuring.

Output: reports/validation.json
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
PY = sys.executable

# =====================================================================
#  Requirements -> specification -> test
# =====================================================================
REQUIREMENTS = [
    ("REQ-001", "Every business metric has exactly one definition",
     "platform/semantic/metrics.py::Registry.add_metric",
     ["test_a_metric_cannot_be_defined_twice"]),
    ("REQ-002", "A query may not reference an undeclared dimension",
     "platform/semantic/metrics.py::Metric.compile",
     ["test_unknown_dimension_is_rejected"]),
    ("REQ-003", "A query may not reference an undefined metric",
     "platform/semantic/metrics.py::Registry.get",
     ["test_unknown_metric_is_rejected"]),
    ("REQ-004", "Ratio metrics equal numerator divided by denominator",
     "platform/semantic/metrics.py::Metric.compile",
     ["test_ratio_equals_numerator_over_denominator"]),
    ("REQ-005", "Dimensional slices reconcile to the reported total",
     "platform/warehouse/build_warehouse.py::vw_member_month",
     ["test_dimensional_slices_reconcile_on_the_numerator"]),
    ("REQ-006", "Member-months equal the sum of enrolment durations (no fan-out)",
     "platform/warehouse/build_warehouse.py::vw_member_month",
     ["test_member_months_equal_the_sum_of_enrolment_durations"]),
    ("REQ-007", "Member-months with zero claims remain in the denominator",
     "platform/warehouse/build_warehouse.py::dim_month spine",
     ["test_zero_claim_member_months_stay_in_the_denominator"]),
    ("REQ-008", "Censored members do not reduce estimated survival",
     "analytics/survival/member_survival.py::kaplan_meier",
     ["test_censored_observations_do_not_drop_survival",
      "test_censoring_changes_the_estimate"]),
    ("REQ-009", "Survival estimates are independent of input row order",
     "analytics/survival/member_survival.py::cox_ph",
     ["test_cox_is_invariant_to_row_order"]),
    ("REQ-010", "E/M coding intensity is strictly ordered",
     "platform/ingest/domain.py::EM_INTENSITY",
     ["test_em_intensity_is_strictly_ordered"]),
    ("REQ-011", "Reported loss ratio falls within a plausible band",
     "platform/semantic/metrics.py::loss_ratio",
     ["test_loss_ratio_is_in_a_plausible_band"]),
    ("REQ-012", "A metric definition change publishes its blast radius",
     "integrity/metric_regression.py",
     []),          # deliberately untested - reported as a gap
]

# =====================================================================
#  Injected defects
# =====================================================================
BUGS = [
    ("ratio_inverted", "platform/semantic/metrics.py",
     "ELSE {self.numerator} / {self.denominator} END",
     "ELSE {self.denominator} / {self.numerator} END",
     "loss ratio computed upside down"),
    ("duplicate_metric_allowed", "platform/semantic/metrics.py",
     'raise ValueError(f"metric \'{m.name}\' already defined - a metric may "',
     'pass  # raise ValueError(f"metric \'{m.name}\' already defined - a metric may "',
     "a metric can be silently redefined, defeating the whole layer"),
    ("unknown_dimension_ignored", "platform/semantic/metrics.py",
     "        if unknown:",
     "        if False and unknown:",
     "an undeclared dimension is silently dropped instead of rejected"),
    ("km_ignores_censoring", "analytics/survival/member_survival.py",
     "            d += int(e[j])",
     "            d += 1",
     "Kaplan-Meier treats censored members as events"),
    ("km_wrong_risk_set", "analytics/survival/member_survival.py",
     "        at_risk = n - i",
     "        at_risk = n",
     "the at-risk denominator never shrinks"),
    ("cox_drops_censored_from_risk_set", "analytics/survival/member_survival.py",
     "        return -float(np.sum(eta[ev_idx] - (m + np.log(risk_sum[ev_first]))))",
     "        return -float(np.sum(eta[ev_idx] - (m + np.log(risk_sum[ev_idx]))))",
     "Breslow ties handled wrong, so tied event times use different risk sets"),
    ("month_off_by_one", "platform/ingest/domain.py",
     "    mo = (START_DATE.month - 1 + m) % 12 + 1",
     "    mo = (START_DATE.month + m) % 12 + 1",
     "month labels shift by one, silently misdating every report"),
    ("em_intensity_unordered", "platform/ingest/domain.py",
     'EM_CODES = ["99212", "99213", "99214", "99215"]',
     'EM_CODES = ["99213", "99212", "99214", "99215"]',
     "coding intensity order scrambled, so upcoding z-scores are meaningless"),
    ("member_month_spine_dropped", "platform/warehouse/build_warehouse.py",
     "        LEFT JOIN claims c ON c.member_id = e.member_id",
     "        INNER JOIN claims c ON c.member_id = e.member_id",
     "zero-claim member-months vanish, inflating PMPM"),
    ("denominator_fanout", "platform/warehouse/build_warehouse.py",
     "              ON m.month_index >= s.start_month AND m.month_index < s.end_month",
     "              ON m.month_index >= s.start_month",
     "exposure join fans out, adding member-months that never existed"),
]

SUITE = ["-m", "pytest", "tests/", "-q", "-x", "--no-header",
         "-p", "no:cacheprovider"]


# =====================================================================
#  Audit trail
# =====================================================================
class AuditTrail:
    """Append-only, hash-chained.

    Each entry carries the hash of the previous one, so altering an old
    record invalidates every hash after it. A plain log lets you edit
    yesterday's decision; a chain makes that detectable, which is the
    entire regulatory point.
    """

    GENESIS = "0" * 64

    def __init__(self):
        self.entries = []

    def append(self, actor, action, subject, detail):
        prev = self.entries[-1]["hash"] if self.entries else self.GENESIS
        body = {
            "seq": len(self.entries),
            "actor": actor, "action": action, "subject": subject,
            "detail": detail,
            # fixed timestamp: a wall-clock value would make the whole
            # report unreproducible by checksum
            "recorded_at": "2026-01-01T00:00:00Z",
            "prev_hash": prev,
        }
        body["hash"] = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.entries.append(body)
        return body

    def verify(self):
        prev = self.GENESIS
        for e in self.entries:
            recomputed = hashlib.sha256(json.dumps(
                {k: v for k, v in e.items() if k != "hash"},
                sort_keys=True).encode()).hexdigest()
            if e["prev_hash"] != prev or recomputed != e["hash"]:
                return {"intact": False, "broken_at_seq": e["seq"]}
            prev = e["hash"]
        return {"intact": True, "broken_at_seq": None}


def run_suite(cwd, timeout=900):
    try:
        p = subprocess.run([PY, *SUITE], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-400:]
    except subprocess.TimeoutExpired:
        return 99, "TIMEOUT"


def collect_test_names():
    out = subprocess.run(
        [PY, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True)
    names = set()
    for line in out.stdout.splitlines():
        if "::" in line:
            names.add(line.split("::")[-1].split("[")[0].strip())
    return names


def main():
    # ---------------- traceability
    available = collect_test_names()
    matrix, gaps = [], []
    for rid, text, spec, tests in REQUIREMENTS:
        present = [t for t in tests if t in available]
        missing = [t for t in tests if t not in available]
        covered = bool(tests) and not missing
        matrix.append({
            "requirement": rid, "statement": text, "specification": spec,
            "tests": tests, "tests_found": present, "tests_missing": missing,
            "covered": covered,
        })
        if not covered:
            gaps.append({"requirement": rid, "statement": text,
                         "reason": ("no test declared" if not tests
                                    else f"declared tests not found: {missing}")})

    # ---------------- audit trail
    trail = AuditTrail()
    trail.append("pipeline", "metric_definition_change",
                 "loss_ratio", "excluded pharmacy; 33 historical figures moved")
    trail.append("pipeline", "model_promoted", "risk_adjustment.ridge",
                 "calibration slope 1.0418, within the 0.9-1.1 band")
    trail.append("investigator", "provider_flagged", "NPI3000xxx",
                 "peer intensity z > 2.5 within specialty")
    trail.append("pipeline", "copilot_refusal", "member SSN request",
                 "outside the governed vocabulary")
    intact_before = trail.verify()

    # tamper with a settled record and prove the chain notices
    trail.entries[1]["detail"] = "calibration slope 1.0000 (edited after the fact)"
    after_tamper = trail.verify()
    trail.entries[1]["detail"] = ("calibration slope 1.0418, within the "
                                  "0.9-1.1 band")   # restore

    # ---------------- injected bugs
    baseline_rc, baseline_out = run_suite(ROOT)
    bug_results = []
    if baseline_rc != 0:
        bug_summary = {"error": "baseline suite is RED; bug scoring is meaningless",
                       "output": baseline_out}
    else:
        for bug_id, rel, find, repl, breaks in BUGS:
            with tempfile.TemporaryDirectory(prefix="aegis_mut_") as tmp:
                work = Path(tmp) / "repo"
                # `data` is COPIED, and the warehouse is rebuilt below.
                #
                # The first version excluded data/ and relied on the copied
                # .duckdb file. Three defects survived as a result - two of
                # them mutations of the warehouse SQL itself - because the
                # tests were querying a database that had been built by
                # UNMUTATED code. The mutation was present in the source and
                # absent from everything the suite could observe, which made
                # the harness report a hole in the tests that was really a
                # hole in the harness.
                shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
                    "__pycache__", ".hypothesis", ".pytest_cache", ".git"))
                target = work / rel
                src = target.read_text(encoding="utf-8")
                if find not in src:
                    bug_results.append({"bug": bug_id, "applied": False,
                                        "caught": None,
                                        "note": "anchor not found - stale mutation, "
                                                "counted neither way"})
                    continue
                target.write_text(src.replace(find, repl, 1), encoding="utf-8")
                # rebuild so warehouse mutations reach the data the tests read
                build = subprocess.run(
                    [PY, "platform/warehouse/build_warehouse.py"], cwd=work,
                    capture_output=True, text=True, timeout=600)
                if build.returncode != 0:
                    # a mutation that breaks the build IS caught - loudly
                    bug_results.append({
                        "bug": bug_id, "file": rel, "applied": True,
                        "breaks": breaks, "caught": True, "exit_code": build.returncode,
                        "evidence": "warehouse build failed under mutation"})
                    continue
                rc, out = run_suite(work)
                bug_results.append({
                    "bug": bug_id, "file": rel, "applied": True, "breaks": breaks,
                    "caught": rc != 0, "exit_code": rc,
                    "evidence": out.strip().splitlines()[-1] if out.strip() else "",
                })
        applied = [b for b in bug_results if b["applied"]]
        caught = [b for b in applied if b["caught"]]
        survived = [b for b in applied if not b["caught"]]
        bug_summary = {
            "injected": len(applied), "caught": len(caught),
            "survived": len(survived),
            "detection_rate": round(len(caught) / len(applied), 4) if applied else None,
            "survivors": [{"bug": b["bug"], "breaks": b["breaks"]} for b in survived],
            "stale": len([b for b in bug_results if not b["applied"]]),
            "detail": bug_results,
        }

    summary = {
        "traceability": {
            "requirements": len(REQUIREMENTS),
            "covered": sum(1 for m in matrix if m["covered"]),
            "gaps": gaps,
            "coverage_pct": round(100 * sum(1 for m in matrix if m["covered"])
                                  / len(REQUIREMENTS), 1),
            "matrix": matrix,
            "note": ("REQ-012 has no test and is listed as a gap rather than "
                     "removed from the register. A traceability matrix that "
                     "only contains the requirements you happened to test is "
                     "a list of tests, not a matrix."),
        },
        "audit_trail": {
            "entries": len(trail.entries),
            "chain_intact": intact_before["intact"],
            "tamper_detected": not after_tamper["intact"],
            "tamper_detected_at_seq": after_tamper["broken_at_seq"],
            "note": ("an entry was edited after the fact and the chain caught "
                     "it; a plain log would have accepted the edit silently"),
            "entries_detail": trail.entries,
        },
        "injected_bugs": bug_summary,
    }
    with open(REPORTS / "validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    t = summary["traceability"]
    print(f"TRACEABILITY  {t['covered']}/{t['requirements']} requirements covered "
          f"({t['coverage_pct']}%)")
    for g in gaps:
        print(f"  GAP  {g['requirement']}  {g['statement'][:56]}  ({g['reason']})")
    a = summary["audit_trail"]
    print(f"\nAUDIT TRAIL   {a['entries']} entries, chain intact: {a['chain_intact']}")
    print(f"              tamper detected: {a['tamper_detected']} "
          f"(at seq {a['tamper_detected_at_seq']})")
    b = summary["injected_bugs"]
    if "error" in b:
        print(f"\nINJECTED BUGS  {b['error']}")
    else:
        print(f"\nINJECTED BUGS {b['caught']}/{b['injected']} caught "
              f"({b['detection_rate']:.0%})"
              + (f", {b['stale']} stale" if b["stale"] else ""))
        for r in b["detail"]:
            if r["applied"]:
                print(f"  {'CAUGHT ' if r['caught'] else 'SURVIVED'}  {r['bug']}")
        for s in b["survivors"]:
            print(f"\n  HOLE: {s['bug']} -> {s['breaks']}")


if __name__ == "__main__":
    main()
