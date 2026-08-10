"""
test_semantic_and_analytics.py
---------------------------------
Tests for the properties that make Aegis's numbers trustworthy.

Written as invariants and boundary cases rather than as recordings of
current output, because `qa/injected_bugs.py` breaks the code on purpose
and counts what this suite notices. A test asserting
"loss_ratio == 0.8519" fails for every change and diagnoses none of them.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "platform" / "semantic"))
sys.path.insert(0, str(ROOT / "platform" / "ingest"))
sys.path.insert(0, str(ROOT / "analytics" / "survival"))
sys.path.insert(0, str(ROOT / "integrity"))

from metrics import REGISTRY, Metric, compile_query, metric_catalog  # noqa: E402
from domain import EM_INTENSITY, month_index, month_label  # noqa: E402
from member_survival import kaplan_meier, cox_ph, median_survival  # noqa: E402

DB = ROOT / "database" / "aegis.duckdb"
SETTINGS = settings(max_examples=100, deadline=None)


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(DB), read_only=True)
    yield c
    c.close()


# ==================================================================
#  Semantic layer governance
# ==================================================================
def test_unknown_dimension_is_rejected():
    """Governance must fail at compile time, not return a wrong shape."""
    with pytest.raises(ValueError, match="unknown dimension"):
        compile_query("loss_ratio", ["not_a_real_dimension"])


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        compile_query("invented_metric", [])


def test_a_metric_cannot_be_defined_twice():
    """The entire premise: one definition, one place."""
    with pytest.raises(ValueError, match="already defined"):
        REGISTRY.add_metric(Metric("loss_ratio", "dup", "1", None, "grain"))


def test_every_catalogued_metric_compiles_and_runs(con):
    for m in metric_catalog():
        rows = con.execute(compile_query(m["name"], [])).fetchall()
        assert len(rows) == 1, f"{m['name']} did not return a single total"


def test_every_declared_dimension_works_on_a_ratio_metric(con):
    for dim in sorted(REGISTRY.dimensions):
        rows = con.execute(compile_query("loss_ratio", [dim])).fetchall()
        assert rows, f"dimension {dim} returned nothing"


# ==================================================================
#  Metric arithmetic invariants
# ==================================================================
def test_ratio_equals_numerator_over_denominator(con):
    v, num, den = con.execute(compile_query("loss_ratio", [])).fetchone()
    assert abs(v - num / den) < 1e-9


def test_dimensional_slices_reconcile_on_the_numerator(con):
    """A RATIO does not sum across slices; its numerator does. Getting this
    backwards is how people 'validate' by averaging ratios."""
    total = con.execute(compile_query("paid_amount", [])).fetchone()[0]
    for dim in ("region", "plan", "product"):
        parts = sum(r[1] for r in con.execute(
            compile_query("paid_amount", [dim])).fetchall())
        assert abs(total - parts) < 0.01, f"{dim} slices do not reconcile"


def test_member_months_equal_the_sum_of_enrolment_durations(con):
    """The fan-out check. If a join duplicated rows, every ratio silently
    gains denominator that does not exist."""
    expected = con.execute(
        "SELECT SUM(end_month - start_month) FROM fct_enrollment_span "
        "WHERE end_month > start_month").fetchone()[0]
    actual = con.execute(compile_query("member_months", [])).fetchone()[0]
    assert int(expected) == int(actual)


def test_denied_lines_never_exceed_total_lines(con):
    worst = con.execute(
        "SELECT MAX(denied_lines - total_lines) FROM vw_member_month").fetchone()[0]
    assert worst <= 0


def test_zero_claim_member_months_stay_in_the_denominator(con):
    """Dropping them turns PMPM into 'average cost among members who
    claimed', a different and much larger number."""
    zeros = con.execute(
        "SELECT COUNT(*) FROM vw_member_month WHERE total_lines = 0").fetchone()[0]
    assert zeros > 0, "the month spine is not producing zero-claim months"


def test_loss_ratio_is_in_a_plausible_band(con):
    lr = con.execute(compile_query("loss_ratio", [])).fetchone()[0]
    assert 0.3 <= lr <= 1.5, f"loss ratio {lr} is not a real insurance number"


# ==================================================================
#  Calendar arithmetic
# ==================================================================
@given(st.integers(min_value=0, max_value=23))
@SETTINGS
def test_month_label_round_trips(m):
    assert month_index(month_label(m)) == m


def test_month_label_boundaries():
    """December to January is where naive month arithmetic breaks."""
    assert month_label(0) == "2024-01"
    assert month_label(11) == "2024-12"
    assert month_label(12) == "2025-01"


# ==================================================================
#  Survival estimators
# ==================================================================
def test_kaplan_meier_is_monotone_non_increasing():
    rng = np.random.default_rng(7)
    d = rng.integers(1, 40, 300)
    e = rng.integers(0, 2, 300)
    km = kaplan_meier(d, e)
    s = km["survival"]
    assert all(s[i] >= s[i + 1] for i in range(len(s) - 1))
    assert all(0.0 <= x <= 1.0 for x in s)


def test_censored_observations_do_not_drop_survival():
    """The defining property. With no events, survival must stay at 1.0 no
    matter how many members leave the risk set."""
    km = kaplan_meier(np.arange(1, 51), np.zeros(50, dtype=int))
    assert km["survival"] == [] or all(abs(x - 1.0) < 1e-9 for x in km["survival"])


def test_censoring_changes_the_estimate():
    """Same durations, different censoring, must give a different curve -
    otherwise the estimator is ignoring the event indicator."""
    d = np.arange(1, 41)
    all_events = kaplan_meier(d, np.ones(40, dtype=int))
    half = kaplan_meier(d, np.tile([1, 0], 20))
    assert all_events["survival"][-1] < half["survival"][-1]


def test_at_risk_counts_strictly_decrease():
    """The risk set must SHRINK as members leave it.

    An earlier version asserted only `>=`, which a constant sequence
    satisfies - so a mutation pinning at_risk to n passed. Non-increasing is
    not the property; shrinking is.
    """
    rng = np.random.default_rng(3)
    km = kaplan_meier(rng.integers(1, 30, 200), np.ones(200, dtype=int))
    ar = km["at_risk"]
    assert all(ar[i] > ar[i + 1] for i in range(len(ar) - 1)),         "at-risk counts must fall at every event time, not merely not rise"
    assert ar[0] > ar[-1]


def test_cox_recovers_a_known_positive_effect():
    """A covariate that raises the hazard must get a positive coefficient.
    Sign recovery is the minimum bar; the report measures magnitude."""
    rng = np.random.default_rng(11)
    n = 800
    x = rng.normal(0, 1, n)
    # higher x -> shorter time
    t = np.clip(rng.exponential(np.exp(-0.8 * x) * 10), 1, 60).astype(int)
    e = (t < 60).astype(int)
    beta, _, _ = cox_ph(x.reshape(-1, 1), t, e)
    assert beta[0] > 0.3, f"expected a clear positive coefficient, got {beta[0]}"


def test_cox_is_invariant_to_row_order():
    """The partial likelihood sorts internally; a shuffled input must give
    the identical fit, or the risk sets are being built wrong."""
    rng = np.random.default_rng(5)
    n = 400
    X = rng.normal(0, 1, (n, 2))
    t = rng.integers(1, 30, n)
    e = rng.integers(0, 2, n)
    b1, _, _ = cox_ph(X, t, e)
    p = rng.permutation(n)
    b2, _, _ = cox_ph(X[p], t[p], e[p])
    assert np.allclose(b1, b2, atol=1e-4)


def test_median_survival_is_the_first_time_below_half():
    km = {"times": [1, 2, 3, 4], "survival": [0.9, 0.7, 0.45, 0.2]}
    assert median_survival(km) == 3


# ==================================================================
#  Coding intensity ordering
# ==================================================================
def test_em_intensity_is_strictly_ordered():
    """Upcoding is defined by ORDER. If these stop being ordered, every
    peer-relative z-score becomes meaningless."""
    vals = [EM_INTENSITY[c] for c in ("99212", "99213", "99214", "99215")]
    assert vals == sorted(vals) and len(set(vals)) == 4
