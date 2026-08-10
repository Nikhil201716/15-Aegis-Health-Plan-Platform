"""
member_survival.py
---------------------
Time-to-lapse analysis on enrolment spans where 35.6% of members have NOT
lapsed by the time we look.

Those members are the whole point. They are RIGHT-CENSORED: their lapse
time is unknown and greater than their observed tenure. The instinctive
move - label them "did not churn" and fit a classifier - throws away the
distinction between "stayed 22 months and is still here" and "stayed 2
months and is still here", and treats the second as evidence of loyalty.

Four estimators are fitted on identical data and scored against the hazard
the generator injected:

  NAIVE LOGISTIC     churned yes/no, ignoring tenure entirely. The control
                     arm, and the thing most teams actually ship.
  KAPLAN-MEIER       non-parametric survival by cohort, with Greenwood
                     confidence bands and at-risk counts.
  COX PH             partial likelihood (Breslow ties), hand-implemented.
                     The right family; a continuous-time approximation to
                     a process that is actually discrete.
  DISCRETE-TIME      person-month logistic hazard. This one MATCHES the
  HAZARD             data-generating process exactly, and is included
                     because "Cox is the survival model" is a habit rather
                     than a decision - when data arrive in monthly
                     buckets, the discrete-time model is the correct one.

Everything is implemented directly rather than through lifelines: the
partial likelihood is fifteen lines, and writing it is the difference
between using survival analysis and understanding it. It also keeps the
project runnable with no extra dependency.

Output: reports/survival.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "platform" / "ingest"))
from domain import LAPSE_COEFFS, N_MONTHS  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# The covariates the generator actually used. Named here so the scoring is
# a like-for-like comparison rather than a fishing expedition.
COVARIATES = ["premium_burden_x10", "had_denial", "utilisation_low",
              "channel_individual"]


def load():
    spans = pd.read_csv(DATA / "enrollment_spans.csv")
    members = pd.read_csv(DATA / "members.csv")
    d = spans.merge(members[["member_id", "premium_burden", "channel", "region",
                             "plan", "age_band"]], on="member_id")
    # premium_burden enters the generator as burden*10; matching the scale
    # here means the recovered coefficient is directly comparable
    d["premium_burden_x10"] = d.premium_burden * 10
    d["channel_individual"] = (d.channel == "individual").astype(int)
    return d


# =====================================================================
#  Kaplan-Meier
# =====================================================================
def kaplan_meier(durations, events):
    """Non-parametric survival, with Greenwood variance.

    At each event time: S(t) = prod(1 - d_i / n_i), where n_i is the number
    still AT RISK. Censored observations leave the risk set without
    counting as events - which is exactly the information a classifier
    discards.
    """
    order = np.argsort(durations)
    t, e = np.asarray(durations)[order], np.asarray(events)[order]
    times, surv, at_risk_out, var_sum, s = [], [], [], 0.0, 1.0
    n = len(t)
    i = 0
    while i < n:
        ti = t[i]
        j = i
        d = 0
        while j < n and t[j] == ti:
            d += int(e[j])
            j += 1
        at_risk = n - i
        if d > 0:
            s *= (1 - d / at_risk)
            var_sum += d / (at_risk * (at_risk - d)) if at_risk > d else 0.0
            times.append(int(ti))
            surv.append(float(s))
            at_risk_out.append(int(at_risk))
        i = j
    surv = np.array(surv)
    se = surv * np.sqrt(var_sum) if len(surv) else surv
    return {
        "times": times,
        "survival": [round(float(x), 5) for x in surv],
        "at_risk": at_risk_out,
        "ci_lower": [round(float(max(a - 1.96 * b, 0)), 5) for a, b in zip(surv, se)],
        "ci_upper": [round(float(min(a + 1.96 * b, 1)), 5) for a, b in zip(surv, se)],
    }


def median_survival(km):
    for t, s in zip(km["times"], km["survival"]):
        if s <= 0.5:
            return t
    return None


def logrank(d1, e1, d2, e2):
    """Log-rank test between two groups. Compares observed events to those
    expected under equal hazards, at every event time."""
    times = np.unique(np.concatenate([d1[e1 == 1], d2[e2 == 1]]))
    O1 = E1 = V = 0.0
    for t in times:
        n1, n2 = (d1 >= t).sum(), (d2 >= t).sum()
        n = n1 + n2
        if n < 2:
            continue
        o1 = ((d1 == t) & (e1 == 1)).sum()
        o = o1 + ((d2 == t) & (e2 == 1)).sum()
        O1 += o1
        E1 += o * n1 / n
        if n > 1:
            V += o * (n1 / n) * (1 - n1 / n) * (n - o) / (n - 1)
    if V <= 0:
        return {"chi2": None, "p_value": None}
    chi2 = (O1 - E1) ** 2 / V
    from scipy.stats import chi2 as chi2_dist
    return {"chi2": round(float(chi2), 4),
            "p_value": round(float(1 - chi2_dist.cdf(chi2, 1)), 6)}


# =====================================================================
#  Cox proportional hazards, by partial likelihood
# =====================================================================
def cox_ph(X, durations, events):
    """Breslow partial likelihood, vectorised.

    For each event time the contribution is beta'x for the member who
    lapsed, minus log of the summed hazard over everyone still at risk.
    Censored members enter the risk set but never the numerator - which is
    precisely how censoring is USED rather than discarded.

    The first implementation rebuilt the risk set with a boolean mask per
    event: 15,446 events x 24,000 members per likelihood evaluation, times
    a hundred optimiser steps, and it did not finish. Sorting by duration
    makes the risk set a SUFFIX of the array, so a reverse cumulative sum
    gives every risk-set total in one pass - the same arithmetic, O(n)
    instead of O(n^2).

    Ties are handled Breslow-style: everyone sharing an event time sees the
    identical risk set, so tied indices are mapped back to the first index
    of their tie group.
    """
    X = np.asarray(X, dtype=float)
    order = np.argsort(durations, kind="stable")
    X, t, e = X[order], np.asarray(durations)[order], np.asarray(events)[order].astype(bool)
    n, p = X.shape

    # first_idx[i] = index of the first row sharing t[i]; that row's suffix
    # is the correct Breslow risk set for every member of the tie group
    first_idx = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        first_idx[i:j] = i
        i = j

    ev_idx = np.where(e)[0]
    ev_first = first_idx[ev_idx]

    def neg_ll(beta):
        eta = X @ beta
        m = eta.max()
        w = np.exp(eta - m)
        risk_sum = np.cumsum(w[::-1])[::-1]      # risk_sum[i] = sum(w[i:])
        return -float(np.sum(eta[ev_idx] - (m + np.log(risk_sum[ev_first]))))

    def grad(beta):
        eta = X @ beta
        m = eta.max()
        w = np.exp(eta - m)
        risk_sum = np.cumsum(w[::-1])[::-1]
        # weighted covariate totals over each suffix, same trick per column
        wx = X * w[:, None]
        wx_sum = np.cumsum(wx[::-1], axis=0)[::-1]
        expected = wx_sum[ev_first] / risk_sum[ev_first][:, None]
        return -(X[ev_idx] - expected).sum(axis=0)

    res = minimize(neg_ll, np.zeros(p), jac=grad, method="BFGS",
                   options={"maxiter": 300, "gtol": 1e-6})
    try:
        se = np.sqrt(np.clip(np.diag(res.hess_inv), 0, None))
    except Exception:
        se = np.full(p, np.nan)
    return res.x, se, float(-res.fun)


# =====================================================================
#  Discrete-time hazard - the correct specification here
# =====================================================================
def person_month(d):
    """Explode spans into one row per member-month at risk, vectorised.

    had_denial is TIME-VARYING: a member carries it only from denial_month
    onward. Collapsing it to a per-member flag - which the naive and Cox
    arms both do - attributes the denial's effect to months before it
    happened and biases the coefficient toward zero.
    """
    tenure = d.tenure_months.to_numpy(int)
    ev = d.event_observed.to_numpy(int)
    # a member contributes months 0..tenure-1 always, plus month `tenure`
    # only if the lapse was actually observed there
    n_rows = tenure + ev
    idx = np.repeat(np.arange(len(d)), n_rows)
    month = np.concatenate([np.arange(k) for k in n_rows]) if len(d) else np.array([])

    start = d.start_month.to_numpy(int)[idx]
    dmonth = d.denial_month.to_numpy(int)[idx]
    had_denial = ((dmonth >= 0) & (month >= (dmonth - start))).astype(int)

    out = pd.DataFrame({
        "premium_burden_x10": d.premium_burden_x10.to_numpy(float)[idx],
        "had_denial": had_denial,
        "utilisation_low": d.utilisation_low.to_numpy(int)[idx],
        "channel_individual": d.channel_individual.to_numpy(int)[idx],
        "tenure_at_month": month,
        "lapsed": ((month == tenure[idx]) & (ev[idx] == 1)).astype(int),
    })
    return out


def main():
    d = load()
    dur = d.tenure_months.to_numpy()
    ev = d.event_observed.to_numpy()

    censoring_rate = float((ev == 0).mean())

    # ---- Kaplan-Meier overall and by channel
    km_all = kaplan_meier(dur, ev)
    km_by = {}
    for ch, g in d.groupby("channel"):
        km_by[ch] = kaplan_meier(g.tenure_months.to_numpy(), g.event_observed.to_numpy())
    ind = d[d.channel == "individual"]
    emp = d[d.channel == "employer"]
    lr = logrank(ind.tenure_months.to_numpy(), ind.event_observed.to_numpy(),
                 emp.tenure_months.to_numpy(), emp.event_observed.to_numpy())

    # ---- Cox
    Xc = d[COVARIATES].to_numpy(dtype=float)
    beta, se, ll = cox_ph(Xc, dur, ev)
    cox_coeffs = {c: round(float(b), 4) for c, b in zip(COVARIATES, beta)}

    # ---- discrete-time hazard (matches the generator's process)
    pm = person_month(d)
    dt_model = LogisticRegression(max_iter=1000, C=1e6)
    dt_model.fit(pm[COVARIATES + ["tenure_at_month"]].to_numpy(float), pm.lapsed)
    dt_coeffs = {c: round(float(v), 4) for c, v in
                 zip(COVARIATES + ["tenure_at_month"], dt_model.coef_[0])}

    # ---- naive control arm: churned yes/no, tenure discarded
    naive = LogisticRegression(max_iter=1000, C=1e6)
    naive.fit(d[COVARIATES].to_numpy(float), ev)
    naive_coeffs = {c: round(float(v), 4) for c, v in zip(COVARIATES, naive.coef_[0])}

    # ---- score every arm against the injected truth
    truth = {
        "premium_burden_x10": LAPSE_COEFFS["premium_burden"],
        "had_denial": LAPSE_COEFFS["had_denial"],
        "utilisation_low": LAPSE_COEFFS["utilisation_low"],
        "channel_individual": LAPSE_COEFFS["channel_individual"],
        "tenure_at_month": LAPSE_COEFFS["tenure_months"],
    }

    def score(est):
        errs = {k: round(est[k] - truth[k], 4) for k in est if k in truth}
        mae = float(np.mean([abs(v) for v in errs.values()]))
        return {"coefficients": est, "error_vs_truth": errs,
                "mean_abs_error": round(mae, 4)}

    arms = {
        "naive_logistic_churned_yes_no": score(naive_coeffs),
        "cox_proportional_hazards": score(cox_coeffs),
        "discrete_time_hazard": score(dt_coeffs),
    }

    # ---- competing risks: not every exit is a lapse
    exits = d[d.event_observed == 1]
    cause_mix = exits.exit_cause.value_counts(normalize=True).round(4).to_dict()
    vol = d.copy()
    vol["is_voluntary"] = ((vol.event_observed == 1)
                           & (vol.exit_cause == "voluntary_lapse")).astype(int)
    km_vol_naive = kaplan_meier(vol.tenure_months.to_numpy(),
                                vol.is_voluntary.to_numpy())
    # 1 - KM treats other exit causes as censored, which assumes a member
    # who died could still have lapsed later. The cumulative incidence
    # function does not, and is always the smaller number.
    naive_incidence = 1 - km_vol_naive["survival"][-1] if km_vol_naive["survival"] else 0
    cif = float(((vol.event_observed == 1)
                 & (vol.exit_cause == "voluntary_lapse")).mean())

    summary = {
        "n_members": int(len(d)),
        "n_events": int(ev.sum()),
        "censoring_rate": round(censoring_rate, 4),
        "median_survival_months": median_survival(km_all),
        "kaplan_meier_overall": km_all,
        "kaplan_meier_by_channel": {k: {"times": v["times"],
                                        "survival": v["survival"],
                                        "median": median_survival(v)}
                                    for k, v in sorted(km_by.items())},
        "logrank_individual_vs_employer": lr,
        "injected_truth": truth,
        "estimators": arms,
        "competing_risks": {
            "exit_cause_mix": cause_mix,
            "naive_1_minus_km_voluntary_lapse": round(float(naive_incidence), 4),
            "cumulative_incidence_voluntary_lapse": round(cif, 4),
            "overstatement": round(float(naive_incidence) - cif, 4),
            "why": ("1-KM treats death and plan-switch as censoring, which "
                    "assumes those members could still have lapsed later. They "
                    "could not. The cumulative incidence function is the honest "
                    "number and is always smaller."),
        },
        "cox_log_partial_likelihood": round(ll, 2),
    }
    with open(REPORTS / "survival.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"SURVIVAL   {len(d):,} members, {int(ev.sum()):,} lapses, "
          f"{censoring_rate:.1%} right-censored")
    print(f"           median survival {median_survival(km_all)} months")
    print(f"           log-rank individual vs employer: chi2={lr['chi2']}, "
          f"p={lr['p_value']}")
    print(f"\nCOEFFICIENT RECOVERY vs the injected hazard")
    print(f"{'covariate':<24} {'truth':>8} {'naive':>9} {'cox':>9} {'discrete':>10}")
    for c in COVARIATES + ["tenure_at_month"]:
        t = truth[c]
        nv = naive_coeffs.get(c)
        cx = cox_coeffs.get(c)
        dt = dt_coeffs.get(c)
        f = lambda v: f"{v:>9.4f}" if v is not None else f"{'-':>9}"
        print(f"{c:<24} {t:>8.3f} {f(nv)} {f(cx)} {f(dt):>10}")
    print(f"\n{'estimator':<34} {'mean abs error':>15}")
    for k, v in arms.items():
        print(f"{k:<34} {v['mean_abs_error']:>15.4f}")
    cr = summary["competing_risks"]
    print(f"\nCOMPETING RISKS  1-KM says {cr['naive_1_minus_km_voluntary_lapse']:.1%} "
          f"voluntary lapse, CIF says {cr['cumulative_incidence_voluntary_lapse']:.1%} "
          f"(overstated by {cr['overstatement']:.1%})")


if __name__ == "__main__":
    main()
