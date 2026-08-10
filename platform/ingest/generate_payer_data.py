"""
generate_payer_data.py
-------------------------
Generates the Aegis book of business, with every injected mechanism
recorded so downstream analysis can be scored rather than admired.

The design point that matters most is CENSORING. Members are simulated
month by month with a real hazard of lapsing, and the observation window
closes at OBSERVATION_END_MONTH. A member still enrolled at that point has
not "not churned" - their lapse time is simply unknown and greater than
their observed tenure. Recording `event_observed = 0` for them is what
makes survival analysis possible and what makes the naive
churned-yes/no control arm measurably wrong.

Outputs:
    data/members.csv              one row per member (demographics, plan)
    data/enrollment_spans.csv     tenure + event/censoring flag
    data/providers.csv
    data/claim_lines.csv          the fact table
    data/clinical_notes.csv       free text for the AI layer
    data/ground_truth_payer.json  the answer key
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain import (  # noqa: E402
    BASELINE_MONTHLY_HAZARD, CHANNELS, CLAIM_STATUS, DENIAL_REASONS,
    EM_ALLOWED_AMOUNT, EM_CODES, EM_INTENSITY, EM_MINUTES,
    FRAUD_RING_SPECIALTIES, LAPSE_CAUSE_SHARES, LAPSE_COEFFS,
    LEVEL_SHIFT_MONTH, LEVEL_SHIFT_REGION, LEVEL_SHIFT_SIZE, N_MEMBERS,
    N_MONTHS, N_FRAUD_PROVIDERS, N_PROVIDERS, NULL_PROXY_NAME,
    NULL_PROXY_TRUE_EFFECT, OBSERVATION_END_MONTH, PLACE_OF_SERVICE, PLANS,
    PRODUCTS, REGION_MONTHLY_TREND, REGIONS, RISK_COEFFS, SEASONAL_AMPLITUDE,
    SEED_CLAIMS, SEED_FRAUD, SEED_LAPSE, SEED_MEMBERS, SEED_NOTES,
    SEED_PROVIDERS, SERVICE_CATEGORIES, SPECIALTIES, UPCODE_SHIFT, COST_SCALE,
    month_label,
)

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PLAN_PREMIUM = {"Bronze": 310.0, "Silver": 430.0, "Gold": 585.0, "Platinum": 760.0}
PLAN_ACTUARIAL_VALUE = {"Bronze": 0.60, "Silver": 0.70, "Gold": 0.80, "Platinum": 0.90}


# =====================================================================
#  Members
# =====================================================================
def generate_members(rng):
    n = N_MEMBERS
    plan = rng.choice(PLANS, size=n, p=[0.28, 0.37, 0.24, 0.11])
    product = rng.choice(PRODUCTS, size=n, p=[0.44, 0.41, 0.15])
    region = rng.choice(REGIONS, size=n, p=[0.24, 0.21, 0.19, 0.20, 0.16])
    channel = rng.choice(CHANNELS, size=n, p=[0.52, 0.21, 0.19, 0.08])

    age = np.clip(rng.normal(44, 16, n), 18, 89).astype(int)
    income = np.clip(rng.lognormal(10.9, 0.52, n), 14_000, 400_000)
    chronic = rng.poisson(np.clip(0.35 + (age - 40) * 0.021, 0.05, None), n)

    premium = np.array([PLAN_PREMIUM[p] for p in plan])
    premium_burden = np.clip(premium * 12 / income, 0.01, 0.60)

    # A variable with ZERO true effect on cost that correlates with region,
    # so the fairness audit has something real to find nothing in.
    region_base = {r: v for r, v in zip(sorted(REGIONS),
                                        [0.32, 0.55, 0.61, 0.28, 0.47])}
    proxy = np.clip(np.array([region_base[r] for r in region])
                    + rng.normal(0, 0.11, n), 0, 1)

    return pd.DataFrame({
        "member_id": [f"MEM{600000 + i}" for i in range(n)],
        "age": age,
        "age_band": np.clip(age // 10 * 10, 20, 80),
        "sex": rng.choice(["F", "M"], size=n),
        "annual_income": income.round(2),
        "plan": plan, "product": product, "region": region, "channel": channel,
        "monthly_premium": premium,
        "actuarial_value": [PLAN_ACTUARIAL_VALUE[p] for p in plan],
        "chronic_count": chronic,
        "premium_burden": premium_burden.round(4),
        NULL_PROXY_NAME: proxy.round(4),
        "enrolled_month": rng.integers(0, 8, n),      # staggered start
    })


# =====================================================================
#  Enrollment spans - where the censoring lives
# =====================================================================
def simulate_lapse(members, rng):
    """Month-by-month survival simulation with a known hazard.

    The hazard is a logistic function of member attributes using
    LAPSE_COEFFS, so a correctly specified Cox model should recover those
    coefficients up to sampling error. Members who never lapse before the
    window closes are RIGHT-CENSORED.
    """
    n = len(members)
    prem = members.premium_burden.to_numpy()
    chan_ind = (members.channel.to_numpy() == "individual").astype(float)
    start = members.enrolled_month.to_numpy()

    # low utilisation is decided up front, per member
    util_low = (rng.random(n) < 0.31).astype(float)
    # whether a member has experienced a denial is time-varying in reality;
    # here it is drawn once and its onset month recorded, so the hazard can
    # switch part-way through the span
    had_denial_ever = (rng.random(n) < 0.24)
    denial_month = np.where(had_denial_ever, rng.integers(1, N_MONTHS, n), 10**6)

    tenure = np.zeros(n, dtype=int)
    observed = np.zeros(n, dtype=int)
    cause = np.array([""] * n, dtype=object)

    cause_names = sorted(LAPSE_CAUSE_SHARES)
    cause_p = np.array([LAPSE_CAUSE_SHARES[c] for c in cause_names])
    cause_p = cause_p / cause_p.sum()

    active = np.ones(n, dtype=bool)
    for m in range(N_MONTHS):
        in_window = active & (start <= m)
        if not in_window.any():
            continue
        t = (m - start).astype(float)
        lin = (
            np.log(BASELINE_MONTHLY_HAZARD / (1 - BASELINE_MONTHLY_HAZARD))
            + LAPSE_COEFFS["premium_burden"] * (prem * 10)
            + LAPSE_COEFFS["had_denial"] * (m >= denial_month).astype(float)
            + LAPSE_COEFFS["tenure_months"] * t
            + LAPSE_COEFFS["utilisation_low"] * util_low
            + LAPSE_COEFFS["channel_individual"] * chan_ind
        )
        h = 1.0 / (1.0 + np.exp(-lin))
        draw = rng.random(n)
        lapsing = in_window & (draw < h)
        if lapsing.any():
            idx = np.where(lapsing)[0]
            tenure[idx] = m - start[idx]
            observed[idx] = 1
            cause[idx] = rng.choice(cause_names, size=len(idx), p=cause_p)
            active[idx] = False

    # everyone still active when the window closes is CENSORED
    still = np.where(active)[0]
    tenure[still] = OBSERVATION_END_MONTH - start[still]
    observed[still] = 0
    cause[still] = ""

    spans = pd.DataFrame({
        "member_id": members.member_id,
        "start_month": start,
        "tenure_months": np.maximum(tenure, 0),
        "event_observed": observed,
        "exit_cause": cause,
        "utilisation_low": util_low.astype(int),
        "denial_month": np.where(denial_month > N_MONTHS, -1, denial_month),
        "had_denial": (denial_month <= N_MONTHS).astype(int),
    })
    spans["end_month"] = spans.start_month + spans.tenure_months
    return spans


# =====================================================================
#  Providers, including the upcoding ring
# =====================================================================
def generate_providers(rng):
    n = N_PROVIDERS
    spec = rng.choice(SPECIALTIES, size=n,
                      p=[0.26, 0.21, 0.11, 0.12, 0.13, 0.10, 0.07])
    region = rng.choice(REGIONS, size=n)
    df = pd.DataFrame({
        "provider_id": [f"NPI{3000000 + i}" for i in range(n)],
        "specialty": spec, "region": region,
        "panel_size": rng.integers(120, 2600, n),
        "years_practising": rng.integers(1, 38, n),
        "is_fraud": 0,
    })

    # The ring is drawn only from the specialties that bill E/M codes
    # heavily - a "fraud ring" of oncologists who never bill 99213 would
    # be undetectable by design and would flatter the detector.
    eligible = df.index[df.specialty.isin(FRAUD_RING_SPECIALTIES)].to_numpy()
    ring = rng.choice(eligible, size=min(N_FRAUD_PROVIDERS, len(eligible)),
                      replace=False)
    df.loc[sorted(ring), "is_fraud"] = 1
    return df


# =====================================================================
#  Claim lines
# =====================================================================
def generate_claims(members, spans, providers, rng, fraud_rng):
    """One row per claim line, only within a member's enrolled months.

    Cost carries the injected regional trend, seasonality and the level
    shift, so the hierarchical forecasting study has a real signal to
    recover rather than noise to fit.
    """
    m = members.merge(spans[["member_id", "start_month", "end_month"]], on="member_id")
    prov = providers.reset_index(drop=True)
    prov_by_region = {r: prov.index[prov.region == r].to_numpy() for r in sorted(REGIONS)}
    prov_ids = prov["provider_id"].to_numpy()
    fraud_set = set(prov.loc[prov.is_fraud == 1, "provider_id"])

    # Expected CLAIM LINES per member-month, rising with chronic burden
    # and age.
    #
    # The first version used (1.4 + 0.85*chronic)/12, which reads like an
    # annual visit count divided into months and produced 0.2 lines per
    # member-month - 43k lines in total. That is not merely "small": at
    # that volume each of the 900 providers held about 16 professional
    # claims, and peer-relative coding intensity, which is the entire basis
    # of the upcoding detector, cannot be estimated from 16 observations.
    # The fraud study would have been measuring sampling noise.
    #
    # A real member generates roughly 12-25 claim LINES a year (a single
    # visit yields several: professional, lab, imaging, pharmacy), so the
    # monthly rate belongs near 1-2.
    lam = (12.0 + 7.0 * m.chronic_count.to_numpy()
           + 0.18 * (m.age.to_numpy() - 40).clip(0)) / 12.0

    rows = []
    cat_p = np.array([0.05, 0.19, 0.34, 0.07, 0.20, 0.08, 0.05, 0.02])
    cat_p = cat_p / cat_p.sum()

    # Pre-extract to arrays. `m.product` is NOT the product column - it
    # resolves to DataFrame.product(), the aggregation method - so attribute
    # access silently returned a function and failed only at .iat[]. Column
    # names that collide with DataFrame methods (product, count, min, max,
    # size) must be reached by bracket, and pulling everything into plain
    # numpy up front removes the whole class of mistake as well as being far
    # faster inside a loop this size.
    a_member = m["member_id"].to_numpy()
    a_start = m["start_month"].to_numpy(dtype=int)
    a_end = m["end_month"].to_numpy(dtype=int)
    a_region = m["region"].to_numpy()
    a_plan = m["plan"].to_numpy()
    a_product = m["product"].to_numpy()
    a_av = m["actuarial_value"].to_numpy(dtype=float)

    for i in range(len(m)):
        lo, hi = int(a_start[i]), int(a_end[i])
        if hi <= lo:
            continue
        region = a_region[i]
        pool = prov_by_region[region]
        if len(pool) == 0:
            continue
        n_months = hi - lo
        counts = rng.poisson(lam[i], n_months)
        for k, c in enumerate(counts):
            if c == 0:
                continue
            month = lo + k
            trend = (1 + REGION_MONTHLY_TREND[region]) ** month
            seasonal = 1 + SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (month % 12) / 12)
            shift = (1 + LEVEL_SHIFT_SIZE
                     if region == LEVEL_SHIFT_REGION and month >= LEVEL_SHIFT_MONTH
                     else 1.0)
            for _ in range(int(c)):
                pidx = int(pool[rng.integers(len(pool))])
                pid = prov_ids[pidx]
                cat = SERVICE_CATEGORIES[int(rng.choice(len(SERVICE_CATEGORIES), p=cat_p))]

                # Professional visits carry an E/M code - the surface the
                # upcoding ring operates on.
                if cat == "professional":
                    true_level = int(np.clip(rng.binomial(3, 0.38), 0, 3))
                    billed_level = true_level
                    if pid in fraud_set and fraud_rng.random() < UPCODE_SHIFT:
                        billed_level = min(true_level + 1, len(EM_CODES) - 1)
                    em = EM_CODES[billed_level]
                    base = EM_ALLOWED_AMOUNT[em]
                    minutes = EM_MINUTES[em]
                    true_em = EM_CODES[true_level]
                else:
                    em, true_em, minutes = "", "", 0
                    base = float(rng.lognormal(
                        {"inpatient": 8.6, "outpatient": 6.4, "emergency": 7.1,
                         "pharmacy": 4.1, "lab": 3.9, "imaging": 5.7,
                         "behavioral_health": 4.9}.get(cat, 5.0), 0.75))

                allowed = base * trend * seasonal * shift * COST_SCALE
                status = "paid"
                reason = ""
                r = rng.random()
                if r < 0.081:
                    status = "denied"
                    reason = DENIAL_REASONS[int(rng.integers(len(DENIAL_REASONS)))]
                elif r < 0.106:
                    status = "pended"

                paid = 0.0 if status != "paid" else allowed * a_av[i]
                rows.append((
                    a_member[i], pid, month, month_label(month), cat,
                    PLACE_OF_SERVICE[int(rng.integers(len(PLACE_OF_SERVICE)))],
                    em, true_em, minutes, round(allowed, 2), round(paid, 2),
                    status, reason, region, a_plan[i], a_product[i],
                ))
        if len(rows) > 900_000:
            break                       # hard ceiling, not the intended path

    df = pd.DataFrame(rows, columns=[
        "member_id", "provider_id", "month_index", "month", "service_category",
        "place_of_service", "em_code", "true_em_code", "encounter_minutes",
        "allowed_amount", "paid_amount", "status", "denial_reason",
        "region", "plan", "product"])
    df.insert(0, "claim_line_id", [f"CL{i:08d}" for i in range(len(df))])
    return df


def main():
    m_rng = np.random.default_rng(SEED_MEMBERS)
    p_rng = np.random.default_rng(SEED_PROVIDERS)
    c_rng = np.random.default_rng(SEED_CLAIMS)
    l_rng = np.random.default_rng(SEED_LAPSE)
    f_rng = np.random.default_rng(SEED_FRAUD)

    members = generate_members(m_rng)
    spans = simulate_lapse(members, l_rng)
    providers = generate_providers(p_rng)
    claims = generate_claims(members, spans, providers, c_rng, f_rng)

    censoring_rate = float((spans.event_observed == 0).mean())
    upcoded = int((claims.em_code != claims.true_em_code).sum())
    prof = int((claims.service_category == "professional").sum())

    truth = {
        "lapse_coeffs": LAPSE_COEFFS,
        "baseline_monthly_hazard": BASELINE_MONTHLY_HAZARD,
        "lapse_cause_shares": LAPSE_CAUSE_SHARES,
        "censoring_rate": round(censoring_rate, 4),
        "n_events_observed": int(spans.event_observed.sum()),
        "fraud_provider_ids": sorted(providers.loc[providers.is_fraud == 1,
                                                   "provider_id"]),
        "n_fraud_providers": int(providers.is_fraud.sum()),
        "upcode_shift": UPCODE_SHIFT,
        "n_upcoded_lines": upcoded,
        "n_professional_lines": prof,
        "region_monthly_trend": REGION_MONTHLY_TREND,
        "seasonal_amplitude": SEASONAL_AMPLITUDE,
        "level_shift": {"region": LEVEL_SHIFT_REGION, "month": LEVEL_SHIFT_MONTH,
                        "size": LEVEL_SHIFT_SIZE},
        "risk_coeffs": RISK_COEFFS,
        "null_proxy_name": NULL_PROXY_NAME,
        "null_proxy_true_effect": NULL_PROXY_TRUE_EFFECT,
        "note": ("analysis code may read these ONLY to score itself; nothing "
                 "downstream may consult them to make a decision"),
    }

    members.to_csv(DATA / "members.csv", index=False)
    spans.to_csv(DATA / "enrollment_spans.csv", index=False)
    providers.to_csv(DATA / "providers.csv", index=False)
    # true_em_code is the answer key for the fraud study and is split out so
    # the detector cannot consume it by accident
    claims.drop(columns=["true_em_code"]).to_csv(DATA / "claim_lines.csv", index=False)
    claims[["claim_line_id", "em_code", "true_em_code"]].to_csv(
        DATA / "claim_scoring.csv", index=False)
    with open(DATA / "ground_truth_payer.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    print(f"MEMBERS    {len(members):,} across {len(PLANS)} plans, "
          f"{len(REGIONS)} regions")
    print(f"SPANS      {truth['n_events_observed']:,} lapses observed, "
          f"{censoring_rate:.1%} RIGHT-CENSORED (still enrolled at cut-off)")
    print(f"           exit causes: "
          f"{spans[spans.event_observed == 1].exit_cause.value_counts().to_dict()}")
    print(f"PROVIDERS  {len(providers):,}, {truth['n_fraud_providers']} in the "
          f"upcoding ring ({100 * truth['n_fraud_providers'] / len(providers):.1f}%)")
    print(f"CLAIMS     {len(claims):,} lines, {prof:,} professional, "
          f"{upcoded:,} upcoded ({100 * upcoded / max(prof, 1):.1f}% of professional)")
    print(f"           denial rate {100 * (claims.status == 'denied').mean():.2f}%, "
          f"total allowed ${claims.allowed_amount.sum():,.0f}")
    print(f"TREND      level shift in {LEVEL_SHIFT_REGION} at month "
          f"{LEVEL_SHIFT_MONTH} (+{LEVEL_SHIFT_SIZE:.0%})")


if __name__ == "__main__":
    main()
