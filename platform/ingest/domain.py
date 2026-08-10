"""
domain.py
------------
The single source of truth for Aegis Health Plan, and for every mechanism
deliberately injected into its data.

Aegis is a health insurance payer: it enrols members, contracts with
providers, adjudicates the claims those providers submit, and answers to a
regulator for how it reports all of it. That shape is why one project
legitimately needs a governed metrics layer, survival analysis,
hierarchical forecasting, fraud detection and validated QA - each has a
real owner inside the business:

  ACTUARIAL / FINANCE   loss ratio and PMPM must mean exactly one thing,
                        because a restated quarter is a regulatory event
                        rather than an inconvenience
  MEMBERSHIP            members lapse, and most of them have NOT lapsed by
                        the time you look - the ones still enrolled are
                        censored observations, not negatives
  PLANNING              claims cost is forecast at plan, region and
                        product level, and those levels have to sum
  SPECIAL INVESTIGATIONS providers who upcode adapt when you catch them,
                        so a detector's strength is unknown until it has
                        been attacked
  COMPLIANCE            every automated decision needs an audit trail and
                        a traceable test

WHAT IS INJECTED, AND WHY IT IS KEPT SEPARATE
---------------------------------------------
Each generator writes its mechanism into a ground_truth_*.json. Analysis
code may read those values ONLY to score itself, never to make a decision.
Three earlier projects declared a ground truth that was never actually
present in the data - a concept drift that was only a column label, a
price elasticity demand never responded to, a fairness proxy correlated
0.04 - and each was caught only because something scored itself against
the claimed mechanism and got an impossible answer.

Seeds are split per concern so regenerating claims does not reshuffle
members, and every collection feeding an RNG is sorted: Projects 10 and 13
both shipped fixed seeds that were still not reproducible across
processes, because `set` iteration order and builtin `hash()` vary per
run.
"""

from datetime import date

# ---------------------------------------------------------------- seeds
SEED_MEMBERS = 1501
SEED_PROVIDERS = 1502
SEED_CLAIMS = 1503
SEED_LAPSE = 1504
SEED_FRAUD = 1505
SEED_NOTES = 1506
SEED_FORECAST = 1507

# ------------------------------------------------------------- calendar
# 24 months of history. Fixed origin so results never depend on when the
# pipeline is run - a project that embeds datetime.now() in its outputs
# can never be checksum-verified.
START_DATE = date(2024, 1, 1)
N_MONTHS = 24
OBSERVATION_END_MONTH = N_MONTHS - 1      # anything still active here is CENSORED

# --------------------------------------------------------------- scale
N_MEMBERS = 24_000
N_PROVIDERS = 900
N_CLAIM_LINES_TARGET = 380_000

# Scales allowed amounts so the book runs at a realistic loss ratio.
#
# Without it the generated book ran at 1.42 - paying $1.42 in claims per
# $1.00 of premium, which is not a hard business, it is an insolvent one,
# and any analyst reading the dashboard would stop there. Real commercial
# payers sit near 0.80-0.90 (US ACA rules effectively floor it at 0.80).
#
# Scaling COST rather than VOLUME is deliberate: claim-line counts are what
# give each provider enough professional claims for peer-relative coding
# intensity to be estimable, and cutting volume to fix the ratio would have
# quietly destroyed the fraud study to fix a finance number.
COST_SCALE = 0.60

# --------------------------------------------------------------- enums
# sorted() wherever these feed an RNG.
PLANS = ["Bronze", "Silver", "Gold", "Platinum"]
PRODUCTS = ["HMO", "PPO", "EPO"]
REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["employer", "individual", "exchange", "medicare_advantage"]

SERVICE_CATEGORIES = ["inpatient", "outpatient", "professional", "emergency",
                      "pharmacy", "lab", "imaging", "behavioral_health"]
PLACE_OF_SERVICE = ["office", "hospital_inpatient", "hospital_outpatient",
                    "emergency_room", "telehealth", "home"]

# Evaluation & Management codes, ordered by intensity. Upcoding means
# billing a higher-intensity code than the encounter justifies, so the
# ORDER here is semantically meaningful and the fraud detector reasons
# about position, not identity.
EM_CODES = ["99212", "99213", "99214", "99215"]
EM_INTENSITY = {c: i for i, c in enumerate(EM_CODES)}
EM_ALLOWED_AMOUNT = {"99212": 62.0, "99213": 104.0, "99214": 158.0, "99215": 212.0}

SPECIALTIES = ["family_medicine", "internal_medicine", "cardiology",
               "orthopedics", "psychiatry", "dermatology", "oncology"]

CLAIM_STATUS = ["paid", "denied", "pended"]
DENIAL_REASONS = ["no_prior_auth", "not_covered", "duplicate",
                  "coordination_of_benefits", "medical_necessity"]

LAPSE_CAUSES = ["voluntary_lapse", "switched_plan", "aged_out", "deceased"]

# =====================================================================
#  Injected mechanism 1 - member lapse hazard (survival analysis)
# =====================================================================
# A member's monthly hazard of lapsing. The coefficients below are the
# ANSWER KEY for analytics/survival: a Cox model fitted on the enrollment
# spans should recover these, and a naive logistic regression that ignores
# censoring should not.
#
# Most members are still enrolled at OBSERVATION_END_MONTH. They are
# censored - "has not lapsed YET" - and treating them as negatives is the
# specific error this module exists to quantify.
BASELINE_MONTHLY_HAZARD = 0.018
LAPSE_COEFFS = {
    "premium_burden": 0.95,       # premium as share of income - strongest driver
    "had_denial": 0.55,           # a denied claim materially raises lapse risk
    "tenure_months": -0.030,      # loyalty accumulates
    "utilisation_low": 0.40,      # members who never use it stop paying for it
    "channel_individual": 0.35,   # individual market churns harder than employer
}
# Competing risks: not every exit is a lapse, and lumping them together
# biases the lapse hazard. Shares must sum to 1.
LAPSE_CAUSE_SHARES = {"voluntary_lapse": 0.62, "switched_plan": 0.27,
                      "aged_out": 0.08, "deceased": 0.03}

# =====================================================================
#  Injected mechanism 2 - provider upcoding ring (fraud detection)
# =====================================================================
# A set of providers who systematically bill one E/M level higher than the
# encounter warrants. Detection must use BEHAVIOUR only; is_fraud exists
# solely to score.
#
# UPCODE_SHIFT is the probability a ring provider bumps a given claim up a
# level. Deliberately not 1.0: a provider who upcodes every single claim
# is trivially detectable and teaches nothing.
N_FRAUD_PROVIDERS = 22
UPCODE_SHIFT = 0.42
FRAUD_RING_SPECIALTIES = ["family_medicine", "internal_medicine"]
# Impossible-day rule: a provider billing more encounter-minutes than
# exist in a working day. Ring members occasionally trip this.
IMPOSSIBLE_DAY_MINUTES = 600
EM_MINUTES = {"99212": 10, "99213": 15, "99214": 25, "99215": 40}

# =====================================================================
#  Injected mechanism 3 - claims cost trend (hierarchical forecasting)
# =====================================================================
# Real monthly cost trend per region, plus a seasonal shape. The hierarchy
# is region x product, and the levels MUST sum: the forecast reconciliation
# study is scored against these.
REGION_MONTHLY_TREND = {"North": 0.0042, "South": 0.0031, "East": 0.0068,
                        "West": 0.0025, "Central": 0.0049}
SEASONAL_AMPLITUDE = 0.11          # winter utilisation bump
# One region gets a genuine level shift partway through - a contract change
# - so a forecaster that assumes a stable trend is measurably wrong.
LEVEL_SHIFT_REGION = "East"
LEVEL_SHIFT_MONTH = 15
LEVEL_SHIFT_SIZE = 0.14

# =====================================================================
#  Injected mechanism 4 - risk / cost model signal
# =====================================================================
# True drivers of a member's annual cost. The risk model in analytics/risk
# is scored on CALIBRATION against these, not only on ranking: a model that
# orders members correctly but is systematically 30% low prices every
# contract wrong.
RISK_COEFFS = {
    "age_band": 0.052,
    "chronic_count": 0.610,
    "prior_year_cost_log": 0.480,
    "inpatient_prior": 0.390,
}
# A variable with ZERO true effect on cost that correlates with region.
# The fairness audit must find it carries no signal, exactly as Project 11
# did with a lending proxy - re-tested here on a different population
# rather than assumed to repeat.
NULL_PROXY_NAME = "zip_deprivation_index"
NULL_PROXY_TRUE_EFFECT = 0.0

# ------------------------------------------------------- ground truth keys
GROUND_TRUTH_KEYS = [
    "lapse_coeffs", "baseline_monthly_hazard", "lapse_cause_shares",
    "fraud_provider_ids", "upcode_shift",
    "region_monthly_trend", "level_shift", "risk_coeffs",
    "null_proxy_true_effect", "censoring_rate",
]


def month_label(m: int) -> str:
    """Month index -> 'YYYY-MM'. Integer arithmetic on a fixed origin
    rather than date libraries, so windowing never depends on timezone or
    DST behaviour."""
    y = START_DATE.year + (START_DATE.month - 1 + m) // 12
    mo = (START_DATE.month - 1 + m) % 12 + 1
    return f"{y:04d}-{mo:02d}"


def month_index(label: str) -> int:
    y, mo = (int(x) for x in label.split("-"))
    return (y - START_DATE.year) * 12 + (mo - START_DATE.month)
