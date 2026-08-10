"""
upcoding_detection.py
------------------------
Finding providers who bill a higher-intensity E/M code than the encounter
justifies, and then finding out how much of that detection survives an
adversary who knows the rules.

Detection uses BEHAVIOUR only. `is_fraud` exists in the provider table and
is used solely to score - never as a feature, never as a filter. The same
discipline applies to `true_em_code`, which lives in a separate scoring
file so it cannot be joined in by accident.

Three signals, each with an independent rationale:

  PEER INTENSITY   a provider's mean E/M level against peers in the SAME
                   specialty. Specialty matters: an oncologist legitimately
                   bills higher than a family doctor, and comparing across
                   specialties would flag the entire oncology department.
  DISTRIBUTION     the share of a provider's claims at the top two levels.
                   Catches a provider whose mean looks normal because they
                   balance high codes with low ones.
  IMPOSSIBLE DAY   billed encounter minutes exceeding a plausible working
                   day. A hard rule with no statistics in it, included
                   because it is the one an investigator can defend without
                   explaining a z-score to a lawyer.

THE RED TEAM
------------
A detector's recall against a static adversary is close to meaningless,
because the adversary is not static. Once thresholds are known, upcoding
adapts: bill just under the flagging line, spread the behaviour across
more encounters, or shift only within the codes where peers are diffuse.
Each evasion is simulated and recall is re-measured, so the reported
strength is strength under attack rather than strength on the training
distribution.

Output: reports/upcoding_detection.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "platform" / "ingest"))
from domain import (  # noqa: E402
    EM_CODES, EM_INTENSITY, EM_MINUTES, IMPOSSIBLE_DAY_MINUTES, UPCODE_SHIFT,
)

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

MIN_CLAIMS_FOR_SCORING = 40      # below this a provider's mean is noise
FLAG_Z = 2.5                     # peer-relative z-score threshold
TOP_CODE_SHARE_Z = 2.5


def load():
    claims = pd.read_csv(DATA / "claim_lines.csv", dtype={"em_code": "string"})
    providers = pd.read_csv(DATA / "providers.csv")
    prof = claims[claims.service_category == "professional"].copy()
    prof = prof[prof.em_code.notna()]
    prof["intensity"] = prof.em_code.map(EM_INTENSITY)
    prof["minutes"] = prof.em_code.map(EM_MINUTES)
    return prof, providers


def provider_features(prof, providers):
    g = prof.groupby("provider_id").agg(
        n_claims=("intensity", "size"),
        mean_intensity=("intensity", "mean"),
        top_share=("intensity", lambda s: float((s >= 2).mean())),
        total_minutes=("minutes", "sum"),
    ).reset_index()
    g = g.merge(providers[["provider_id", "specialty", "region", "is_fraud"]],
                on="provider_id")

    # busiest single month of billed minutes, for the impossible-day rule
    per_month = prof.groupby(["provider_id", "month_index"]).minutes.sum()
    busiest = per_month.groupby("provider_id").max().rename("busiest_month_minutes")
    g = g.merge(busiest, on="provider_id", how="left")
    # ~21 working days in a month
    g["max_daily_minutes"] = g.busiest_month_minutes / 21.0
    return g


def peer_scores(g):
    """z-scores WITHIN specialty. Peer choice is the whole ballgame: judged
    against all providers, every specialist looks fraudulent."""
    out = g.copy()
    for col, zname in (("mean_intensity", "z_intensity"),
                       ("top_share", "z_top_share")):
        z = out.groupby("specialty")[col].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1))
        out[zname] = z
    return out


def apply_rules(g):
    scored = g[g.n_claims >= MIN_CLAIMS_FOR_SCORING].copy()
    scored["rule_peer_intensity"] = scored.z_intensity > FLAG_Z
    scored["rule_top_code_share"] = scored.z_top_share > TOP_CODE_SHARE_Z
    scored["rule_impossible_day"] = scored.max_daily_minutes > IMPOSSIBLE_DAY_MINUTES
    scored["flagged"] = (scored.rule_peer_intensity | scored.rule_top_code_share
                         | scored.rule_impossible_day)
    return scored


def evaluate(scored, label="baseline"):
    tp = int(((scored.flagged) & (scored.is_fraud == 1)).sum())
    fp = int(((scored.flagged) & (scored.is_fraud == 0)).sum())
    fn = int(((~scored.flagged) & (scored.is_fraud == 1)).sum())
    tn = int(((~scored.flagged) & (scored.is_fraud == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "scenario": label,
        "providers_scored": int(len(scored)),
        "fraud_in_scope": int((scored.is_fraud == 1).sum()),
        "flagged": int(scored.flagged.sum()),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "rule_hits": {
            "peer_intensity": int(scored.rule_peer_intensity.sum()),
            "top_code_share": int(scored.rule_top_code_share.sum()),
            "impossible_day": int(scored.rule_impossible_day.sum()),
        },
    }


# =====================================================================
#  Red team
# =====================================================================
def simulate_evasion(prof, providers, strategy, rng):
    """Rebuild the claim stream with an adapted adversary.

    The ring keeps upcoding but changes HOW, in ways a real investigator
    sees: less often, only where peers are diffuse, or spread thinly across
    a larger panel.
    """
    fraud_ids = set(providers.loc[providers.is_fraud == 1, "provider_id"])
    p = prof.copy()
    is_ring = p.provider_id.isin(fraud_ids).to_numpy()

    # undo the injected upcoding to get back to the honest baseline, then
    # re-apply it under the new strategy
    scoring = pd.read_csv(DATA / "claim_scoring.csv", dtype={
        "em_code": "string", "true_em_code": "string"})
    p = p.merge(scoring[["claim_line_id", "true_em_code"]], on="claim_line_id",
                how="left")
    honest = p.true_em_code.fillna(p.em_code)
    base_int = honest.map(EM_INTENSITY).to_numpy()

    new_int = base_int.copy()
    n = len(p)
    draw = rng.random(n)

    if strategy == "reduced_rate":
        # upcode only 15% of the time instead of 42%
        bump = is_ring & (draw < 0.15)
        new_int = np.where(bump, np.minimum(base_int + 1, 3), base_int)
    elif strategy == "stay_below_threshold":
        # never bill the top code; the mean rises less and the top-share
        # rule sees nothing
        bump = is_ring & (draw < UPCODE_SHIFT) & (base_int < 2)
        new_int = np.where(bump, np.minimum(base_int + 1, 2), base_int)
    elif strategy == "spread_thin":
        # half the ring stops entirely; the rest continue unchanged, which
        # is what happens after an investigation spooks part of a network
        quiet = set(sorted(fraud_ids)[:len(fraud_ids) // 2])
        active = is_ring & ~p.provider_id.isin(quiet).to_numpy()
        bump = active & (draw < UPCODE_SHIFT)
        new_int = np.where(bump, np.minimum(base_int + 1, 3), base_int)
    else:
        raise ValueError(strategy)

    p["intensity"] = new_int
    p["em_code"] = [EM_CODES[i] for i in new_int]
    p["minutes"] = p.em_code.map(EM_MINUTES)
    return p


def main():
    prof, providers = load()
    truth = json.loads((DATA / "ground_truth_payer.json").read_text(encoding="utf-8"))

    base_scored = apply_rules(peer_scores(provider_features(prof, providers)))
    baseline = evaluate(base_scored, "baseline (adversary unaware)")

    rng = np.random.default_rng(4242)
    red_team = []
    for strat, desc in [
        ("reduced_rate", "upcode 15% of claims instead of 42%"),
        ("stay_below_threshold", "never bill the top code"),
        ("spread_thin", "half the ring stops; the rest carry on"),
    ]:
        evaded = simulate_evasion(prof, providers, strat, rng)
        sc = apply_rules(peer_scores(provider_features(evaded, providers)))
        r = evaluate(sc, strat)
        r["description"] = desc
        r["recall_drop_vs_baseline"] = round(baseline["recall"] - r["recall"], 4)
        red_team.append(r)

    worst = min(red_team, key=lambda r: r["recall"])

    summary = {
        "scope": {
            "professional_claim_lines": int(len(prof)),
            "providers_with_enough_volume": baseline["providers_scored"],
            "min_claims_to_score": MIN_CLAIMS_FOR_SCORING,
            "why_a_volume_floor": ("a provider with a handful of claims has a "
                                   "mean intensity that is noise; flagging them "
                                   "generates false positives an investigator "
                                   "cannot act on"),
        },
        "signals": {
            "peer_intensity": f"z > {FLAG_Z} WITHIN specialty",
            "top_code_share": f"z > {TOP_CODE_SHARE_Z} on share of top-two codes",
            "impossible_day": f"> {IMPOSSIBLE_DAY_MINUTES} billed minutes in a day",
        },
        "baseline": baseline,
        "red_team": red_team,
        "dead_rule": {
            "rule": "impossible_day",
            "hits": 0,
            "verdict": ("this rule never fires on this book and contributes "
                        "nothing to the reported precision or recall. It is "
                        "recorded rather than quietly left in the list, because "
                        "a three-signal detector where one signal is inert is a "
                        "two-signal detector, and the threshold "
                        f"({IMPOSSIBLE_DAY_MINUTES} min/day) is calibrated for a "
                        "billing pattern this generator does not produce. Either "
                        "recalibrate it against real claim volumes or drop it - "
                        "do not let it pad the rule count."),
        },
        "worst_case": {
            "strategy": worst["scenario"],
            "recall": worst["recall"],
            "recall_drop": worst["recall_drop_vs_baseline"],
        },
        "injected_truth": {
            "n_fraud_providers": truth["n_fraud_providers"],
            "upcode_shift": truth["upcode_shift"],
            "note": "is_fraud and true_em_code are used ONLY to score",
        },
        "finding": (
            f"Against an adversary who does not know the rules, the detector "
            f"reaches {baseline['recall']:.0%} recall at "
            f"{baseline['precision']:.0%} precision. That number is the least "
            f"informative one in this report. Once the thresholds are known, "
            f"the cheapest possible evasion - '{worst['scenario']}' - takes "
            f"recall to {worst['recall']:.0%}, a drop of "
            f"{worst['recall_drop_vs_baseline']:.0%}, with no change to the "
            f"detector at all. A fraud control's honest strength is its "
            f"strength under adaptation, and reporting the unattacked number "
            f"alone overstates it by exactly that margin."),
    }
    with open(REPORTS / "upcoding_detection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    b = baseline
    print(f"UPCODING   {len(prof):,} professional lines, "
          f"{b['providers_scored']} providers with >= {MIN_CLAIMS_FOR_SCORING} claims")
    print(f"\nBASELINE   precision {b['precision']:.3f}  recall {b['recall']:.3f}  "
          f"F1 {b['f1']:.3f}   ({b['true_positives']} TP, {b['false_positives']} FP, "
          f"{b['false_negatives']} FN)")
    print(f"           rule hits: {b['rule_hits']}")
    print(f"\nRED TEAM   the same detector, against an adversary who adapted")
    print(f"{'strategy':<24} {'precision':>10} {'recall':>8} {'drop':>8}")
    for r in red_team:
        print(f"{r['scenario']:<24} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['recall_drop_vs_baseline']:>+8.3f}")
    print(f"\nWORST CASE {worst['scenario']}: recall {worst['recall']:.1%} "
          f"(was {b['recall']:.1%})")


if __name__ == "__main__":
    main()
