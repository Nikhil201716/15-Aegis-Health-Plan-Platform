"""
risk_adjustment.py
---------------------
Predicting member cost, and being judged on whether the predictions are
RIGHT rather than merely well-ordered.

Risk adjustment is not a ranking problem. The output sets what a plan is
paid for a member, so a model that orders members perfectly but is
systematically 30% low prices every contract wrong and loses money on all
of them simultaneously. Discrimination (AUC, Gini) cannot see that;
calibration can, and it is the metric this module leads with.

Evaluated here:

  DISCRIMINATION   Spearman rank correlation and decile lift - can the
                   model tell an expensive member from a cheap one?
  CALIBRATION      predicted vs actual by decile, calibration slope and
                   intercept, and the payment error that follows from
                   miscalibration. A slope below 1 means the model is
                   compressed toward the mean and systematically
                   underpays the sick and overpays the healthy.
  FAIRNESS         a variable with ZERO true effect on cost is included in
                   the data and correlates with region. Project 11 found
                   such a proxy did not drive disparate outcomes once real
                   signal was present; this re-tests that claim on a
                   different population rather than assuming it repeats.

The split is TIME-BASED. A random split lets a member's own later months
inform their earlier ones, and in a book where cost is highly
autocorrelated that inflates every metric.

Output: reports/risk_adjustment.json
"""

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "platform" / "ingest"))
from domain import NULL_PROXY_NAME, NULL_PROXY_TRUE_EFFECT, RISK_COEFFS  # noqa: E402

DB_PATH = ROOT / "database" / "aegis.duckdb"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TRAIN_MONTHS = 12          # predict months 12-23 from months 0-11
FEATURES = ["age_band", "chronic_count", "prior_cost_log", "prior_inpatient",
            "prior_lines", NULL_PROXY_NAME]


def build_dataset(con):
    """One row per member: features from the first year, target from the
    second. Point-in-time by construction - nothing from the target window
    can enter a feature."""
    return con.execute(f"""
        WITH prior AS (
            SELECT member_id,
                   SUM(paid_amount)                          AS prior_cost,
                   SUM(total_lines)                          AS prior_lines,
                   COUNT(*)                                  AS prior_months
            FROM vw_member_month WHERE month_index < {TRAIN_MONTHS}
            GROUP BY 1
        ),
        prior_ip AS (
            SELECT member_id, COUNT(*) AS prior_inpatient
            FROM fct_claim_line
            WHERE month_index < {TRAIN_MONTHS} AND service_category = 'inpatient'
            GROUP BY 1
        ),
        target AS (
            SELECT member_id,
                   SUM(paid_amount)                          AS future_cost,
                   COUNT(*)                                  AS future_months
            FROM vw_member_month WHERE month_index >= {TRAIN_MONTHS}
            GROUP BY 1
        )
        SELECT m.member_id, m.age_band, m.chronic_count, m.region, m.plan,
               m.{NULL_PROXY_NAME},
               COALESCE(p.prior_cost, 0)        AS prior_cost,
               COALESCE(p.prior_lines, 0)       AS prior_lines,
               COALESCE(pi.prior_inpatient, 0)  AS prior_inpatient,
               t.future_cost, t.future_months
        FROM dim_member m
        JOIN prior p   ON p.member_id = m.member_id
        JOIN target t  ON t.member_id = m.member_id
        LEFT JOIN prior_ip pi ON pi.member_id = m.member_id
        WHERE t.future_months >= 3
    """).df()


def calibration(y_true, y_pred, n_bins=10):
    """Predicted vs actual by decile of prediction, plus the slope of a
    regression of actual on predicted. Slope 1.0 is perfect; below 1 means
    the model is compressed toward the mean."""
    order = np.argsort(y_pred)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, b in enumerate(bins):
        rows.append({
            "decile": i + 1,
            "n": int(len(b)),
            "mean_predicted": round(float(y_pred[b].mean()), 2),
            "mean_actual": round(float(y_true[b].mean()), 2),
            "ratio": round(float(y_pred[b].mean() / y_true[b].mean()), 4)
            if y_true[b].mean() else None,
        })
    if np.ptp(y_pred) == 0:      # a constant prediction has no slope
        return rows, float("nan"), float(y_pred[0])
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return rows, float(slope), float(intercept)


def evaluate(name, y_true, y_pred):
    rows, slope, intercept = calibration(y_true, y_pred)
    # A constant predictor has no rank correlation - scipy returns nan, and
    # `json.dump` writes a bare NaN token, which is not valid JSON and fails
    # only later, in the browser. Every NaN is converted at the boundary.
    rho = (None if np.ptp(y_pred) == 0
           else float(spearmanr(y_true, y_pred).statistic))
    top = np.argsort(y_pred)[-len(y_pred) // 10:]
    lift = float(y_true[top].mean() / y_true.mean())
    # what miscalibration costs: total predicted vs total actual spend
    payment_error = float((y_pred.sum() - y_true.sum()) / y_true.sum())
    return {
        "model": name,
        "spearman_rank_correlation": round(rho, 4) if rho is not None else None,
        "top_decile_lift": round(lift, 3),
        "r2": round(float(1 - ((y_true - y_pred) ** 2).sum()
                          / ((y_true - y_true.mean()) ** 2).sum()), 4),
        "mae": round(float(np.abs(y_true - y_pred).mean()), 2),
        "calibration_slope": (None if not np.isfinite(slope) else round(slope, 4)),
        "calibration_intercept": round(intercept, 2),
        "well_calibrated": bool(np.isfinite(slope) and 0.9 <= slope <= 1.1),
        "aggregate_payment_error_pct": round(payment_error * 100, 3),
        "calibration_by_decile": rows,
    }


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    d = build_dataset(con)
    con.close()

    d["prior_cost_log"] = np.log1p(d.prior_cost)
    X = d[FEATURES].to_numpy(dtype=float)
    y = d.future_cost.to_numpy(dtype=float)

    # split by member id hash-free: first 70% by prior cost rank would leak
    # the target's driver, so split randomly with a fixed seed on MEMBERS,
    # while the TIME split above is what prevents outcome leakage
    rng = np.random.default_rng(1509)
    idx = rng.permutation(len(d))
    cut = int(len(d) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    ridge = Ridge(alpha=1.0).fit(X[tr], y[tr])
    gbm = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                    max_depth=3, learning_rate=0.05).fit(X[tr], y[tr])
    mean_only = np.repeat(y[tr].mean(), len(te))

    results = [
        evaluate("mean_baseline", y[te], mean_only),
        evaluate("ridge", y[te], ridge.predict(X[te])),
        evaluate("gradient_boosting", y[te], gbm.predict(X[te])),
    ]

    # ---- fairness: does the null proxy carry any signal at all?
    proxy_i = FEATURES.index(NULL_PROXY_NAME)
    without = [f for f in FEATURES if f != NULL_PROXY_NAME]
    Xw = d[without].to_numpy(dtype=float)
    gbm_w = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                      max_depth=3, learning_rate=0.05).fit(Xw[tr], y[tr])
    with_proxy = evaluate("gbm_with_proxy", y[te], gbm.predict(X[te]))
    without_proxy = evaluate("gbm_without_proxy", y[te], gbm_w.predict(Xw[te]))

    # prediction parity across regions: does the model systematically
    # over- or under-predict for any region?
    preds = gbm.predict(X)
    by_region = {}
    for r in sorted(d.region.unique()):
        m = (d.region == r).to_numpy()
        by_region[r] = {
            "n": int(m.sum()),
            "mean_predicted": round(float(preds[m].mean()), 2),
            "mean_actual": round(float(y[m].mean()), 2),
            "prediction_ratio": round(float(preds[m].mean() / y[m].mean()), 4),
        }
    ratios = [v["prediction_ratio"] for v in by_region.values()]
    disparity = round(max(ratios) / min(ratios), 4)

    summary = {
        "n_members": int(len(d)),
        "split": f"features from months 0-{TRAIN_MONTHS - 1}, "
                 f"target from months {TRAIN_MONTHS}+",
        "why_time_split": ("a random split over member-months lets a member's "
                           "later cost inform their earlier features; cost is "
                           "highly autocorrelated so every metric inflates"),
        "models": results,
        "fairness": {
            "null_proxy": NULL_PROXY_NAME,
            "true_effect_on_cost": NULL_PROXY_TRUE_EFFECT,
            "gbm_r2_with_proxy": with_proxy["r2"],
            "gbm_r2_without_proxy": without_proxy["r2"],
            "r2_cost_of_removing_it": round(with_proxy["r2"] - without_proxy["r2"], 5),
            "prediction_parity_by_region": by_region,
            "max_min_prediction_ratio": disparity,
            "verdict": ("the proxy has zero true effect by construction; the "
                        "measured cost of removing it is the number that decides "
                        "whether keeping it was ever justified"),
        },
        "injected_truth": {"risk_coeffs": RISK_COEFFS},
        "finding": "",
    }

    best = max(results, key=lambda r: r["r2"])
    cal = [r for r in results if r["model"] != "mean_baseline"]
    summary["finding"] = (
        f"Ranking and pricing are different jobs. {best['model']} reaches "
        f"Spearman {best['spearman_rank_correlation']:.3f} with a top-decile "
        f"lift of {best['top_decile_lift']:.2f}x, so it orders members well. "
        f"Its calibration slope is {best['calibration_slope']:.3f}: "
        + ("within tolerance, so the ranking can be priced directly."
           if best["well_calibrated"] else
           "outside the 0.9-1.1 band, meaning predictions are compressed "
           "toward the mean - the sickest members are underpriced and the "
           "healthiest overpriced, by a margin AUC would never reveal.")
        + f" Aggregate payment error is "
        f"{best['aggregate_payment_error_pct']:+.2f}%. Removing the "
        f"zero-effect proxy costs {summary['fairness']['r2_cost_of_removing_it']:+.5f} "
        f"R-squared, which is the entire case for keeping it.")

    with open(REPORTS / "risk_adjustment.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"RISK MODEL {len(d):,} members, {summary['split']}\n")
    print(f"{'model':<20} {'R2':>8} {'spearman':>9} {'lift':>7} "
          f"{'cal slope':>10} {'calibrated':>11} {'pay err%':>9}")
    for r in results:
        sp = f"{r['spearman_rank_correlation']:.4f}" if r['spearman_rank_correlation'] is not None else "-"
        cs = f"{r['calibration_slope']:.4f}" if r['calibration_slope'] is not None else "-"
        print(f"{r['model']:<20} {r['r2']:>8.4f} {sp:>9} "
              f"{r['top_decile_lift']:>7.2f} {cs:>10} "
              f"{str(r['well_calibrated']):>11} {r['aggregate_payment_error_pct']:>+9.2f}")

    print(f"\nCALIBRATION by decile ({best['model']})")
    print(f"  {'decile':>7} {'predicted':>12} {'actual':>12} {'ratio':>8}")
    for row in best["calibration_by_decile"]:
        print(f"  {row['decile']:>7} {row['mean_predicted']:>12,.0f} "
              f"{row['mean_actual']:>12,.0f} {row['ratio']:>8.3f}")

    f_ = summary["fairness"]
    print(f"\nFAIRNESS   '{NULL_PROXY_NAME}' has TRUE effect {NULL_PROXY_TRUE_EFFECT}")
    print(f"           R2 with {f_['gbm_r2_with_proxy']:.4f} / without "
          f"{f_['gbm_r2_without_proxy']:.4f} -> cost of removal "
          f"{f_['r2_cost_of_removing_it']:+.5f}")
    print(f"           prediction parity max/min ratio across regions: {disparity}")


if __name__ == "__main__":
    main()
