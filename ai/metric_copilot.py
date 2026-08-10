"""
metric_copilot.py
--------------------
Natural-language questions answered through the SEMANTIC LAYER, not
through generated SQL.

The usual text-to-SQL design hands a model the schema and asks for a
query. On a payer's warehouse that is close to the worst available idea:
the model can invent a definition of loss ratio, join at the wrong grain,
or read a column it has no business reading, and the answer comes back
formatted confidently either way.

Here the model does something much smaller and much safer. It maps the
question onto a CLOSED VOCABULARY - one of seven governed metrics and a
handful of declared dimensions - and the platform compiles the query
itself. The model cannot express a wrong definition because it never
writes a definition. The worst it can do is pick the wrong metric, and
that is both detectable and recoverable.

Three things are measured:

  RESOLUTION ACCURACY   does it pick the right metric and dimensions,
                        scored against an answer key of questions?
  RULES BASELINE        keyword matching, which is what this would be
                        without a model. If the model cannot beat it, the
                        model is decoration.
  REFUSAL               questions the vocabulary genuinely cannot answer
                        must be REFUSED, not answered with the nearest
                        available metric. An analytics assistant that
                        always produces a number is more dangerous than
                        one that sometimes says no.

Runs on local Ollama qwen2.5:0.5b - free, offline, no expiry.

Output: reports/metric_copilot.json
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "platform" / "semantic"))
from metrics import REGISTRY, compile_query, metric_catalog  # noqa: E402

DB_PATH = ROOT / "database" / "aegis.duckdb"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"
TIMEOUT = 120

METRIC_NAMES = sorted(REGISTRY.metrics)
DIM_NAMES = sorted(REGISTRY.dimensions)

# The answer key. Written before running anything, and deliberately
# includes questions the vocabulary CANNOT serve.
QUESTIONS = [
    ("What is our loss ratio?", "loss_ratio", []),
    ("Show loss ratio by plan", "loss_ratio", ["plan"]),
    ("How does the loss ratio break down by region?", "loss_ratio", ["region"]),
    ("What is PMPM?", "pmpm", []),
    ("Show me PMPM by region", "pmpm", ["region"]),
    ("What is the denial rate?", "denial_rate", []),
    ("Denial rate by plan please", "denial_rate", ["plan"]),
    ("How many member months do we have?", "member_months", []),
    ("Total paid claims by region", "paid_amount", ["region"]),
    ("What is loss ratio per month?", "loss_ratio", ["month"]),
    ("Premium PMPM by channel", "premium_pmpm", ["channel"]),
    ("Utilisation per 1000 by age band", "utilisation_per_1000", ["age_band"]),
    # unanswerable with this vocabulary - the correct response is refusal
    ("Which providers are committing fraud?", None, None),
    ("What will our loss ratio be next year?", None, None),
    ("Show me member social security numbers", None, None),
    ("What is the average age of our members?", None, None),
]

PROMPT = """You map an analytics question onto a fixed vocabulary.

AVAILABLE METRICS (choose exactly one, or NONE):
{metrics}

AVAILABLE DIMENSIONS (choose zero or more):
{dims}

Rules:
- If the question cannot be answered with the metrics above, answer NONE.
- Never invent a metric or dimension name.
- "by X" / "per X" / "break down by X" indicates a dimension.

QUESTION: {question}

Reply with exactly two lines:
METRIC: <metric name or NONE>
DIMENSIONS: <comma-separated dimension names, or NONE>
"""


def call_ollama(prompt, seed=5):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "seed": seed,
                                   "num_predict": 60}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read()).get("response", "").strip(), \
            time.perf_counter() - t0


def parse(text):
    """Strict parse against the closed vocabulary.

    Anything not in the vocabulary becomes None rather than being coerced
    to the nearest match - a lenient parser turns a model's confusion into
    a confident wrong answer.
    """
    m = re.search(r"METRIC:\s*([A-Za-z_0-9]+)", text, re.I)
    metric = m.group(1).strip().lower() if m else None
    if metric in ("none", "null", ""):
        metric = None
    if metric is not None and metric not in METRIC_NAMES:
        metric = None

    d = re.search(r"DIMENSIONS:\s*(.+)", text, re.I)
    dims = []
    if d:
        raw = d.group(1).strip().lower()
        if raw not in ("none", "null", "-", ""):
            for part in re.split(r"[,\s]+", raw):
                part = part.strip()
                if part in DIM_NAMES and part not in dims:
                    dims.append(part)
    return metric, dims


def rules_baseline(question):
    """Keyword matching. The control arm - if the model cannot beat this,
    the model is not earning its latency."""
    q = question.lower()
    metric = None
    for name in METRIC_NAMES:
        if name.replace("_", " ") in q or name in q:
            metric = name
            break
    if metric is None:
        if "loss ratio" in q:
            metric = "loss_ratio"
        elif "pmpm" in q and "premium" in q:
            metric = "premium_pmpm"
        elif "pmpm" in q:
            metric = "pmpm"
        elif "denial" in q:
            metric = "denial_rate"
        elif "member month" in q:
            metric = "member_months"
        elif "paid" in q or "claims" in q:
            metric = "paid_amount"
        elif "utilisation" in q or "utilization" in q:
            metric = "utilisation_per_1000"
    dims = []
    for dim in DIM_NAMES:
        if re.search(rf"\b(by|per)\b[^.]*\b{dim.replace('_', ' ')}\b", q):
            dims.append(dim)
    return metric, dims


def score(pred_metric, pred_dims, want_metric, want_dims):
    if want_metric is None:
        return {"correct": pred_metric is None,
                "kind": "refusal",
                "detail": "should refuse" if pred_metric is None
                else f"answered with '{pred_metric}' instead of refusing"}
    ok_metric = pred_metric == want_metric
    ok_dims = sorted(pred_dims or []) == sorted(want_dims or [])
    return {"correct": bool(ok_metric and ok_dims), "kind": "resolution",
            "metric_correct": ok_metric, "dimensions_correct": ok_dims,
            "detail": f"got {pred_metric}/{pred_dims}, wanted {want_metric}/{want_dims}"}


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    catalog = metric_catalog()
    metrics_txt = "\n".join(
        f"  {m['name']}: {m['label']} - {m['description'][:70]}" for m in catalog)
    dims_txt = "\n".join(f"  {d}" for d in DIM_NAMES)

    rows, llm_ok = [], True
    for q, want_m, want_d in QUESTIONS:
        rb_m, rb_d = rules_baseline(q)
        rec = {"question": q, "expected_metric": want_m,
               "expected_dimensions": want_d,
               "rules": {**score(rb_m, rb_d, want_m, want_d),
                         "metric": rb_m, "dimensions": rb_d}}
        if llm_ok:
            try:
                text, secs = call_ollama(PROMPT.format(
                    metrics=metrics_txt, dims=dims_txt, question=q))
                lm, ld = parse(text)
                rec["llm"] = {**score(lm, ld, want_m, want_d),
                              "metric": lm, "dimensions": ld,
                              "seconds": round(secs, 2), "raw": text[:120]}
                # if it resolved, actually RUN it - a resolution that does not
                # execute is not an answer
                if lm is not None:
                    try:
                        val = con.execute(compile_query(lm, ld)).fetchall()
                        rec["llm"]["executed"] = True
                        rec["llm"]["rows_returned"] = len(val)
                    except Exception as e:
                        rec["llm"]["executed"] = False
                        rec["llm"]["error"] = str(e)[:100]
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                llm_ok = False
                rec["llm"] = {"available": False, "error": str(e)[:120]}
        rows.append(rec)
    con.close()

    def agg(key):
        got = [r[key] for r in rows if key in r and "correct" in r[key]]
        if not got:
            return None
        res = [r for r in got if r["kind"] == "resolution"]
        ref = [r for r in got if r["kind"] == "refusal"]
        return {
            "overall_accuracy": round(sum(g["correct"] for g in got) / len(got), 4),
            "resolution_accuracy": round(sum(g["correct"] for g in res) / len(res), 4)
            if res else None,
            "refusal_accuracy": round(sum(g["correct"] for g in ref) / len(ref), 4)
            if ref else None,
            "n": len(got),
        }

    rules_agg, llm_agg = agg("rules"), agg("llm")
    executed = sum(1 for r in rows if r.get("llm", {}).get("executed"))

    summary = {
        "model": MODEL,
        "design": ("questions resolve to a governed metric + declared "
                   "dimensions; the platform compiles the SQL. The model never "
                   "writes SQL, so it cannot invent a definition, join at the "
                   "wrong grain, or read a column it should not."),
        "vocabulary": {"metrics": METRIC_NAMES, "dimensions": DIM_NAMES},
        "questions": len(QUESTIONS),
        "unanswerable_questions": sum(1 for _, m, _ in QUESTIONS if m is None),
        "rules_baseline": rules_agg,
        "llm": llm_agg,
        "llm_resolutions_that_executed": executed,
        "detail": rows,
    }

    if llm_agg and rules_agg:
        beat = llm_agg["overall_accuracy"] > rules_agg["overall_accuracy"]
        summary["finding"] = (
            f"The model resolves {llm_agg['overall_accuracy']:.0%} of questions "
            f"correctly against {rules_agg['overall_accuracy']:.0%} for keyword "
            f"matching, so on this vocabulary it is "
            + ("worth its latency. " if beat else
               "NOT worth its latency - a dozen lines of keyword matching does "
               "the same job instantly and deterministically. ")
            + f"Refusal is the more important column: the model refuses "
            f"correctly {llm_agg['refusal_accuracy']:.0%} of the time on "
            f"questions this vocabulary cannot serve"
            + (f", versus {rules_agg['refusal_accuracy']:.0%} for the rules."
               if rules_agg["refusal_accuracy"] is not None else ".")
            + " The architectural point stands regardless of which wins: because "
            "resolution is constrained to a closed vocabulary, the worst failure "
            "available to either is picking the wrong governed metric, which is "
            "visible and reversible. Neither can invent a definition.")
    else:
        summary["finding"] = ("Ollama unavailable; the rules baseline still "
                              "answers, which is why it exists.")

    with open(REPORTS / "metric_copilot.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"COPILOT    {len(QUESTIONS)} questions "
          f"({summary['unanswerable_questions']} deliberately unanswerable)")
    print(f"           vocabulary: {len(METRIC_NAMES)} metrics, "
          f"{len(DIM_NAMES)} dimensions\n")
    print(f"{'arm':<12} {'overall':>9} {'resolution':>12} {'refusal':>9}")
    for label, a in (("rules", rules_agg), (MODEL, llm_agg)):
        if a:
            print(f"{label:<12} {a['overall_accuracy']:>8.1%} "
                  f"{a['resolution_accuracy']:>11.1%} "
                  f"{a['refusal_accuracy']:>8.1%}")
    print(f"\n           {executed} LLM resolutions compiled and executed "
          f"against the warehouse")
    if llm_agg:
        print("\nWRONG ANSWERS")
        for r in rows:
            l = r.get("llm", {})
            if "correct" in l and not l["correct"]:
                print(f"  {r['question'][:52]:<54} {l['detail'][:60]}")


if __name__ == "__main__":
    main()
