# Aegis — Health Plan Intelligence Platform

A health insurance payer's analytics platform: 24,000 members, 900
providers, 367,313 claim lines over 24 months, and a governed semantic
layer that every number on every screen compiles from.

The domain is chosen because a payer genuinely needs all five disciplines
at once — this is one business, not five demos stapled together:

| Discipline | What it owns here |
|---|---|
| **Data analyst** | metric definitions, loss ratio by plan, cohort and funnel analysis |
| **Data engineer** | warehouse at a declared grain, the semantic layer, exposure joins |
| **QA / testing** | metric regression tests, requirement traceability, injected-defect scoring |
| **AI engineer** | natural language compiled to governed metrics, with refusal measured |
| **ML engineer** | survival models, hierarchical forecasting, risk adjustment with calibration |

Everything runs locally and free: DuckDB, pandas, scikit-learn, FastAPI,
and an optional local Ollama `qwen2.5:0.5b`.

```bash
python run_all.py               # build everything
python run_all.py --verify      # prove the data is byte-identical in a new process
python -m pytest tests/ -q      # 21 tests
python qa/validation.py         # traceability, audit trail, injected defects
python app/api/main.py          # console at http://127.0.0.1:8600
```

---

## Results

Every number below is read from a file in `reports/`, produced by code in
this repository. Where a result contradicts the hypothesis that motivated
the work, the contradiction is the result.

### The book of business

| | |
|---|---|
| Loss ratio | 0.8519 |
| PMPM (paid per member-month) | $384.59 |
| Member months | 266,329 |
| Loss ratio by plan | Bronze **1.059** · Silver 0.887 · Gold 0.752 · Platinum 0.625 |

Bronze running above 1.0 is adverse selection in the cheapest tier — a
real pattern an analyst can act on.

### Metric governance — the QA discipline nobody builds

A semantic layer makes one thing testable that is otherwise impossible:
the blast radius of a definition change.

Applying a change that sounds like a clarification — **excluding pharmacy
from loss ratio** — moves **33 of 50 published historical figures, all of
them materially**:

| Figure | Before | After | Change |
|---|---|---|---|
| Bronze loss ratio | 1.0585 | 1.0345 | −2.27% |
| 2025-01 loss ratio | 0.9313 | 0.9094 | −2.35% |
| 2025-12 loss ratio | 0.9102 | 0.8885 | −2.38% |

Every one is a number someone has already read and acted on. The change is
defensible; making it without publishing this list is not. 5/5 invariants
pass, and 50/50 golden values reproduce with zero drift.

### Survival analysis — 35.6% of members are censored

Members still enrolled at the cut-off have not "not churned". Their lapse
time is unknown and greater than their observed tenure.

Four estimators, identical data, scored against the injected hazard:

| Covariate | Truth | Naive logistic | Cox PH | Discrete-time |
|---|---|---|---|---|
| premium burden | 0.950 | 1.5698 | 0.7720 | **0.9518** |
| had denial | 0.550 | 0.4767 | 0.2123 | **0.5329** |
| low utilisation | 0.400 | 0.6720 | 0.3683 | **0.4170** |
| individual channel | 0.350 | 0.5709 | 0.3180 | **0.3595** |
| **Mean abs error** | | **0.2965** | **0.1449** | **0.0113** |

The discrete-time hazard is **26× more accurate than the naive classifier
and 13× more accurate than Cox.** Cox is not automatically the right
survival model: when data arrive in monthly buckets, the discrete-time
person-month model matches the generating process and Cox is an
approximation to it.

Competing risks matter too: 1−KM claims 50.5% voluntary lapse, the
cumulative incidence function says 40.0%. Treating death as censoring
assumes those members could still have lapsed later.

Kaplan-Meier, the Cox partial likelihood and the log-rank test are
implemented directly rather than imported — the partial likelihood is
fifteen lines, and writing it is the difference between using survival
analysis and understanding it.

### Fraud detection — and what it's worth under attack

| Scenario | Precision | Recall |
|---|---|---|
| Baseline (adversary unaware) | 0.913 | **0.955** |
| Adversary upcodes 15% not 42% | 0.571 | **0.182** |
| Half the ring stops | 0.786 | 0.500 |
| Never bill the top code | 0.895 | 0.773 |

**A detector that looks 95.5% effective is 18.2% effective against the
cheapest possible adaptation**, with no change to the detector at all. The
unattacked number is the least informative one in the report.

One rule (`impossible_day`) fires zero times and is recorded as inert
rather than left to pad the rule count.

### Hierarchical forecasting

Independent forecasts at each level are incoherent by 0.49% — the regional
numbers and the total in the same board pack do not add up.

| Approach | MAPE % | Coherent |
|---|---|---|
| base (independent) | 13.479 | **no** |
| bottom-up | 13.341 | yes |
| **top-down** | **9.545** | yes |
| OLS reconciled | 13.568 | yes |
| MinT diagonal | 13.586 | yes |

This contradicts the usual expectation that MinT wins. It loses here, and
the by-level MAPE shows why: leaves run ~17% error against ~4% at the
total, so any method weighting leaf forecasts inherits that noise. That is
a property of *this* hierarchy — which is the argument for measuring
reconciliation on your own data rather than adopting whichever method the
literature reports as best.

### Risk adjustment — calibration, not ranking

| Model | R² | Spearman | Calib. slope | Calibrated | Payment error |
|---|---|---|---|---|---|
| mean baseline | −0.0000 | — | — | no | −0.07% |
| **ridge** | **0.1254** | 0.3476 | **1.0418** | yes | −0.73% |
| gradient boosting | 0.1127 | 0.3376 | 0.9670 | yes | −0.59% |

A model that ranks well but is miscalibrated prices every contract wrong,
and discrimination metrics cannot see it. Removing the zero-effect proxy
`zip_deprivation_index` **improves** R² by 0.0014 — it carries no signal,
exactly as injected, which is the entire case against keeping it.

### AI layer — natural language, tightly bounded

Questions resolve to a **governed metric plus declared dimensions**; the
platform compiles the SQL. The model never writes SQL, so it cannot invent
a definition, join at the wrong grain, or reach a column it shouldn't.

| Arm | Overall | Resolution | Refusal |
|---|---|---|---|
| **keyword rules** | **87.5%** | **91.7%** | **75.0%** |
| qwen2.5:0.5b | 25.0% | 33.3% | **0.0%** |

The 0.5B model loses decisively to a dozen lines of keyword matching, and
refuses **nothing** — it answered "show me member social security numbers"
with a utilisation metric.

The architecture is what saved it. Because resolution is constrained to a
closed vocabulary, the worst either arm can do is pick a wrong *governed*
metric. Neither could invent a definition or reach a PII column, which is
precisely what an unconstrained text-to-SQL assistant would have done.

### Validation and test engineering

- **Traceability**: 11/12 requirements mapped to specification, test and
  evidence. REQ-012 has no test and is listed as a **gap** rather than
  removed from the register — a matrix containing only the requirements
  you happened to test is a list of tests.
- **Audit trail**: hash-chained entries; a settled record was edited and
  the chain caught it at seq 1. A plain log would have accepted the edit.
- **Injected defects**: ten real bugs planted, suite run against each.

---

## Reproducibility

`python run_all.py --verify` regenerates the data in a **new process** and
compares SHA-256 checksums. The new process is the point: two earlier
projects in this portfolio had fixed seeds and were still non-deterministic
across runs (`rng.choice(list(a_set))`, `abs(hash(s))`), and both are
perfectly stable within a single process.

## Layout

```
platform/ingest/       domain.py (single source of truth) + generator
platform/warehouse/    DuckDB, declared grain, fan-out and idempotency checks
platform/semantic/     metrics as code - the governed definitions
integrity/             metric regression tests, upcoding detection + red team
analytics/survival/    Kaplan-Meier, Cox, discrete-time hazard, competing risks
analytics/forecasting/ hierarchical reconciliation (BU / TD / OLS / MinT)
analytics/risk/        risk adjustment with calibration and fairness
ai/                    NL -> governed metrics, with a rules control arm
qa/                    traceability, audit trail, injected-defect scoring
app/api|web/           console server and UI
tests/                 21 property and invariant tests
reports/               every number quoted above, as JSON
```

## Requirements

Python 3.11+, `pip install -r requirements.txt`. Ollama with
`qwen2.5:0.5b` is optional — the copilot's rules baseline runs without it,
and outperforms it.
