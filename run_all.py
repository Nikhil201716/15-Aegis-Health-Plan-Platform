"""
run_all.py
-------------
Builds Aegis from nothing, in dependency order, then checks it built the
same thing as last time.

    python run_all.py            full pipeline
    python run_all.py --verify   regenerate data in a NEW process, compare checksums
    python run_all.py --skip-llm skip the Ollama stage
    python run_all.py --skip-qa  skip injected-bug scoring (the slow stage)

--verify runs in a fresh process deliberately. Two earlier projects in this
portfolio shipped generators with fixed seeds that were still not
reproducible across runs:

    rng.choice(list(SOME_SET))    Project 10 - set iteration order
    abs(hash(customer_id))        Project 13 - builtin hash randomisation

Both are perfectly stable WITHIN one process, so a same-process re-run
cannot detect either. A seed is a claim about reproducibility; a checksum
taken in a new process is a measurement of it.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REPORTS = ROOT / "reports"

STAGES = [
    ("Generate payer data",        "platform/ingest/generate_payer_data.py",     "generate"),
    ("Build warehouse",            "platform/warehouse/build_warehouse.py",      "warehouse"),
    ("Metric regression testing",  "integrity/metric_regression.py",             "analysis"),
    ("Member survival analysis",   "analytics/survival/member_survival.py",      "analysis"),
    ("Hierarchical forecasting",   "analytics/forecasting/hierarchical_forecast.py", "analysis"),
    ("Upcoding + red team",        "integrity/upcoding_detection.py",            "analysis"),
    ("Risk adjustment",            "analytics/risk/risk_adjustment.py",          "analysis"),
    ("Metric copilot (local LLM)", "ai/metric_copilot.py",                       "llm"),
    ("Validation & injected bugs", "qa/validation.py",                           "qa"),
]

CHECKSUM_GLOBS = ["data/**/*.csv", "data/**/*.json"]
DATA_STAGES = {"generate"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint():
    out = {}
    for pat in CHECKSUM_GLOBS:
        for p in sorted(ROOT.glob(pat)):
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)
    return out


def run(label, script):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.run([PY, str(ROOT / script)], cwd=ROOT).returncode
    secs = round(time.perf_counter() - t0, 1)
    print(f"[{'ok' if rc == 0 else 'FAILED'}] {label} in {secs}s", flush=True)
    return {"label": label, "script": script, "ok": rc == 0, "seconds": secs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    if args.verify:
        base = REPORTS / "reproducibility.json"
        if not base.exists():
            sys.exit("no baseline - run the full pipeline first")
        before = json.loads(base.read_text(encoding="utf-8"))["fingerprint"]
        for label, script, stage in STAGES:
            if stage in DATA_STAGES and not run(label, script)["ok"]:
                sys.exit(f"{label} failed")
        after = fingerprint()
        changed = sorted(k for k in before if before.get(k) != after.get(k))
        missing = sorted(set(before) - set(after))
        print(f"\n{'=' * 72}\nREPRODUCIBILITY\n{'=' * 72}")
        print(f"  files compared : {len(before)}")
        print(f"  identical      : {len(before) - len(changed)}")
        print(f"  CHANGED        : {len(changed)}")
        for k in changed:
            print(f"     {k}\n       was {before[k][:16]}  now {after.get(k, '(gone)')[:16]}")
        ok = not changed and not missing
        print(f"\n  reproducible across processes: {ok}")
        if not ok:
            print("  a fixed seed did NOT make this deterministic - look for set "
                  "iteration order, builtin hash(), or dict ordering")
        sys.exit(0 if ok else 1)

    results = []
    for label, script, stage in STAGES:
        if stage == "llm" and args.skip_llm:
            print(f"[skipped] {label}")
            continue
        if stage == "qa" and args.skip_qa:
            print(f"[skipped] {label}")
            continue
        r = run(label, script)
        results.append(r)
        if not r["ok"] and stage in ("generate", "warehouse"):
            sys.exit(f"\n{label} failed and everything downstream depends on it")

    fp = fingerprint()
    REPORTS.mkdir(exist_ok=True)
    json.dump({"fingerprint": fp, "n_files": len(fp), "algorithm": "sha256",
               "note": ("re-run with --verify in a NEW process; same-process "
                        "reruns cannot catch hash-randomisation bugs"),
               "stages": results},
              open(REPORTS / "reproducibility.json", "w", encoding="utf-8"), indent=2)

    failed = [r for r in results if not r["ok"]]
    print(f"\n{'=' * 72}\nPIPELINE COMPLETE\n{'=' * 72}")
    for r in results:
        print(f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['label']:<30} {r['seconds']:>7.1f}s")
    print(f"\n  {len(results) - len(failed)}/{len(results)} stages succeeded in "
          f"{round(sum(r['seconds'] for r in results), 1)}s")
    print(f"  {len(fp)} data files fingerprinted")
    print("\n  Next:")
    print("    python run_all.py --verify      confirm reproducibility")
    print("    python -m pytest tests/ -q      run the test suite")
    print("    python app/api/main.py          serve the console on :8600")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
