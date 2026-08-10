"""
main.py
----------
The Aegis console server.

Every figure served here comes through the semantic layer, so a number on
a screen and the same number in a report cannot disagree - they are the
same compiled definition. `/api/metric` is deliberately generic: it takes
a governed metric name and declared dimensions, and refuses anything else,
which is the same contract the AI copilot operates under.

Run:  python app/api/main.py   ->  http://127.0.0.1:8600
"""

import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
WEB = ROOT / "app" / "web"
DB = ROOT / "database" / "aegis.duckdb"

sys.path.insert(0, str(ROOT / "platform" / "semantic"))
from metrics import REGISTRY, compile_query, metric_catalog  # noqa: E402

app = FastAPI(title="Aegis Health Plan Intelligence", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.middleware("http")
async def no_store(request, call_next):
    r = await call_next(request)
    if not request.url.path.startswith("/api/"):
        r.headers["Cache-Control"] = "no-store, must-revalidate"
    return r


_cache = {}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def report(name):
    if name not in _cache:
        p = REPORTS / f"{name}.json"
        if not p.exists():
            raise HTTPException(404, f"report '{name}' not built - run run_all.py")
        _cache[name] = json.loads(p.read_text(encoding="utf-8"))
    return _cache[name]


def q(sql):
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


@app.get("/api/health")
def health():
    return {"status": "ok", "product": "Aegis",
            "reports": sorted(p.stem for p in REPORTS.glob("*.json"))}


@app.get("/api/catalog")
def catalog():
    """The governed vocabulary - what the console and the copilot may ask for."""
    return clean({"metrics": metric_catalog(),
                  "dimensions": [{"name": d.name, "description": d.description}
                                 for d in REGISTRY.dimensions.values()]})


@app.get("/api/metric")
def metric(name: str = Query(...), by: str = Query("")):
    """Compile and run a governed metric. Anything outside the vocabulary
    is a 422, not a best-effort guess."""
    dims = [d for d in by.split(",") if d.strip()] if by else []
    try:
        sql = compile_query(name, dims)
    except ValueError as e:
        raise HTTPException(422, str(e))
    rows = q(sql)
    out = []
    for r in rows:
        rec = {d: r[i] for i, d in enumerate(dims)}
        rec["value"] = r[len(dims)]
        out.append(rec)
    return clean({"metric": name, "dimensions": dims, "sql": sql, "rows": out})


@app.get("/api/summary")
def summary():
    lr = q(compile_query("loss_ratio", []))[0][0]
    pm = q(compile_query("pmpm", []))[0][0]
    mm = q(compile_query("member_months", []))[0][0]
    dr = q(compile_query("denial_rate", []))[0][0]
    out = {"loss_ratio": lr, "pmpm": pm, "member_months": mm, "denial_rate": dr}
    for key, rep, path in [
        ("censoring_rate", "survival", ("censoring_rate",)),
        ("median_survival_months", "survival", ("median_survival_months",)),
        ("upcoding_recall", "upcoding_detection", ("baseline", "recall")),
        ("upcoding_recall_under_attack", "upcoding_detection", ("worst_case", "recall")),
        ("forecast_best_mape", "hierarchical_forecast", ("results", 0, "mape_pct_overall")),
        ("bug_detection_rate", "validation", ("injected_bugs", "detection_rate")),
    ]:
        try:
            v = report(rep)
            for step in path:
                v = v[step]
            out[key] = v
        except Exception:
            out[key] = None
    try:
        b = report("metric_regression")["definition_change_blast_radius"]
        out["figures_moved_by_definition_change"] = b["materially_moved"]
    except Exception:
        out["figures_moved_by_definition_change"] = None
    try:
        r = report("risk_adjustment")["models"]
        best = max(r, key=lambda x: x["r2"])
        out["risk_r2"] = best["r2"]
        out["risk_calibration_slope"] = best["calibration_slope"]
    except Exception:
        pass
    return clean(out)


for slug, rep in [("survival", "survival"), ("forecast", "hierarchical_forecast"),
                  ("fraud", "upcoding_detection"), ("risk", "risk_adjustment"),
                  ("governance", "metric_regression"), ("validation", "validation"),
                  ("copilot", "metric_copilot")]:
    def _make(r=rep):
        def _h():
            return clean(report(r))
        return _h
    app.add_api_route(f"/api/{slug}", _make(), methods=["GET"])


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8600, log_level="warning")
