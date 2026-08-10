/* Aegis console. Every number is fetched through the semantic layer, so a
   figure on a screen and the same figure in a report are the same compiled
   definition rather than two implementations that happen to agree. */

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 0) => n === null || n === undefined || Number.isNaN(n) ? "—"
  : Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (n, d = 1) => n === null || n === undefined ? "—" : (n * 100).toFixed(d) + "%";
const money = (n) => n === null || n === undefined ? "—" : "$" + fmt(n, 0);
const PILL_OK = '<span class="pill ok">yes</span>';
const PILL_BAD = '<span class="pill critical">GAP</span>';
const PILL_CAUGHT = '<span class="pill ok">caught</span>';
const PILL_SURVIVED = '<span class="pill critical">survived</span>';

async function api(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(p + " -> " + r.status);
  return r.json();
}

function colors() {
  const cs = getComputedStyle(document.documentElement), g = (n) => cs.getPropertyValue(n).trim();
  return { dim: g("--dim"), grid: g("--border"),
    pal: [g("--accent"), g("--ok"), g("--warn"), g("--danger")] };
}

function chart(el, spec) {
  if (!el) return;
  const c = colors();
  const traces = spec.series.map((s, i) => ({
    x: spec.x, y: s.y, name: s.name,
    type: spec.type === "line" ? "scatter" : "bar",
    mode: spec.type === "line" ? "lines+markers" : undefined,
    marker: { color: c.pal[i % c.pal.length] },
    line: { color: c.pal[i % c.pal.length], width: 2 } }));
  Plotly.newPlot(el, traces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: c.dim, size: 11 }, margin: { l: 64, r: 14, t: 12, b: 64 },
    xaxis: { gridcolor: c.grid, title: spec.xaxis || "", tickangle: spec.tickangle || 0, automargin: true },
    yaxis: { gridcolor: c.grid, title: spec.yaxis || "", automargin: true },
    showlegend: spec.series.length > 1, legend: { orientation: "h", y: -0.3 },
  }, { displayModeBar: false, responsive: true });
  el._spec = spec;
}

function table(rows, cols) {
  if (!rows || !rows.length) return '<p class="hint">No rows.</p>';
  const h = cols.map((c) => '<th class="' + (c.num ? "num" : "") + '">' + esc(c.label) + "</th>").join("");
  const b = rows.map((r) => "<tr>" + cols.map((c) => {
    const v = c.get ? c.get(r) : r[c.key];
    return '<td class="' + (c.num ? "num" : "") + '">' + (c.raw ? v : esc(v)) + "</td>";
  }).join("") + "</tr>").join("");
  return '<div class="scroll"><table><thead><tr>' + h + "</tr></thead><tbody>" + b + "</tbody></table></div>";
}

function note(text) { return '<div class="note">' + esc(text) + "</div>"; }

const TITLES = { overview: "Overview", metrics: "Metrics explorer",
  members: "Member retention", planning: "Forecasting", integrity: "Integrity",
  risk: "Risk adjustment", validation: "Validation" };
const WS = { overview: rOverview, metrics: rMetrics, members: rMembers,
  planning: rPlanning, integrity: rIntegrity, risk: rRisk, validation: rValidation };
const state = { loaded: {} };

async function go(n) {
  document.querySelectorAll(".nav-item").forEach((x) => x.classList.toggle("active", x.dataset.ws === n));
  document.querySelectorAll(".ws").forEach((s) => s.classList.add("hidden"));
  $("#ws-" + n).classList.remove("hidden");
  location.hash = n;
  if (TITLES[n]) $("#page-title").textContent = TITLES[n];
  if (!state.loaded[n]) {
    try { await WS[n](); state.loaded[n] = true; }
    catch (e) { $("#ws-" + n).insertAdjacentHTML("afterbegin", note("Failed to load: " + e.message)); }
  }
}

function kpis(el, items) {
  $(el).innerHTML = items.map((it) =>
    '<div class="kpi ' + (it[2] || "") + '"><div class="v">' + esc(it[1]) +
    '</div><div class="k">' + esc(it[0]) + "</div></div>").join("");
}

async function rOverview() {
  const s = await api("/api/summary");
  kpis("#ov-kpis", [
    ["Loss ratio", s.loss_ratio ? s.loss_ratio.toFixed(4) : "—", s.loss_ratio > 0.9 ? "warn" : "ok"],
    ["PMPM", money(s.pmpm)],
    ["Member months", fmt(s.member_months)],
    ["Denial rate", pct(s.denial_rate)],
    ["Right-censored members", pct(s.censoring_rate), "warn"],
    ["Median tenure", s.median_survival_months + " mo"],
    ["Upcoding recall", pct(s.upcoding_recall, 0), "ok"],
    ["…once adversary adapts", pct(s.upcoding_recall_under_attack, 0), "danger"],
    ["Figures moved by 1 change", fmt(s.figures_moved_by_definition_change), "danger"],
    ["Injected defects caught", pct(s.bug_detection_rate, 0), "ok"],
  ]);
  $("#ov-note").innerHTML = note(
    "Aegis is a health insurance payer. Every metric on every screen compiles from a " +
    "single governed definition, and every analytical claim is scored against a mechanism " +
    "the generator injected — so 'correct' is a measurement, not an impression. The red " +
    "tiles are the honest ones: a fraud detector that looks 95% effective is 18% effective " +
    "once the adversary adapts, and one clarifying definition change silently restated " +
    fmt(s.figures_moved_by_definition_change) + " already-published historical figures.");
}

async function rMetrics() {
  const cat = await api("/api/catalog");
  $("#mx-metric").innerHTML = cat.metrics.map((m) =>
    '<option value="' + m.name + '">' + esc(m.label) + "</option>").join("");
  $("#mx-dim").innerHTML = '<option value="">(total)</option>' + cat.dimensions.map((d) =>
    '<option value="' + d.name + '">' + esc(d.name) + "</option>").join("");
  $("#mx-catalog").innerHTML = table(cat.metrics, [
    { key: "name", label: "Metric" }, { key: "definition", label: "Definition" },
    { key: "grain", label: "Grain" }, { key: "owner", label: "Owner" }]);
  const run = async () => {
    const m = $("#mx-metric").value, d = $("#mx-dim").value;
    const r = await api("/api/metric?name=" + m + "&by=" + d);
    $("#mx-sql").textContent = r.sql;
    chart($("#mx-chart"), {
      x: d ? r.rows.map((x) => x[d]) : [m], type: "bar", tickangle: -20,
      series: [{ name: m, y: r.rows.map((x) => x.value) }], yaxis: m });
  };
  $("#mx-btn").onclick = run;
  await run();
}

async function rMembers() {
  const s = await api("/api/survival");
  kpis("#sv-kpis", [
    ["Members", fmt(s.n_members)], ["Lapses observed", fmt(s.n_events)],
    ["Right-censored", pct(s.censoring_rate), "warn"],
    ["Median tenure", s.median_survival_months + " mo"],
    ["Log-rank p", String(s.logrank_individual_vs_employer.p_value)],
  ]);
  const by = s.kaplan_meier_by_channel, keys = Object.keys(by);
  chart($("#sv-chart"), { x: by[keys[0]].times, type: "line",
    series: keys.map((k) => ({ name: k, y: by[k].survival })),
    xaxis: "months enrolled", yaxis: "survival probability" });
  const rows = Object.keys(s.estimators).map((k) => ({ k, mae: s.estimators[k].mean_abs_error }));
  const ratio = (s.estimators.naive_logistic_churned_yes_no.mean_abs_error /
                 s.estimators.discrete_time_hazard.mean_abs_error).toFixed(0);
  $("#sv-table").innerHTML = table(rows, [
    { key: "k", label: "Estimator" },
    { label: "Mean abs error vs injected truth", num: true, get: (r) => r.mae.toFixed(4) }]) +
    note("The discrete-time hazard matches how the data actually arrive — monthly buckets — " +
      "and recovers the injected coefficients " + ratio + "x more accurately than treating " +
      "churn as a yes/no label. Cox sits in between. " + pct(s.censoring_rate) +
      " of members are still enrolled: censored observations, not negatives.");
}

async function rPlanning() {
  const f = await api("/api/forecast");
  chart($("#fc-chart"), { x: f.results.map((r) => r.approach), type: "bar", tickangle: -15,
    series: [{ name: "MAPE %", y: f.results.map((r) => r.mape_pct_overall) }], yaxis: "MAPE %" });
  $("#fc-note").innerHTML = table(f.results, [
    { key: "approach", label: "Approach" },
    { label: "MAPE %", num: true, get: (r) => r.mape_pct_overall.toFixed(3) },
    { label: "Coherent", get: (r) => (r.is_coherent ? "yes" : "NO") }]) + note(f.finding);
}

async function rIntegrity() {
  const u = await api("/api/fraud");
  const arms = [u.baseline].concat(u.red_team);
  chart($("#fr-chart"), { x: arms.map((a) => a.scenario), type: "bar", tickangle: -20,
    series: [{ name: "recall", y: arms.map((a) => a.recall) },
             { name: "precision", y: arms.map((a) => a.precision) }], yaxis: "score" });
  $("#fr-note").innerHTML = note(u.finding) + note(u.dead_rule.verdict);
  const g = await api("/api/governance");
  const b = g.definition_change_blast_radius;
  $("#gv-note").innerHTML = '<p class="hint">' + esc(b.change) + "</p>" +
    table(b.largest_moves.slice(0, 8), [
      { key: "metric", label: "Metric" }, { key: "slice", label: "Slice" },
      { label: "Before", num: true, get: (r) => r.before.toFixed(4) },
      { label: "After", num: true, get: (r) => r.after.toFixed(4) },
      { label: "Change", num: true, get: (r) => (r.relative_change * 100).toFixed(2) + "%" }]) +
    note(b.figures_moved + " of " + b.figures_compared + " historical figures moved, " +
      b.materially_moved + " materially. " + b.why_this_matters);
}

async function rRisk() {
  const r = await api("/api/risk");
  const best = r.models.reduce((a, b) => (b.r2 > a.r2 ? b : a));
  chart($("#rk-chart"), { x: best.calibration_by_decile.map((d) => "D" + d.decile), type: "line",
    series: [{ name: "predicted", y: best.calibration_by_decile.map((d) => d.mean_predicted) },
             { name: "actual", y: best.calibration_by_decile.map((d) => d.mean_actual) }],
    xaxis: "predicted-cost decile", yaxis: "annual cost ($)" });
  $("#rk-note").innerHTML = note(r.finding);
  $("#rk-table").innerHTML = table(r.models, [
    { key: "model", label: "Model" },
    { label: "R2", num: true, get: (m) => m.r2.toFixed(4) },
    { label: "Spearman", num: true, get: (m) => (m.spearman_rank_correlation === null ? "—" : m.spearman_rank_correlation) },
    { label: "Calib. slope", num: true, get: (m) => (m.calibration_slope === null ? "—" : m.calibration_slope) },
    { label: "Calibrated", get: (m) => (m.well_calibrated ? "yes" : "no") },
    { label: "Payment err %", num: true, get: (m) => m.aggregate_payment_error_pct.toFixed(2) }]) +
    note("The zero-effect proxy '" + r.fairness.null_proxy + "' costs " +
      r.fairness.r2_cost_of_removing_it + " R-squared to remove, which is the entire case for keeping it.");
}

async function rValidation() {
  const v = await api("/api/validation");
  const t = v.traceability;
  $("#vl-matrix").innerHTML = '<p class="hint">' + t.covered + "/" + t.requirements +
    " requirements covered (" + t.coverage_pct + "%). Gaps are listed, not removed from the register.</p>" +
    table(t.matrix, [
      { key: "requirement", label: "ID" }, { key: "statement", label: "Requirement" },
      { label: "Covered", raw: true, get: (r) => (r.covered ? PILL_OK : PILL_BAD) }]) +
    note("Audit trail: " + v.audit_trail.entries + " hash-chained entries; a settled record " +
      "was edited and the chain caught it at seq " + v.audit_trail.tamper_detected_at_seq + ".");
  const b = v.injected_bugs;
  $("#vl-bugs").innerHTML = b.error ? note(b.error) :
    '<p class="hint">' + b.caught + "/" + b.injected + " injected defects caught (" +
    (b.detection_rate * 100).toFixed(0) + "%).</p>" +
    table(b.detail.filter((d) => d.applied), [
      { key: "bug", label: "Defect" },
      { label: "Caught", raw: true, get: (r) => (r.caught ? PILL_CAUGHT : PILL_SURVIVED) },
      { key: "breaks", label: "What it breaks" }]);
  try {
    const c = await api("/api/copilot");
    const arms = [];
    if (c.rules_baseline) arms.push(Object.assign({ arm: "keyword rules" }, c.rules_baseline));
    if (c.llm && c.llm.n) arms.push(Object.assign({ arm: c.model }, c.llm));
    $("#vl-copilot").innerHTML = table(arms, [
      { key: "arm", label: "Arm" },
      { label: "Overall", num: true, get: (r) => pct(r.overall_accuracy, 1) },
      { label: "Resolution", num: true, get: (r) => pct(r.resolution_accuracy, 1) },
      { label: "Refusal", num: true, get: (r) => pct(r.refusal_accuracy, 1) }]) + note(c.finding);
  } catch (e) { $("#vl-copilot").innerHTML = note("Copilot report not built."); }
}

function initTheme() {
  const s = localStorage.getItem("aegis-theme") || "dark";
  document.documentElement.setAttribute("data-theme", s);
  $("#theme").textContent = s === "light" ? "🌙" : "☀️";
  $("#theme").onclick = () => {
    const n = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", n);
    localStorage.setItem("aegis-theme", n);
    $("#theme").textContent = n === "light" ? "🌙" : "☀️";
    document.querySelectorAll(".chart").forEach((el) => el._spec && chart(el, el._spec));
  };
}

(async function boot() {
  initTheme();
  document.querySelectorAll(".nav-item").forEach((n) =>
    n.addEventListener("click", () => go(n.dataset.ws)));
  // respond to fragment changes too, or browser back/forward silently
  // desyncs the view from the heading
  window.addEventListener("hashchange", () => {
    const w = (location.hash || "").replace("#", "");
    if (w in WS) go(w);
  });
  const r = (location.hash || "").replace("#", "");
  await go(r in WS ? r : "overview");
})();
