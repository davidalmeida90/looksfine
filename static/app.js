/* LooksFine front end. Vanilla JS, SVG charts drawn here, data from the local server (/api/*). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const main = $("#main");
const tip = $("#tip");
let timers = [];
let META = null;           // experiments + runs, for labels

// ------------------------------------------------------------------ helpers
async function api(p) {
  const r = await fetch(p, { cache: "no-store" });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const MINUS = "−";
const sgn = (v, d = 2) => (v == null || !isFinite(v) ? "n/a" : (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(d));
const f = (v, d = 4) => (v == null || !isFinite(v) ? "n/a" : (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d));
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "n/a" : (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d) + "%");
const VARIANT_LABEL = { "qlike loss": "QLIKE loss", "mix loss": "MSE + QLIKE loss", "pool day": "Day pooling", "pool both": "Last bar + day pooling", "L512": "512-bar context", "Kronos-base": "Kronos-base" };
const prettyVariant = (v) => (v || "").split(", ").map((p) => VARIANT_LABEL[p] || p).join(" + ");
const gain = (model, ref) => (model == null || ref == null ? null : ((ref - model) / ref) * 100); // % lower loss than ref
const int = (v) => (v == null ? "n/a" : Number(v).toLocaleString("en-US"));
function dur(sec) {
  if (sec == null || !isFinite(sec)) return "n/a";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}
function ago(ts) {
  const s = Date.now() / 1000 - ts;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }) + " " + d.toTimeString().slice(0, 5);
}
function agoS(ts) {
  const s = Date.now() / 1000 - ts;
  if (s < 90) return "now";
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400) return `${Math.round(s / 3600)}h`;
  return new Date(ts * 1000).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}
const sigBadge = (t) => (t != null && Math.abs(t) >= 1.96 ? `<span class="sig" title="|t| at least 1.96: significant at 5%">sig</span>` : "");
const ARM_SHORT = { kronos: "Kronos fine-tune", randinit: "Random init", control: "Control (head only)", frozen: "Kronos frozen", lora: "Kronos LoRA", baseline: "Baseline" };
const armVar = (arm) => ({ kronos: "--s-kronos", randinit: "--s-randinit", control: "--s-control", baseline: "--s-base" }[arm] || "--s-other");
const swatch = (arm, dashed) => (dashed || arm === "baseline" ? `<span class="sw base"></span>` : `<span class="sw" style="background:var(${armVar(arm)})"></span>`);
const ICON = {
  completed: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4.8 8.2l2.1 2.1 4.3-4.6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  stopped: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
};
function pill(st) {
  const word = { completed: "Completed", running: "Running", starting: "Starting", stopped: "Stopped" }[st] || st;
  const ic = st === "running" || st === "starting" ? `<span class="dot"></span>` : ICON[st] || "";
  return `<span class="pill ${st}">${ic}${word}</span>`;
}
function every(ms, fn) { const id = setInterval(fn, ms); timers.push(id); return id; }
function showErr(e) { main.innerHTML = `<div class="err">Could not load: ${esc(e.message)}</div>`; }

// ------------------------------------------------------------------ tooltip
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 16, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 16;
  if (y + r.height > innerHeight - 8) y = innerHeight - r.height - 8;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => (tip.hidden = true);

// ------------------------------------------------------------------ charts
function nice(lo, hi, n = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1, [0, 1]];
  if (hi - lo < 1e-12) { hi += 0.5; lo -= 0.5; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || 10 * mag;
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toFixed(10));
  return [a, b, ticks];
}

/* Line chart. o.series: [{label, arm|color, dash, pts:[[x,y]], width, opacity, dots, nolabel}], o.cats: x labels when x is an index.
   o.yfmt, o.xfmt, o.height, o.hlines [{y,label}], o.vlines [{x,label}], o.diag, o.ylabel */
function lineChart(el, o) {
  el.classList.add("chart");
  const draw = () => {
    const W = Math.max(el.clientWidth, 280), H = o.height || 260;
    const labeled = o.series.filter((s) => !s.nolabel).length <= 4 && !o.nolabels;
    const m = { l: 58, r: labeled ? (o.rpad ?? 120) : 16, t: 14, b: 30 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const pts = o.series.flatMap((s) => s.pts);
    if (!pts.length) { el.innerHTML = `<div class="empty">No data yet.</div>`; return; }
    let xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]).concat((o.hlines || []).map((h) => h.y));
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (o.cats) { x0 = 0; x1 = Math.max(o.cats.length - 1, 1); }
    if (o.diag) { const lo = Math.min(x0, ...ys), hi = Math.max(x1, ...ys); x0 = lo; x1 = hi; ys = ys.concat([lo, hi]); }
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const lo0 = y0, hi0 = y1;
    const pad = (y1 - y0) * 0.08 || 0.01; y0 -= pad; y1 += pad;
    if (lo0 >= 0 && y0 < 0) y0 = 0;                       // losses and shares never go below zero
    let [ya, yb, yt] = nice(y0, y1, o.yticks || 5);
    if (o.ymax != null && hi0 <= o.ymax && yb > o.ymax) { yb = o.ymax; yt = yt.filter((t) => t <= o.ymax); }
    if (lo0 >= 0 && ya < 0) { ya = 0; yt = yt.filter((t) => t >= 0); }
    const X = (v) => m.l + (x1 === x0 ? iw / 2 : ((v - x0) / (x1 - x0)) * iw);
    const Y = (v) => m.t + ih - ((v - ya) / (yb - ya)) * ih;
    const yfmt = o.yfmt || ((v) => f(v, 3)), xfmt = o.xfmt || ((v) => (o.cats ? o.cats[v] : String(v)));
    let g = "";
    for (const t of yt) g += `<line class="gl" x1="${m.l}" x2="${m.l + iw}" y1="${Y(t)}" y2="${Y(t)}"/><text class="ax" x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${esc(yfmt(t))}</text>`;
    // x ticks
    let xt;
    if (o.cats) {
      const n = o.cats.length, every_ = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / 70))));
      xt = o.cats.map((_, i) => i).filter((i) => i % every_ === 0 || (n <= 8));
    } else xt = nice(x0, x1, Math.max(2, Math.floor(iw / 90)))[2].filter((v) => v >= x0 - 1e-9 && v <= x1 + 1e-9);
    for (const t of xt) g += `<text class="ax" x="${X(t)}" y="${H - 8}" text-anchor="middle">${esc(xfmt(t))}</text>`;
    if (o.ylabel) g += `<text class="ax" x="${m.l}" y="${m.t - 3}" text-anchor="start" style="font-size:11px">${esc(o.ylabel)}</text>`;
    if (o.diag) g += `<line x1="${X(x0)}" y1="${Y(x0)}" x2="${X(x1)}" y2="${Y(x1)}" style="stroke:var(--s-base);stroke-width:1.5;stroke-dasharray:5 4"/>`;
    for (const h of o.hlines || []) {
      g += `<line x1="${m.l}" x2="${m.l + iw}" y1="${Y(h.y)}" y2="${Y(h.y)}" style="stroke:var(--s-base);stroke-width:1.5;stroke-dasharray:5 4"/>`;
      if (h.label) g += `<text class="dl" x="${m.l + iw + 8}" y="${Y(h.y) + 4}">${esc(h.label)}</text>`;
    }
    for (const v of o.vlines || []) {
      g += `<line x1="${X(v.x)}" x2="${X(v.x)}" y1="${m.t}" y2="${m.t + ih}" style="stroke:var(--text-2);stroke-width:1;stroke-dasharray:2 3"/>`;
      if (v.label) g += `<text class="ann" x="${X(v.x) + 5}" y="${m.t + 11}">${esc(v.label)}</text>`;
    }
    const colorOf = (s) => s.color || `var(${armVar(s.arm)})`;
    // series
    const ends = [];
    for (const s of o.series) {
      if (!s.pts.length) continue;
      const d = s.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      const dash = s.dash || s.arm === "baseline" ? "stroke-dasharray:6 4;" : "";
      if (!s.dotsOnly) g += `<path d="${d}" style="fill:none;stroke:${colorOf(s)};stroke-width:${s.width || 2};${dash}opacity:${s.opacity ?? 1};stroke-linejoin:round;stroke-linecap:round"/>`;
      if (s.dots || s.dotsOnly) for (const p of s.pts) g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" style="fill:${colorOf(s)};stroke:var(--surface);stroke-width:2"/>`;
      if (labeled && !s.nolabel) { const p = s.pts[s.pts.length - 1]; ends.push({ y: Y(p[1]), x: X(p[0]), label: s.label }); }
    }
    // direct labels, de-overlapped
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
    for (const e of ends) g += `<text class="dl" x="${e.x + 9}" y="${e.y + 4}">${esc(e.label)}</text>`;
    g += `<line class="xh" id="xh" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/><g id="hd"></g>`;
    g += `<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" id="ov"/>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(o.aria || "line chart")}">${g}</svg>`;
    // hover: nearest x, every series at that x
    const svg = el.querySelector("svg"), ov = el.querySelector("#ov"), xh = el.querySelector("#xh"), hd = el.querySelector("#hd");
    const ux = [...new Set(pts.map((p) => p[0]))].sort((a, b) => a - b);
    ov.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      let best = ux[0], bd = Infinity;
      for (const v of ux) { const dd = Math.abs(X(v) - px); if (dd < bd) { bd = dd; best = v; } }
      xh.setAttribute("x1", X(best)); xh.setAttribute("x2", X(best)); xh.setAttribute("visibility", "visible");
      let rows = "", dots = "";
      const tol = (x1 - x0) / Math.max(ux.length, 1) * 1.5;
      for (const s of o.series) {
        let q = null, qd = Infinity;
        for (const p of s.pts) { const dd = Math.abs(p[0] - best); if (dd < qd) { qd = dd; q = p; } }
        if (!q || (o.cats ? qd > 0 : qd > tol)) continue;
        rows += `<div class="r"><span>${s.arm === "baseline" || s.dash ? `<span class="sw base"></span>` : `<span class="sw" style="background:${colorOf(s)}"></span>`}${esc(s.label)}</span><b>${esc(yfmt(q[1]))}</b></div>`;
        dots += `<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="4" style="fill:${colorOf(s)};stroke:var(--surface);stroke-width:2"/>`;
      }
      hd.innerHTML = dots;
      showTip(`<div class="h">${esc(o.tipx ? o.tipx(best) : xfmt(best))}</div>${rows}`, ev);
    });
    ov.addEventListener("mouseleave", () => { xh.setAttribute("visibility", "hidden"); hd.innerHTML = ""; hideTip(); });
  };
  draw();
  const ro = new ResizeObserver(() => draw());
  ro.observe(el);
}

/* Dot plot for comparing losses that differ in the third decimal: axis does not start at zero, so dots, not bars. */
function dotPlot(el, rows, o = {}) {
  el.classList.add("chart");
  const draw = () => {
    const W = Math.max(el.clientWidth, 280), rh = 34, H = rows.length * rh + 34;
    const m = { l: Math.min(330, W * 0.42), r: 70, t: 6, b: 26 };
    const iw = W - m.l - m.r;
    const vals = rows.map((r) => r.value).filter((v) => v != null);
    const [a, b, ticks] = nice(Math.min(...vals), Math.max(...vals), 4);
    const X = (v) => m.l + ((v - a) / (b - a)) * iw;
    let g = "";
    for (const t of ticks) g += `<line class="gl" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}"/><text class="ax" x="${X(t)}" y="${H - 8}" text-anchor="middle">${esc(f(t, 3))}</text>`;
    rows.forEach((r, i) => {
      const y = m.t + i * rh + rh / 2;
      g += `<text class="dl" x="${m.l - 12}" y="${y + 4}" text-anchor="end" style="fill:var(--text)">${esc(r.label.length > 52 ? r.label.slice(0, 51) + "…" : r.label)}</text>`;
      g += `<line x1="${m.l}" x2="${X(r.value)}" y1="${y}" y2="${y}" style="stroke:var(--line);stroke-width:1"/>`;
      g += r.dashed
        ? `<circle cx="${X(r.value)}" cy="${y}" r="6" style="fill:var(--surface);stroke:var(--s-base);stroke-width:2.5;stroke-dasharray:3 2"/>`
        : `<circle cx="${X(r.value)}" cy="${y}" r="7" style="fill:${r.color};stroke:var(--surface);stroke-width:2"/>`;
      g += `<text class="ax" x="${X(r.value) + 13}" y="${y + 4}" style="fill:var(--text-2)">${esc(f(r.value, 4))}</text>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(o.aria || "dot plot")}">${g}</svg>`;
  };
  draw();
  new ResizeObserver(draw).observe(el);
}

/* Horizontal bars from zero (accuracy is a magnitude). rows: [{label, value 0..1, color, sub}] */
function hbars(el, rows, o = {}) {
  el.classList.add("chart");
  const draw = () => {
    const W = Math.max(el.clientWidth, 280), rh = 36, H = rows.length * rh + 26;
    const m = { l: Math.min(340, W * 0.42), r: 64, t: 4, b: 22 };
    const iw = W - m.l - m.r;
    const X = (v) => m.l + v * iw;
    let g = "";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="gl" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}"/><text class="ax" x="${X(t)}" y="${H - 6}" text-anchor="middle">${t * 100}%</text>`;
    rows.forEach((r, i) => {
      const y = m.t + i * rh, bh = 18, yy = y + (rh - bh) / 2, w = Math.max(X(r.value) - m.l, 5);
      g += `<text class="dl" x="${m.l - 12}" y="${yy + 13}" text-anchor="end" style="fill:var(--text)">${esc(r.label)}</text>`;
      g += `<path d="M${m.l},${yy} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z" style="fill:${r.color}"><title>${esc(r.label)}: ${(r.value * 100).toFixed(1)}%</title></path>`;
      g += `<text class="ax" x="${X(r.value) + 8}" y="${yy + 13}" style="fill:var(--text-2);font-weight:500">${(r.value * 100).toFixed(1)}%${r.sub ? " " + esc(r.sub) : ""}</text>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(o.aria || "bar chart")}">${g}</svg>`;
  };
  draw();
  new ResizeObserver(draw).observe(el);
}

// bias-corrected EMA (as TensorBoard smooths): follows the curve from the first point
function emaOf(pts, beta = 0.9) {
  const out = []; let e = 0;
  pts.forEach(([x, v], i) => { e = beta * e + (1 - beta) * v; out.push([x, e / (1 - Math.pow(beta, i + 1))]); });
  return out;
}

function spark(vals, w = 120, h = 28) {
  if (vals.length < 2) return "";
  const lo = Math.min(...vals), hi = Math.max(...vals), r = hi - lo || 1;
  const d = vals.map((v, i) => `${i ? "L" : "M"}${((i / (vals.length - 1)) * (w - 4) + 2).toFixed(1)},${(h - 3 - ((v - lo) / r) * (h - 6)).toFixed(1)}`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true"><path d="${d}" style="fill:none;stroke:var(--s-kronos);stroke-width:1.8"/></svg>`;
}

// ------------------------------------------------------------------ sidebar: machine
async function sys() {
  try {
    const s = await api("/api/system");
    const mtr = (u, t) => `<div class="meter"><i style="width:${Math.min(100, (u / t) * 100).toFixed(0)}%"></i></div>`;
    let h = "";
    if (s.gpu) h += `<div class="row"><div class="k"><span>GPU ${esc(s.gpu.name.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", ""))}</span><b>${s.gpu.util.toFixed(0)}%</b></div>${mtr(s.gpu.util, 100)}</div>
      <div class="row"><div class="k"><span>VRAM</span><b>${s.gpu.mem_used.toFixed(1)} / ${s.gpu.mem_total.toFixed(1)} GB</b></div>${mtr(s.gpu.mem_used, s.gpu.mem_total)}</div>`;
    if (s.ram) h += `<div class="row"><div class="k"><span>RAM</span><b>${s.ram.used.toFixed(1)} / ${s.ram.total.toFixed(1)} GB</b></div>${mtr(s.ram.used, s.ram.total)}</div>`;
    h += `<div class="row"><div class="k"><span>Disk ${esc(s.disk.drive.replace("\\", ""))} free</span><b>${s.disk.free.toFixed(1)} GB</b></div>${mtr(s.disk.total - s.disk.free, s.disk.total)}</div>`;
    $("#sys").innerHTML = h;
  } catch (e) { /* server busy: keep the last reading */ }
}
async function badge() {
  try {
    const d = await api("/api/runs");
    const n = d.runs.filter((r) => r.status === "running" || r.status === "starting").length;
    const b = $("#nav-running");
    b.hidden = !n; b.textContent = n ? `${n} live` : "";
  } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------------ labels
async function meta(force) {
  if (META && !force) return META;
  const [ex, runs] = await Promise.all([api("/api/experiments"), api("/api/runs")]);
  const label = {};
  for (const fam of ex.families) {
    for (const g of fam.groups) label[g.runs.join("+")] = { text: `${fam.label}: ${ARM_SHORT[g.arm] || g.arm}${g.variant ? ` (${g.variant})` : ""}, ${g.seeds} seed${g.seeds > 1 ? "s" : ""}`, arm: g.arm, fam: fam.family };
  }
  for (const r of runs.runs) if (!label[r.name]) label[r.name] = { text: r.name, arm: r.arm, fam: r.family };
  label.HAR = { text: "HAR-RV", arm: "baseline" };
  label.HARPLUS = { text: "HAR + extras (linear)", arm: "baseline" };
  label.GBM = { text: "LightGBM on HAR features", arm: "baseline" };
  META = { ex, runs, label };
  return META;
}

// ------------------------------------------------------------------ page: experiments
async function pageExp() {
  const [ex, rd] = await Promise.all([api("/api/experiments"), api("/api/runs")]);
  const fams = Object.fromEntries(ex.families.map((x) => [x.family, x]));
  const grp = (fam, arm) => fams[fam]?.groups.find((g) => g.arm === arm && !g.variant);
  const baseH = (fam, b = "HAR") => fams[fam]?.baselines.find((x) => x.group === b);
  // the final recipe (chosen on VAL, LOG.md 25 Nov) when its runs exist, else the default recipe
  const FINAL = "mix loss";
  const grpF = (fam, arm) => fams[fam]?.groups.find((g) => g.arm === arm && g.variant === FINAL) || grp(fam, arm);
  const k = grpF("u200", "kronos"), c = grpF("u200", "control"), kd = grp("u200", "kronos"), ri = grp("u200", "randinit"), har = baseH("u200");
  const done = rd.runs.filter((r) => r.status === "completed"), live = rd.runs.filter((r) => r.status === "running" || r.status === "starting");
  const gpuH = done.reduce((s, r) => s + (r.minutes || 0), 0) / 60;
  let title = "Kronos fine-tune experiments";
  if (k?.heldout && har?.heldout) title = `Fine-tuned Kronos forecasts next-day volatility ${pct(gain(k.heldout.qlike_cal, har.heldout.qlike_cal))} better than HAR-RV on stocks it never saw`;

  let h = `<div class="eyebrow">Kronos fine-tune research</div><h1>${esc(title)}</h1>
    <p class="sub">Every number below is scored on TEST (Jul 2024 to Nov 2026, after Kronos's pretraining cutoff) with one ruler:
    QLIKE on variance after removing each model's own level bias (calibrated QLIKE, lower is better). Seeds are averaged into one forecast.</p>`;
  h += `<div class="kpis">
    <div class="kpi"><div class="l">Held-out cal. QLIKE</div><div class="v">${f(k?.heldout?.qlike_cal)}</div><div class="d">Kronos fine-tune${k?.variant ? ` (${esc(prettyVariant(k.variant))})` : ""}, ${k?.seeds ?? 0} seeds, ${k?.heldout?.stocks ?? "n/a"} unseen stocks</div></div>
    <div class="kpi"><div class="l">vs HAR-RV</div><div class="v">${pct(gain(k?.heldout?.qlike_cal, har?.heldout?.qlike_cal))}</div><div class="d">lower loss, same held-out rows</div></div>
    <div class="kpi"><div class="l">vs control</div><div class="v">${pct(gain(k?.heldout?.qlike_cal, c?.heldout?.qlike_cal))}</div><div class="d">DM t ${sgn(k?.dm_vs_control_heldout?.qlike_cal)} ${sigBadge(k?.dm_vs_control_heldout?.qlike_cal)}</div></div>
    <div class="kpi"><div class="l">vs random weights</div><div class="v">${pct(gain(kd?.heldout?.qlike_cal, ri?.heldout?.qlike_cal))}</div><div class="d">at ${fams.u200?.train_stocks ?? "all"} training stocks, default recipe</div></div>
    <div class="kpi"><div class="l">Runs</div><div class="v">${done.length}</div><div class="d">${live.length ? `${live.length} running now` : "none running"}</div></div>
    <div class="kpi"><div class="l">GPU time</div><div class="v">${gpuH.toFixed(1)} h</div><div class="d">summed over finished runs</div></div>
  </div>`;

  // release checklist: what "genuinely good" means, computed, never typed in
  if (k && c && har) {
    const n20k = grp("u200n20", "kronos"), n20r = grp("u200n20", "randinit");
    const seedsBeat = k.per_seed.every((p) => p.qlike_cal < c.all.qlike_cal);
    const gH = gain(k.heldout?.qlike_cal, har.heldout?.qlike_cal), gC = gain(k.heldout?.qlike_cal, c.heldout?.qlike_cal);
    const tC = k.dm_vs_control_heldout?.qlike_cal, tH = k.dm_vs_har?.qlike_cal;
    const g20 = n20k && n20r ? gain(n20k.heldout?.qlike_cal, n20r.heldout?.qlike_cal) : null;
    const checks = [
      ["Beats HAR-RV on stocks it never saw", gH != null && gH >= 5 && tH <= -2, `${pct(gH)} lower loss, DM t ${sgn(tH)} (all stocks)`],
      ["Beats the same network without Kronos, on stocks it never saw", tC != null && tC <= -2, `${pct(gC)} lower loss, DM t ${sgn(tC)}`],
      ["Every seed beats the control", seedsBeat, k.per_seed.map((p) => f(p.qlike_cal)).join(", ") + ` vs ${f(c.all.qlike_cal)}`],
      ["Pretraining earns its keep with little data", g20 != null && g20 >= 1, g20 != null ? `at 20 stocks ${pct(g20)} better than random weights` : "not run"],
      ["Clear margin over the control (2% or more)", gC != null && gC >= 2, `${pct(gC)} on held-out stocks`],
    ];
    const rk = k.robust, rc = c.robust;
    if (rk) {
      const ys = Object.keys(rk.years);
      const yH = ys.filter((y) => rk.years[y].model < rk.years[y].HAR).length;
      checks.push(["Lower loss than HAR-RV in every TEST year (held-out stocks)", yH === ys.length, ys.map((y) => `${y} ${f(rk.years[y].model)} vs ${f(rk.years[y].HAR)}`).join(", ")]);
      if (rc) {
        const yC = ys.filter((y) => rc.years[y] && rk.years[y].model < rc.years[y].model).length;
        checks.push(["Lower loss than the control in every TEST year (held-out stocks)", yC === ys.length, ys.map((y) => `${y} ${f(rk.years[y].model)} vs ${f(rc.years[y]?.model)}`).join(", ")]);
      }
      checks.push(["Beats HAR-RV on at least 80% of held-out stocks, one by one", rk.stock_wins_har >= 0.8, `${Math.round(rk.stock_wins_har * rk.stocks)} of ${rk.stocks} stocks`]);
    }
    const pass = checks.filter((x) => x[1]).length;
    h += `<section class="panel"><div class="panel-h"><h2>Release check: ${pass} of ${checks.length} criteria met</h2><span class="dim">Kronos fine-tune${k.variant ? ` (${esc(prettyVariant(k.variant))})` : ""} against its own control, ${fams.u200?.train_stocks ?? "all"} training stocks, ${k.seeds} seeds</span></div>
      <p class="note">The bar for publishing the model. Computed from the runs above; a criterion turns green only on the numbers.</p>
      <div class="tw"><table><tbody>${checks.map(([t, ok, why]) => `<tr><td style="width:150px">${ok ? `<span class="pill completed">${ICON.completed}Met</span>` : `<span class="pill stopped">${ICON.stopped}Not yet</span>`}</td><td>${esc(t)}</td><td class="dim mono" style="font-size:12.5px">${esc(why)}</td></tr>`).join("")}</tbody></table></div></section>`;
  }
  // data efficiency
  const effFams = [["u200n20", 20], ["u200n60", 60], ["u200", 159]].filter(([fm]) => fams[fm]).map(([fm, n]) => [fm, fams[fm].train_stocks || n]);
  if (effFams.length >= 2) {
    const cats = effFams.map(([, n]) => `${n} stocks`);
    const series = ["kronos", "randinit", "control"].map((arm) => ({
      arm, label: ARM_SHORT[arm], dots: true,
      pts: effFams.map(([fm], i) => [i, grp(fm, arm)?.heldout?.qlike_cal]).filter((p) => p[1] != null),
    })).filter((s) => s.pts.length);
    const at = (fm, arm) => grp(fm, arm)?.heldout?.qlike_cal;
    const g0 = gain(at(effFams[0][0], "kronos"), at(effFams[0][0], "randinit")), gL = gain(at("u200", "kronos"), at("u200", "randinit"));
    const et = g0 != null && gL != null
      ? `Pretraining stands in for data: at ${effFams[0][1]} stocks Kronos beats random weights by ${pct(g0)}, at ${effFams[effFams.length - 1][1]} by ${pct(gL)}`
      : "How much data each model needs";
    h += `<section class="panel"><div class="panel-h"><h2>${esc(et)}</h2></div>
      <p class="note">Calibrated QLIKE on the same held-out stocks as the number of training stocks grows. Lower is better. Dashed line: HAR-RV on those stocks.</p>
      <div class="legend">${series.map((s) => `<span>${swatch(s.arm)}${esc(s.label)}</span>`).join("")}<span>${swatch("baseline")}HAR-RV</span></div>
      <div id="eff"></div></section>`;
    main.innerHTML = h;
    lineChart($("#eff"), { series, cats, height: 280, yfmt: (v) => f(v, 3), hlines: har?.heldout ? [{ y: har.heldout.qlike_cal, label: "HAR-RV" }] : [], aria: "data efficiency" });
    h = "";
  }
  // recipe search: which training choices raise the ceiling
  for (const fam of ex.families) {
    const vs = fam.groups.filter((g) => g.vs_default && g.arm === "kronos");   // recipes of the fine-tune; a control trained another way is not a recipe
    if (!vs.length) continue;
    const def = fam.groups.find((g) => g.group === vs[0].vs_default.default_group);
    const ctrlG = fam.groups.find((g) => g.group === fam.control_group);
    const harB = fam.baselines.find((b) => b.group === "HAR");
    const ranked = [...vs].sort((a, b) => gain(b.vs_default.model, b.vs_default.default) - gain(a.vs_default.model, a.vs_default.default));
    const best = ranked[0], gb = gain(best.vs_default.model, best.vs_default.default);
    const fin = vs.find((g) => g.variant === FINAL);    // the recipe chosen on VAL (LOG.md), not the best TEST number
    const title = fin
      ? `Chosen on the validation period: ${prettyVariant(fin.variant)}, ${pct(gain(fin.vs_default.model, fin.vs_default.default))} below the default recipe on held-out stocks (DM t ${sgn(fin.vs_default.t)})`
      : gb > 0 ? `${prettyVariant(best.variant)} lowers held-out loss most: ${pct(gb)} below the default recipe (DM t ${sgn(best.vs_default.t)})`
      : "No variant beats the default recipe yet";
    const s0 = (g) => g?.per_seed?.find((p) => p.seed === 0)?.heldout_cal;
    const drows = [
      ...vs.map((g) => ({ label: prettyVariant(g.variant), value: s0(g), color: `var(${armVar(g.arm)})` })),
      { label: "Default recipe (MSE loss, last bar)", value: s0(def), color: `var(${armVar("kronos")})` },
      { label: "Control: same head, no Kronos", value: s0(ctrlG), color: `var(${armVar("control")})` },
      { label: "HAR-RV", value: harB?.heldout?.qlike_cal, dashed: true },
    ].filter((r) => r.value != null).sort((a, b) => a.value - b.value);
    const tr = ranked.map((g) => {
      const v = g.vs_default, gg = gain(v.model, v.default);
      return `<tr class="click" data-href="#/compare?${[g.runs.join("+"), v.default_runs.join("+"), ...(ctrlG ? [ctrlG.runs.join("+")] : []), "HAR"].map((x) => "g=" + encodeURIComponent(x)).join("&")}&ref=${encodeURIComponent(v.default_runs.join("+"))}&subset=held-out">
        <td><span class="lbl">${esc(prettyVariant(g.variant))}</span></td><td class="n">${v.seeds.join(", ")}</td>
        <td class="n mono">${f(v.model)}</td><td class="n mono dim">${f(v.default)}</td>
        <td class="n mono">${pct(gg)}</td><td class="n mono">${sgn(v.t)} ${sigBadge(v.t)}</td>
        <td class="n mono">${f(v.r2, 3)} <span class="dim">vs ${f(v.default_r2, 3)}</span></td><td class="n mono">${sgn(v.bias, 3)}</td></tr>`;
    }).join("");
    if (h) { main.innerHTML = h; h = ""; }
    main.insertAdjacentHTML("beforeend", `<section class="panel"><div class="panel-h"><h2>${esc(title)}</h2><span class="dim">recipe search, ${esc(fam.label)}</span></div>
      <p class="note">Each variant changes one or two training choices of the Kronos fine-tune. Chart: seed 0 of each, calibrated QLIKE on the held-out stocks, lower is better. Table: the variant against the default recipe with the same seeds, on the same rows; DM t below −2 is a real gain. The recipe was picked by a rule fixed in advance on the validation period (Jan to Jun 2024), never on these TEST numbers.</p>
      <div class="legend"><span>${swatch("kronos")}Kronos fine-tune</span><span>${swatch("control")}Control</span><span>${swatch("baseline")}HAR-RV</span></div>
      <div id="recipe-${esc(fam.family)}"></div>
      <div class="tw"><table><thead><tr><th>Variant</th><th class="n">Seeds</th><th class="n">Held-out</th><th class="n">Default</th><th class="n">Lower loss</th><th class="n">DM t</th><th class="n">R²</th><th class="n">Bias</th></tr></thead>
      <tbody>${tr}</tbody></table></div></section>`);
    dotPlot(document.getElementById(`recipe-${fam.family}`), drows, { aria: "recipe search, calibrated qlike on held-out stocks" });
  }
  // one panel per family
  let fh = "";
  for (const fam of ex.families) {
    const hasH = fam.groups.some((g) => g.heldout);
    const ctrl = fam.groups.find((g) => g.group === fam.control_group);
    fh += `<section class="panel"><div class="panel-h"><h2>${esc(fam.label)}</h2><span class="dim">target ${esc(fam.target === "lrv1" ? "next-day log realised variance" : "next-week log variance")}</span></div>
      <p class="note">${esc(fam.detail)}</p><div class="tw"><table><thead><tr>
      <th>Model</th><th class="n">Seeds</th><th class="n">Cal. QLIKE</th><th class="n">t vs ctrl</th><th class="n">t vs HAR</th>
      ${hasH ? `<th class="n">Held-out</th><th class="n">Held-out t</th>` : ""}
      <th class="n">R²</th><th class="n">IC</th><th class="n">Bias</th><th>Seeds</th></tr></thead><tbody>`;
    const best = Math.min(...fam.groups.map((g) => g.all.qlike_cal));
    for (const g of fam.groups) {
      const link = ctrl && g.group !== ctrl.group
        ? `#/compare?${[g.runs.join("+"), ctrl.runs.join("+"), "HAR"].map((x) => "g=" + encodeURIComponent(x)).join("&")}&ref=${encodeURIComponent(ctrl.runs.join("+"))}${hasH ? "&subset=held-out" : ""}`
        : `#/job/${encodeURIComponent(g.runs[0])}`;
      const ps = g.per_seed.map((p) => p.qlike_cal);
      fh += `<tr class="click${g.all.qlike_cal === best ? " best" : ""}" data-href="${link}">
        <td><span class="lbl">${swatch(g.arm)}${esc(ARM_SHORT[g.arm] || g.arm)}${g.variant ? ` <span class="dim">(${esc(g.variant)})</span>` : ""}</span></td>
        <td class="n">${g.seeds}</td><td class="n mark">${f(g.all.qlike_cal)}</td>
        <td class="n">${g.dm_vs_control ? sgn(g.dm_vs_control.qlike_cal) + sigBadge(g.dm_vs_control.qlike_cal) : '<span class="dim">ref</span>'}</td>
        <td class="n">${sgn(g.dm_vs_har?.qlike_cal)}${sigBadge(g.dm_vs_har?.qlike_cal)}</td>
        ${hasH ? `<td class="n">${f(g.heldout?.qlike_cal)}</td><td class="n">${g.dm_vs_control_heldout ? sgn(g.dm_vs_control_heldout.qlike_cal) + sigBadge(g.dm_vs_control_heldout.qlike_cal) : '<span class="dim">ref</span>'}</td>` : ""}
        <td class="n">${f(g.all.r2, 3)}</td><td class="n">${f(g.all.ic, 3)}</td><td class="n">${sgn(g.all.bias, 3)}</td>
        <td><span class="strip mono" style="font-size:12px;white-space:nowrap" title="${esc(ps.map((v) => f(v)).join(", "))}">${ps.length > 1 ? `${f(Math.min(...ps), 3)} to ${f(Math.max(...ps), 3)}` : '<span class="dim">one</span>'}</span></td></tr>`;
    }
    for (const b of fam.baselines) {
      fh += `<tr class="base"><td><span class="lbl">${swatch("baseline")}${esc(b.arm_label)}</span></td><td class="n dim">n/a</td><td class="n">${f(b.all.qlike_cal)}</td><td class="n dim">n/a</td><td class="n dim">${b.group === "HAR" ? "ref" : "n/a"}</td>
        ${hasH ? `<td class="n">${f(b.heldout?.qlike_cal)}</td><td class="n dim">n/a</td>` : ""}<td class="n">${f(b.all.r2, 3)}</td><td class="n">${f(b.all.ic, 3)}</td><td class="n">${sgn(b.all.bias, 3)}</td><td></td></tr>`;
    }
    fh += `</tbody></table></div><p class="foot">t: Diebold-Mariano on the daily cross-sectional mean of calibrated QLIKE, Newey-West 5 lags, against the control (ctrl) or HAR-RV. Negative = lower loss. Held-out: the stocks never trained on. Click a row to compare it.</p></section>`;
  }
  if (h) main.innerHTML = h + fh; else main.insertAdjacentHTML("beforeend", fh);
  $$("tr[data-href]").forEach((tr) => tr.addEventListener("click", () => (location.hash = tr.dataset.href.slice(1))));
}

// ------------------------------------------------------------------ page: jobs
const JOBS = { fam: "", arm: "", st: "", q: "", sort: "updated", dir: -1, sel: new Set() };
async function pageJobs() {
  const [d, qd] = await Promise.all([api("/api/runs"), api("/api/queue").catch(() => ({ items: [] }))]);
  let QUEUE = qd.items;
  const draw = (d) => {
    const runs = d.runs;
    const live = runs.filter((r) => r.status === "running" || r.status === "starting");
    let h = `<div class="eyebrow">Jobs</div><h1>${live.length ? `${live.length} job${live.length > 1 ? "s" : ""} training now, ${runs.filter((r) => r.status === "completed").length} finished` : `${runs.length} fine-tune runs, none training now`}</h1>
      <p class="sub">Every run in <span class="mono">runs/</span>. Live jobs are read from their logs every 10 seconds; finished jobs are scored from their saved TEST forecasts.</p>`;
    if (live.length) {
      h += `<div class="live">${live.map((r) => {
        const p = r.progress;
        const fr = p ? p.step / p.steps : 0;
        return `<div class="lcard" data-run="${esc(r.name)}"><div class="t"><span class="nm">${esc(r.name)}</span>${pill(r.status)}</div>
          <div class="arm">${esc(r.arm_label)}</div>
          <div class="bar"><i style="width:${(fr * 100).toFixed(1)}%"></i></div>
          <div class="m"><span>${p ? `step ${int(p.step)} / ${int(p.steps)}` : "building windows"}</span><span>${p ? `${dur(p.secs)} in, ~${dur(p.eta)} left` : ""}</span></div>
          <div class="m"><span>${p ? `val QLIKE ${f(p.val_qlike)}` : ""}</span><span>${r.windows ? `${int(r.windows.train)} train windows` : ""}</span></div></div>`;
      }).join("")}</div>`;
    }
    const waiting = QUEUE.filter((x) => x.status === "queued");
    if (waiting.length) {
      h += `<section class="panel"><div class="panel-h"><h2>Up next on the GPU: ${waiting.length} run${waiting.length > 1 ? "s" : ""}, one at a time</h2><span class="dim">${esc(waiting[0].queue)}</span></div>
        <div class="tw"><table><thead><tr><th class="n">#</th><th>Run it will write</th><th>Command</th></tr></thead><tbody>
        ${waiting.map((x, i) => `<tr><td class="n">${i + 1}</td><td><span class="name">${esc(x.run)}</span></td><td class="mono dim" style="font-size:12px">${esc(x.cmd)}</td></tr>`).join("")}</tbody></table></div></section>`;
    }
    const famOpts = Object.entries(d.families).filter(([k]) => runs.some((r) => r.family === k)).map(([k, v]) => `<option value="${k}" ${JOBS.fam === k ? "selected" : ""}>${esc(v.label)}</option>`).join("");
    const armOpts = Object.entries(ARM_SHORT).filter(([k]) => k !== "baseline").map(([k, v]) => `<option value="${k}" ${JOBS.arm === k ? "selected" : ""}>${esc(v)}</option>`).join("");
    h += `<div class="filters">
      <select id="f-fam" aria-label="Experiment"><option value="">All experiments</option>${famOpts}</select>
      <select id="f-arm" aria-label="Model"><option value="">All models</option>${armOpts}</select>
      <select id="f-st" aria-label="Status"><option value="">Any status</option>${["completed", "running", "stopped"].map((s) => `<option ${JOBS.st === s ? "selected" : ""}>${s}</option>`).join("")}</select>
      <input type="search" id="f-q" placeholder="Search run names" value="${esc(JOBS.q)}" aria-label="Search">
      <span class="spacer"></span>
      <button class="btn" id="cmp" ${JOBS.sel.size ? "" : "disabled"}>Compare selected (${JOBS.sel.size})</button></div>
      <div class="panel" style="padding:4px 6px"><div class="tw"><table id="jt"></table></div></div>`;
    main.innerHTML = h;
    $$(".lcard").forEach((el) => el.addEventListener("click", () => (location.hash = `/job/${encodeURIComponent(el.dataset.run)}`)));
    const table = () => {
      let rows = runs.filter((r) => (!JOBS.fam || r.family === JOBS.fam) && (!JOBS.arm || r.arm === JOBS.arm) && (!JOBS.st || r.status === JOBS.st) && (!JOBS.q || r.name.includes(JOBS.q)));
      const key = {
        updated: (r) => r.updated, name: (r) => r.name, ql: (r) => r.test?.qlike_cal ?? 9, har: (r) => r.vs_har_pct ?? -99, r2: (r) => r.test?.r2 ?? -9,
        hold: (r) => r.heldout_test?.qlike_cal ?? 9, min: (r) => r.minutes ?? 0, seed: (r) => r.seed,
      }[JOBS.sort];
      rows.sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * JOBS.dir);
      const th = (k, t, n) => `<th class="sort${n ? " n" : ""}" data-k="${k}">${t}${JOBS.sort === k ? ` <span class="arr">${JOBS.dir > 0 ? "↑" : "↓"}</span>` : ""}</th>`;
      $("#jt").innerHTML = `<thead><tr><th></th><th>Status</th>${th("name", "Run")}<th>Model</th>${th("seed", "Seed", 1)}${th("min", "Train time", 1)}
        ${th("ql", "Cal. QLIKE", 1)}${th("har", "vs HAR", 1)}${th("r2", "R²", 1)}${th("hold", "Held-out", 1)}${th("updated", "Age", 1)}</tr></thead><tbody>${rows.map((r) => `
        <tr class="click" data-run="${esc(r.name)}"><td><input type="checkbox" class="ck" data-ck="${esc(r.name)}" ${JOBS.sel.has(r.name) ? "checked" : ""} ${r.status !== "completed" ? "disabled" : ""} aria-label="select ${esc(r.name)}"></td>
        <td>${pill(r.status)}</td><td><span class="name">${esc(r.name)}</span><div class="dim" style="font-size:12px">${esc(d.families[r.family]?.short || r.family)}</div></td>
        <td><span class="lbl">${swatch(r.arm)}${esc(ARM_SHORT[r.arm] || r.arm)}${r.variant ? ` <span class="dim">(${esc(r.variant)})</span>` : ""}</span></td>
        <td class="n">${r.seed}</td>
        <td class="n">${r.minutes != null ? dur(r.minutes * 60) : r.progress ? `${Math.round((r.progress.step / r.progress.steps) * 100)}%` : "n/a"}</td>
        <td class="n">${f(r.test?.qlike_cal)}</td><td class="n">${r.vs_har_pct != null ? pct(r.vs_har_pct) : "n/a"}</td><td class="n">${f(r.test?.r2, 3)}</td>
        <td class="n">${f(r.heldout_test?.qlike_cal)}</td><td class="n dim" title="${esc(new Date(r.updated * 1000).toLocaleString("en-GB"))}">${agoS(r.updated)}</td></tr>`).join("")}</tbody>`;
      $$("#jt th.sort").forEach((x) => x.addEventListener("click", () => { const k = x.dataset.k; JOBS.dir = JOBS.sort === k ? -JOBS.dir : k === "name" ? 1 : -1; JOBS.sort = k; table(); }));
      $$("#jt tr[data-run]").forEach((tr) => tr.addEventListener("click", (ev) => { if (ev.target.matches("input")) return; location.hash = `/job/${encodeURIComponent(tr.dataset.run)}`; }));
      $$("#jt input[data-ck]").forEach((c) => c.addEventListener("change", () => { c.checked ? JOBS.sel.add(c.dataset.ck) : JOBS.sel.delete(c.dataset.ck); $("#cmp").disabled = !JOBS.sel.size; $("#cmp").textContent = `Compare selected (${JOBS.sel.size})`; }));
    };
    table();
    $("#f-fam").onchange = (e) => { JOBS.fam = e.target.value; table(); };
    $("#f-arm").onchange = (e) => { JOBS.arm = e.target.value; table(); };
    $("#f-st").onchange = (e) => { JOBS.st = e.target.value; table(); };
    $("#f-q").oninput = (e) => { JOBS.q = e.target.value.trim(); table(); };
    $("#cmp").onclick = () => { location.hash = `/compare?${[...JOBS.sel, "HAR"].map((x) => "g=" + encodeURIComponent(x)).join("&")}`; };
  };
  draw(d);
  every(10000, async () => { if (document.activeElement?.id === "f-q") return; try { QUEUE = (await api("/api/queue")).items; draw(await api("/api/runs")); } catch (e) { /* keep */ } });
}

// ------------------------------------------------------------------ page: one job
const JOBTAB = { tab: "overview" };
async function pageJob(name) {
  const d = await api(`/api/run/${encodeURIComponent(name)}`);
  const s = d.summary, card = d.card, lg = d.log;
  const live = !card;
  const famLabel = (await meta()).ex.families.find((x) => x.family === s.family)?.label || s.family;
  let h = `<div class="crumb"><a href="#/jobs">Jobs</a> / ${esc(s.family)}</div>
    <div class="head-row"><div><h1 class="mono">${esc(name)}</h1></div>${pill(s.status)}</div>
    <div class="chips"><span class="chip">Base model <b>${esc(s.base_model)}</b></span><span class="chip">${swatch(s.arm)}<b>${esc(s.arm_label)}</b></span>
    <span class="chip">Data <b>${esc(famLabel)}</b></span><span class="chip">Target <b>${s.target === "lrv1" ? "next-day log RV" : s.target === "lrv5" ? "next-week log RV" : "next-week log variance"}</b></span>
    <span class="chip">Seed <b>${s.seed}</b></span>${s.holdout ? `<span class="chip">Held-out stocks <b>${s.holdout}</b></span>` : ""}${s.train_stocks ? `<span class="chip">Training stocks <b>${s.train_stocks}</b></span>` : ""}</div>`;
  if (live) {
    const p = s.progress;
    h += `<div class="kpis"><div class="kpi"><div class="l">Progress</div><div class="v">${p ? Math.round((p.step / p.steps) * 100) + "%" : "starting"}</div><div class="d">${p ? `step ${int(p.step)} of ${int(p.steps)}` : "building training windows"}</div></div>
      <div class="kpi"><div class="l">Elapsed</div><div class="v">${p ? dur(p.secs) : "n/a"}</div><div class="d">training loop only</div></div>
      <div class="kpi"><div class="l">Time left</div><div class="v">${p ? "~" + dur(p.eta) : "n/a"}</div><div class="d">then scoring TEST</div></div>
      <div class="kpi"><div class="l">Latest val QLIKE</div><div class="v">${p ? f(p.val_qlike) : "n/a"}</div><div class="d">raw QLIKE on VAL</div></div>
      <div class="kpi"><div class="l">Windows</div><div class="v">${s.windows ? int(s.windows.train) : "n/a"}</div><div class="d">${s.windows ? `${int(s.windows.val)} val, ${int(s.windows.test)} test` : ""}</div></div></div>`;
  } else {
    const ctl = d.control, harB = d.baselines?.HAR;
    h += `<div class="kpis">
      <div class="kpi"><div class="l">TEST cal. QLIKE</div><div class="v">${f(s.test?.qlike_cal)}</div><div class="d">raw ${f(s.test?.qlike)}, bias ${sgn(s.test?.bias, 3)}</div></div>
      <div class="kpi"><div class="l">vs HAR-RV</div><div class="v">${pct(s.vs_har_pct)}</div><div class="d">DM t ${sgn(harB?.dm?.qlike_cal)} ${sigBadge(harB?.dm?.qlike_cal)}</div></div>
      ${ctl ? `<div class="kpi"><div class="l">vs control</div><div class="v">${pct(gain(s.test?.qlike_cal, ctl.test.qlike_cal))}</div><div class="d">DM t ${sgn(ctl.dm?.qlike_cal)} ${sigBadge(ctl.dm?.qlike_cal)}, ${ctl.runs.length} seeds</div></div>` : ""}
      <div class="kpi"><div class="l">R² (log RV)</div><div class="v">${f(s.test?.r2, 3)}</div><div class="d">HAR-RV ${f(harB?.test?.r2, 3)}</div></div>
      <div class="kpi"><div class="l">Rank IC</div><div class="v">${f(s.test?.ic, 3)}</div><div class="d">cross-section, daily mean</div></div>
      ${s.heldout_test ? `<div class="kpi"><div class="l">Held-out cal.</div><div class="v">${f(s.heldout_test.qlike_cal)}</div><div class="d">HAR-RV ${f(s.heldout_har?.qlike_cal)}</div></div>` : ""}
      <div class="kpi"><div class="l">Training</div><div class="v">${s.minutes != null ? dur(s.minutes * 60) : "n/a"}</div><div class="d">best step ${int(card.best_step)}</div></div></div>`;
  }
  const tabs = live ? ["training", "logs"] : ["overview", "training", "checkpoints", "stocks", "configuration", "logs"];
  if (!tabs.includes(JOBTAB.tab)) JOBTAB.tab = tabs[0];
  h += `<div class="tabs" role="tablist">${tabs.map((t) => `<button role="tab" data-tab="${t}" class="${JOBTAB.tab === t ? "on" : ""}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}</div><div id="tab"></div>`;
  main.innerHTML = h;
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => { JOBTAB.tab = b.dataset.tab; $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b)); renderTab(); }));
  const hist = card?.history?.length ? card.history : d.live?.history?.length ? d.live.history
    : (lg?.steps || []).map((x) => ({ step: x.step, val_r2: x.val_r2, val_qlike: x.val_qlike, val_cs_ic: x.val_ic }));
  const totalSteps = d.live?.steps || lg?.steps?.[0]?.steps || (hist.length ? hist[hist.length - 1].step : null);
  const epochs = card?.args?.epochs || s.epochs;
  const renderTab = () => {
    const el = $("#tab");
    if (JOBTAB.tab === "overview") return overview(el, d, s);
    if (JOBTAB.tab === "training") return training(el, card || (d.live ? { train_loss: d.live.train_loss, args: d.live.args } : null), hist, s, live, d.live);
    if (JOBTAB.tab === "checkpoints") {
      const tl = card?.train_loss || [];
      const near = (st) => { let b = null, bd = Infinity; for (const [x, v] of tl) { const dd = Math.abs(x - st); if (dd < bd) { bd = dd; b = v; } } return b; };
      const bestQ = Math.min(...hist.map((x) => x.val_qlike));
      el.innerHTML = `<section class="panel"><div class="panel-h"><h2>${hist.length} validation checkpoints, best at step ${int(card?.best_step)}</h2></div>
        <p class="note">The run keeps the weights with the lowest VAL QLIKE (raw). TEST is scored once, with those weights.</p>
        <div class="tw"><table><thead><tr><th class="n">Step</th><th class="n">Epoch</th><th class="n">Val R²</th><th class="n">Val QLIKE</th><th class="n">Val IC</th>${tl.length ? '<th class="n">Train loss</th>' : ""}<th>Kept</th></tr></thead><tbody>
        ${hist.map((x) => `<tr class="${x.val_qlike === bestQ ? "best" : ""}"><td class="n">${int(x.step)}</td><td class="n">${totalSteps && epochs ? ((x.step / totalSteps) * epochs).toFixed(2) : "n/a"}</td>
          <td class="n">${f(x.val_r2, 4)}</td><td class="n mark">${f(x.val_qlike, 4)}</td><td class="n">${f(x.val_cs_ic, 4)}</td>${tl.length ? `<td class="n">${f(near(x.step), 4)}</td>` : ""}
          <td>${x.step === card?.best_step ? '<span class="pill completed">' + ICON.completed + "Best</span>" : ""}</td></tr>`).join("")}</tbody></table></div></section>`;
      return;
    }
    if (JOBTAB.tab === "stocks") return stocksTab(el, d, s, name);
    if (JOBTAB.tab === "configuration") {
      const a = card.args || {};
      const daily = name.startsWith("vol_");
      const info = {
        "Run": name, "Experiment": famLabel, "Base model": s.base_model, "Method": s.arm_label, "Training time": dur((card.minutes || 0) * 60),
        "Best step": int(card.best_step), "Trainable params": card.trainable ? int(card.trainable) : "n/a", "GPU peak": card.gpu_peak_gb ? card.gpu_peak_gb + " GB" : "n/a",
        "Train windows": lg ? int(lg.windows.train) : "n/a", "Val windows": lg ? int(lg.windows.val) : "n/a", "Test windows": lg ? int(lg.windows.test) : "n/a",
        "TRAIN": daily ? "2016-03 to 2021-12" : "2021-07 to 2023-12", "VAL": daily ? "2022-01 to 2023-12" : "2024-01 to 2024-06", "TEST": daily ? "2024-07 to 2026-11" : "2024-07 to 2026-11-10",
        "Input": daily ? `last ${a.L || 128} daily bars` : `last ${a.L || 256} five-minute bars + 9 HAR features`,
        "Optimiser": "AdamW, OneCycle, fp16 autocast", "Checkpoint rule": "lowest VAL QLIKE",
      };
      el.innerHTML = `<section class="panel"><div class="panel-h"><h2>Run</h2></div><div class="kv">${Object.entries(info).map(([k, v]) => `<div><span>${esc(k)}</span><span>${esc(v)}</span></div>`).join("")}</div></section>
        <section class="panel"><div class="panel-h"><h2>Arguments, as passed</h2></div><div class="kv">${Object.entries(a).map(([k, v]) => `<div><span>${esc(k)}</span><span>${esc(JSON.stringify(v))}</span></div>`).join("")}</div></section>`;
      return;
    }
    if (JOBTAB.tab === "logs") {
      el.innerHTML = lg ? `<section class="panel"><div class="panel-h"><h2>${esc(lg.log)}</h2><span class="dim">last write ${ago(lg.mtime)}, warnings hidden</span></div><pre class="log">${esc(lg.tail.join("\n"))}</pre></section>`
        : `<div class="empty">No log file names this run (it was filmed in a take, or run before logs were kept).</div>`;
    }
  };
  renderTab();
  if (live) every(10000, () => { if (location.hash.includes(encodeURIComponent(name)) || location.hash.includes(name)) route(); });
}

function overview(el, d, s) {
  const mo = d.monthly || {};
  const months = Object.keys(mo.model || {}).sort();
  const cats = months.map((m) => m.slice(2));
  const ser = [{ arm: s.arm, label: ARM_SHORT[s.arm] || "This run", pts: months.map((m, i) => [i, mo.model[m]]) }];
  if (mo.control) ser.push({ arm: "control", label: "Control", pts: months.map((m, i) => [i, mo.control[m]]).filter((p) => p[1] != null) });
  if (mo.HAR) ser.push({ arm: "baseline", label: "HAR-RV", pts: months.map((m, i) => [i, mo.HAR[m]]).filter((p) => p[1] != null) });
  const wins = months.filter((m) => mo.HAR && mo.model[m] < mo.HAR[m]).length;
  const cal = d.calibration || [];
  const gap = cal.length ? Math.max(...cal.map((c) => Math.abs(c.real - c.pred))) : null;
  const B = d.baselines || {};
  const rows = [{ k: "This run", arm: s.arm, t: s.test, dm: null }];
  if (d.control) rows.push({ k: `Control, ${d.control.runs.length} seeds averaged`, arm: "control", t: d.control.test, dm: d.control.dm });
  for (const b of Object.values(B)) rows.push({ k: b.label, arm: "baseline", t: b.test, dm: b.dm });
  const sub = d.subsets, by = d.by_year || {};
  el.innerHTML = `<div class="grid2">
    <section class="panel"><div class="panel-h"><h2>${mo.HAR ? `Lower loss than HAR-RV in ${wins} of ${months.length} months` : "Month by month"}</h2></div>
      <p class="note">Calibrated QLIKE per month on TEST. Lower is better.</p>
      <div class="legend">${ser.map((x) => `<span>${swatch(x.arm)}${esc(x.label)}</span>`).join("")}</div><div id="mchart"></div></section>
    <section class="panel"><div class="panel-h"><h2>${gap != null ? `Forecast bins sit within ${f(gap, 2)} log points of realised` : "Calibration"}</h2></div>
      <p class="note">15 equal-count bins of the forecast. Dashed diagonal: forecast equals realised (log variance).</p><div id="cchart"></div></section></div>
    <section class="panel"><div class="panel-h"><h2>Same TEST rows, every model</h2><span class="dim">${int(s.test?.n)} stock-days, ${s.test?.stocks} stocks</span></div>
      <div class="tw"><table><thead><tr><th>Model</th><th class="n">QLIKE</th><th class="n">Cal. QLIKE</th><th class="n">Bias</th><th class="n">R²</th><th class="n">IC</th><th class="n">This run minus row, t (cal.)</th><th class="n">t (sq. error)</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="${r.arm === "baseline" ? "base" : ""}"><td><span class="lbl">${swatch(r.arm)}${esc(r.k)}</span></td><td class="n">${f(r.t?.qlike)}</td><td class="n">${f(r.t?.qlike_cal)}</td>
        <td class="n">${sgn(r.t?.bias, 3)}</td><td class="n">${f(r.t?.r2, 4)}</td><td class="n">${f(r.t?.ic, 3)}</td><td class="n">${r.dm ? sgn(r.dm.qlike_cal) + sigBadge(r.dm.qlike_cal) : '<span class="dim">n/a</span>'}</td>
        <td class="n">${r.dm ? sgn(r.dm.se) : '<span class="dim">n/a</span>'}</td></tr>`).join("")}</tbody></table></div>
      <p class="foot">Negative t: this run has the lower loss. Bias: mean forecast minus realised, in log variance.</p></section>
    <div class="grid2">
      <section class="panel"><div class="panel-h"><h2>${sub ? "Seen stocks vs stocks never trained on" : "Stocks"}</h2></div>
      ${sub ? `<div class="tw"><table><thead><tr><th>Stocks</th><th class="n">Count</th><th class="n">Stock-days</th><th class="n">Cal. QLIKE</th><th class="n">R²</th><th class="n">IC</th></tr></thead><tbody>
        ${Object.entries(sub).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${v?.stocks ?? "n/a"}</td><td class="n">${int(v?.n)}</td><td class="n">${f(v?.qlike_cal)}</td><td class="n">${f(v?.r2, 4)}</td><td class="n">${f(v?.ic, 3)}</td></tr>`).join("")}</tbody></table></div>`
        : `<p class="note">This run trained and tested on the same stocks (no held-out set).</p>`}</section>
      <section class="panel"><div class="panel-h"><h2>By year</h2></div><div class="tw"><table><thead><tr><th>Year</th><th class="n">Stock-days</th><th class="n">QLIKE</th><th class="n">Cal. QLIKE</th><th class="n">R²</th><th class="n">IC</th></tr></thead><tbody>
        ${Object.entries(by).map(([y, v]) => `<tr><td class="mono">${y}</td><td class="n">${int(v.n)}</td><td class="n">${f(v.qlike)}</td><td class="n">${f(v.qlike_cal)}</td><td class="n">${f(v.r2, 4)}</td><td class="n">${f(v.ic, 3)}</td></tr>`).join("")}</tbody></table></div>
        <p class="foot">2026 runs to 10 Nov. Calibration shift is fitted on the whole TEST period.</p></section></div>`;
  lineChart($("#mchart"), { series: ser, cats, height: 260, yfmt: (v) => f(v, 3), tipx: (i) => months[i], aria: "monthly loss" });
  lineChart($("#cchart"), { series: [{ arm: s.arm, label: "bins", pts: cal.map((c) => [c.pred, c.real]), dotsOnly: true, nolabel: true }], diag: true, height: 260,
    yfmt: (v) => f(v, 1), xfmt: (v) => f(v, 1), tipx: (v) => `forecast ${f(v, 2)}`, rpad: 16, aria: "calibration" });
}

const STK = { sym: null, sort: "gain", dir: -1, only: "" };
function stocksTab(el, d, s, name) {
  const rows = d.stocks || [];
  if (!rows.length) { el.innerHTML = `<div class="empty">No per-stock scores.</div>`; return; }
  const better = rows.filter((r) => r.gain > 0).length;
  const hasH = rows.some((r) => r.heldout);
  if (!STK.sym || !rows.find((r) => r.symbol === STK.sym)) STK.sym = (rows.find((r) => r.heldout) || rows[0]).symbol;
  el.innerHTML = `<div class="grid2"><section class="panel"><div class="panel-h"><h2 id="stk-title"></h2></div>
      <p class="note">Realised log variance each TEST day (thin grey) against the forecast made the evening before. Pick a stock in the table.</p>
      <div class="legend"><span><span class="sw" style="background:var(--text-2);opacity:.5"></span>realised</span><span>${swatch(s.arm)}${esc(ARM_SHORT[s.arm] || "this run")}</span><span>${swatch("baseline")}HAR-RV</span></div>
      <div id="stk-chart"></div></section>
    <section class="panel"><div class="panel-h"><h2>Lower loss than HAR-RV on ${better} of ${rows.length} stocks</h2></div>
      ${hasH ? `<div class="filters" style="margin:6px 0 8px"><div class="seg" role="group">${[["", "All"], ["held", "Held-out only"]].map(([k, v]) => `<button data-only="${k}" class="${STK.only === k ? "on" : ""}">${v}</button>`).join("")}</div></div>` : ""}
      <div class="tw" style="max-height:470px;overflow:auto"><table id="stk-t"></table></div></section></div>`;
  const table = () => {
    let r = rows.filter((x) => !STK.only || x.heldout);
    r = [...r].sort((a, b) => (a[STK.sort] > b[STK.sort] ? 1 : -1) * STK.dir);
    const th = (k, t) => `<th class="sort n" data-k="${k}">${t}${STK.sort === k ? (STK.dir > 0 ? " \u2191" : " \u2193") : ""}</th>`;
    $("#stk-t").innerHTML = `<thead><tr><th class="sort" data-k="symbol">Stock</th>${th("n", "Days")}${th("qlike_cal", "Cal. QLIKE")}${th("har", "HAR-RV")}${th("gain", "Gain")}</tr></thead><tbody>
      ${r.map((x) => `<tr class="click${x.symbol === STK.sym ? " best" : ""}" data-sym="${esc(x.symbol)}"><td class="mono">${esc(x.symbol)}${x.heldout ? ' <span class="sig">held-out</span>' : ""}</td>
        <td class="n">${x.n}</td><td class="n mark">${f(x.qlike_cal)}</td><td class="n">${f(x.har)}</td><td class="n">${pct(x.gain)}</td></tr>`).join("")}</tbody>`;
    $$("#stk-t tr[data-sym]").forEach((tr) => tr.addEventListener("click", () => { STK.sym = tr.dataset.sym; table(); chart(); }));
    $$("#stk-t th.sort").forEach((x) => x.addEventListener("click", () => { const k = x.dataset.k; STK.dir = STK.sort === k ? -STK.dir : -1; STK.sort = k; table(); }));
  };
  const chart = async () => {
    const r = rows.find((x) => x.symbol === STK.sym);
    $("#stk-title").textContent = `${STK.sym}${r.heldout ? " (never trained on)" : ""}: ${r.gain >= 0 ? pct(r.gain) + " lower loss than HAR-RV" : pct(-r.gain) + " higher loss than HAR-RV"}`;
    const z = await api(`/api/stock?run=${encodeURIComponent(name)}&sym=${encodeURIComponent(STK.sym)}`);
    const idx = z.dates.map((_, i) => i);
    lineChart($("#stk-chart"), {
      series: [
        { color: "var(--text-2)", label: "realised", pts: idx.map((i) => [i, z.real[i]]), width: 1, opacity: 0.35, nolabel: true },
        { arm: "baseline", label: "HAR-RV", pts: idx.map((i) => [i, z.HAR[i]]), width: 1.6 },
        { arm: s.arm, label: ARM_SHORT[s.arm] || "model", pts: idx.map((i) => [i, z.model[i]]), width: 1.6 },
      ], cats: z.dates.map((x) => x.slice(2, 7)), height: 300, yfmt: (v) => f(v, 1), tipx: (i) => z.dates[i], rpad: 110, aria: "stock forecast",
    });
  };
  el.querySelectorAll("[data-only]").forEach((b) => b.addEventListener("click", () => { STK.only = b.dataset.only; el.querySelectorAll("[data-only]").forEach((x) => x.classList.toggle("on", x === b)); table(); }));
  table(); chart();
}

function training(el, card, hist, s, live, lv) {
  const tl = card?.train_loss || card?.history?.filter((x) => x.train_mse != null).map((x) => [x.step, x.train_mse]) || [];
  const best = card?.best_step;
  const loss = card?.args?.loss || "mse";
  const lossName = { mse: "squared error on log variance", qlike: "QLIKE on variance", mix: "squared error + QLIKE" }[loss];
  let h = "";
  if (tl.length) {
    const ema = emaOf(tl);
    const first = ema[0][1], last = ema[ema.length - 1][1];
    h += `<section class="panel"><div class="panel-h"><h2>${live ? `Live: step ${int(lv?.step)} of ${int(lv?.steps)}, training loss ${f(last, 4)}` : `Training loss fell ${pct(((first - last) / first) * 100)} from first logged step to last`}</h2>${live ? pill("running") : ""}</div>
      <p class="note">Train loss (${esc(lossName)}) every ${tl.length > 1 ? tl[1][0] - tl[0][0] : 25} steps, raw and smoothed.</p>
      <div class="legend"><span>${swatch("baseline")}raw</span><span>${swatch(s.arm)}smoothed (EMA 0.9, bias-corrected)</span></div><div id="tl"></div></section>`;
  } else if (!live) {
    h += `<section class="panel"><p class="note">Per-step train loss is saved from 25 Nov 2026 runs on. This run shows its validation curve only.</p></section>`;
  }
  const bh = hist.find((x) => x.step === best);
  h += `<div class="grid2"><section class="panel"><div class="panel-h"><h2>${bh ? `Validation QLIKE bottomed at ${f(bh.val_qlike)} (step ${int(best)})` : live ? "Validation QLIKE so far" : "Validation QLIKE"}</h2></div>
      <p class="note">Raw QLIKE on VAL (Jan to Jun 2024 intraday). The kept checkpoint is marked.</p><div id="vq"></div></section>
    <section class="panel"><div class="panel-h"><h2>Validation R² and rank IC</h2></div><p class="note">Log variance R² and cross-sectional rank IC on VAL.</p>
      <div class="legend"><span>${swatch(s.arm)}R²</span><span><span class="sw" style="background:var(--s-other)"></span>IC</span></div><div id="vr"></div></section></div>`;
  el.innerHTML = h;
  if (tl.length) {
    const ema = emaOf(tl);
    // one bad batch can be 10x the rest and flatten the curve: draw the raw line up to its 98th percentile (x 1.5), say so
    const srt = tl.map((p) => p[1]).sort((a, b) => a - b), cap = srt[Math.floor(srt.length * 0.98)] * 1.5;
    const nClip = tl.filter((p) => p[1] > cap).length;
    if (nClip) $("#tl").insertAdjacentHTML("beforebegin", `<p class="note" style="margin-top:-6px">${nClip} raw point${nClip > 1 ? "s" : ""} above ${f(cap, 3)} drawn at the top edge so the trend stays readable.</p>`);
    lineChart($("#tl"), { series: [{ arm: "baseline", label: "raw", pts: tl.map((p) => [p[0], Math.min(p[1], cap)]), width: 1, opacity: 0.6, nolabel: true }, { arm: s.arm, label: "smoothed", pts: ema }],
      height: 240, yfmt: (v) => f(v, 3), xfmt: (v) => int(v), tipx: (v) => `step ${int(v)}`, vlines: best ? [{ x: best, label: "kept" }] : [], rpad: 90, aria: "train loss" });
  }
  lineChart($("#vq"), { series: [{ arm: s.arm, label: "val QLIKE", pts: hist.map((x) => [x.step, x.val_qlike]), dots: true }], height: 240, yfmt: (v) => f(v, 3),
    xfmt: (v) => int(v), tipx: (v) => `step ${int(v)}`, vlines: best ? [{ x: best, label: "kept" }] : [], rpad: 90, aria: "validation qlike" });
  lineChart($("#vr"), { series: [{ arm: s.arm, label: "R²", pts: hist.map((x) => [x.step, x.val_r2]), dots: true }, { color: "var(--s-other)", label: "IC", pts: hist.map((x) => [x.step, x.val_cs_ic]), dots: true }],
    height: 240, yfmt: (v) => f(v, 3), xfmt: (v) => int(v), tipx: (v) => `step ${int(v)}`, rpad: 50, aria: "validation r2 and ic" });
}

// ------------------------------------------------------------------ page: compare
async function pageCompare(q) {
  const M = await meta();
  const SUBSETS = ["all", "seen", "held-out"];
  const groups = q.getAll("g"), subset = SUBSETS.includes(q.get("subset")) ? q.get("subset") : "all", ref = q.get("ref") || groups.find((g) => !["HAR", "HARPLUS", "GBM"].includes(g)) || "";
  const opts = [];
  for (const fam of M.ex.families) {
    for (const g of fam.groups) opts.push([g.runs.join("+"), M.label[g.runs.join("+")].text]);
  }
  opts.push(["HAR", "HAR-RV"], ["HARPLUS", "HAR + extras (linear)"]);
  const setQ = (gs, r, sb) => { location.hash = `/compare?${gs.map((x) => "g=" + encodeURIComponent(x)).join("&")}${r ? `&ref=${encodeURIComponent(r)}` : ""}&subset=${encodeURIComponent(SUBSETS.includes(sb) ? sb : "all")}`; };
  let h = `<div class="eyebrow">Compare</div>`;
  const famOf = (g) => M.label[g]?.fam || M.label[g.split("+")[0]]?.fam;
  const fset = new Set(groups.filter((g) => !["HAR", "HARPLUS", "GBM"].includes(g)).map(famOf));
  const common = fset.size === 1 ? [...fset][0] : null;
  const famLabel = common ? M.ex.families.find((x) => x.family === common)?.label : null;
  const lab = (g) => { const t = M.label[g]?.text || g; return famLabel && t.startsWith(famLabel + ": ") ? t.slice(famLabel.length + 2) : t; };
  const armOf = (g) => M.label[g]?.arm || M.label[g.split("+")[0]]?.arm || "other";
  h += `<div class="filters"><select id="add" aria-label="Add a model"><option value="">Add a model…</option>${opts.filter(([k]) => !groups.includes(k)).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join("")}</select>
    <div class="seg" role="group" aria-label="Stocks">${["all", "seen", "held-out"].map((x) => `<button data-sub="${x}" class="${subset === x ? "on" : ""}">${x === "all" ? "All stocks" : x === "seen" ? "Seen" : "Held-out"}</button>`).join("")}</div>
    <span class="spacer"></span>${groups.length ? `<button class="btn ghost" id="clear">Clear</button>` : ""}</div>
    <div class="chips">${groups.map((g) => `<span class="chip" data-chip="${esc(g)}">${swatch(armOf(g))}${esc(lab(g))}${g === ref ? " <b>(reference)</b>" : ` <a href="#" data-ref="${esc(g)}" class="dim" title="make reference">set ref</a>`} <a href="#" data-rm="${esc(g)}" aria-label="remove" class="dim">✕</a></span>`).join("")}</div>`;
  if (!groups.filter((g) => !["HAR", "HARPLUS", "GBM"].includes(g)).length) {
    main.innerHTML = h.replace(`<div class="eyebrow">Compare</div>`, `<div class="eyebrow">Compare</div><h1>Pick models to put side by side</h1><p class="sub">Seed groups are averaged into one forecast. Every model is scored on the rows all of them share.</p>`);
    wireCompare(groups, ref, subset, setQ);
    return;
  }
  main.innerHTML = h + `<div class="loading">Scoring…</div>`;
  let d;
  try {
    d = await api(`/api/compare?${groups.map((g) => "g=" + encodeURIComponent(g)).join("&")}&ref=${encodeURIComponent(ref)}&subset=${encodeURIComponent(subset)}`);
  } catch (e) { main.innerHTML = h + `<div class="err">${esc(e.message)}</div>`; wireCompare(groups, ref, subset, setQ); return; }
  const rows = d.rows;
  const models = rows.filter((r) => !["HAR", "HARPLUS", "GBM"].includes(r.group));
  const arms = models.map((r) => armOf(r.group));
  const slots = ["var(--s-kronos)", "var(--s-randinit)", "var(--s-control)", "var(--s-other)"];
  const distinct = new Set(arms).size === arms.length;
  // each arm keeps its colour (Kronos blue, control green, random init orange); a repeated arm takes the spare purple, then slots
  const seenArm = {}, spare = ["var(--s-other)"], pick = {};
  for (const r of models) {
    const arm = armOf(r.group);
    if (["HAR", "HARPLUS", "GBM"].includes(r.group)) continue;
    pick[r.group] = !seenArm[arm] ? `var(${armVar(arm)})` : (spare.shift() || slots[models.indexOf(r)] || "var(--s-base)");
    seenArm[arm] = true;
  }
  const colorOf = (g) => (["HAR", "HARPLUS", "GBM"].includes(g) ? "var(--s-base)" : distinct ? `var(${armVar(armOf(g))})` : pick[g] || "var(--s-base)");
  const best = [...rows].sort((a, b) => a.test.qlike_cal - b.test.qlike_cal)[0];
  const refRow = rows.find((r) => r.group === d.ref);
  const title = best.group === d.ref ? `${lab(best.group)} has the lowest loss of this set` : `${lab(best.group)}: ${pct(gain(best.test.qlike_cal, refRow.test.qlike_cal))} lower loss than the reference${best.dm && Math.abs(best.dm.qlike_cal) >= 1.96 ? ", significant" : ""}`;
  const sw = (g) => (["HAR", "HARPLUS", "GBM"].includes(g) ? `<span class="sw base"></span>` : `<span class="sw" style="background:${colorOf(g)}"></span>`);
  const months = Object.keys(rows[0].monthly).sort();
  const years = Object.keys(rows[0].by_year).sort();
  h = h.replace(`<div class="eyebrow">Compare</div>`, `<div class="eyebrow">Compare, ${subset === "all" ? "all stocks" : esc(subset) + " stocks"}</div><h1>${esc(title)}</h1>
    <p class="sub">${famLabel ? esc(famLabel) + ". " : ""}${int(d.n_common)} stock-days that every model below forecast. Calibrated QLIKE, lower is better.</p>`);
  h += `<section class="panel"><div id="dots"></div></section>
    <section class="panel"><div class="tw"><table><thead><tr><th>Model</th><th class="n">QLIKE</th><th class="n">Cal. QLIKE</th><th class="n">Bias</th><th class="n">R²</th><th class="n">IC</th>
      <th class="n">DM t vs ref (QLIKE)</th><th class="n">(cal.)</th><th class="n">(sq. error)</th>${years.map((y) => `<th class="n">${y} cal.</th>`).join("")}</tr></thead><tbody>
      ${rows.map((r) => `<tr class="${["HAR", "HARPLUS", "GBM"].includes(r.group) ? "base" : ""}"><td><span class="lbl">${sw(r.group)}${esc(lab(r.group))}</span></td>
        <td class="n">${f(r.test.qlike)}</td><td class="n">${f(r.test.qlike_cal)}</td><td class="n">${sgn(r.test.bias, 3)}</td><td class="n">${f(r.test.r2, 4)}</td><td class="n">${f(r.test.ic, 3)}</td>
        <td class="n">${r.dm ? sgn(r.dm.qlike) : '<span class="dim">ref</span>'}</td><td class="n">${r.dm ? sgn(r.dm.qlike_cal) + sigBadge(r.dm.qlike_cal) : ""}</td><td class="n">${r.dm ? sgn(r.dm.se) : ""}</td>
        ${years.map((y) => `<td class="n">${f(r.by_year[y])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <p class="foot">Raw QLIKE rewards forecasting high; the calibrated column removes each model's own level shift first. Negative t: lower loss than the reference.</p></section>
    <section class="panel"><div class="panel-h"><h2>Month by month</h2></div><div class="legend">${rows.slice(0, 5).map((r) => `<span>${sw(r.group)}${esc(lab(r.group))}</span>`).join("")}</div><div id="mc"></div></section>`;
  main.innerHTML = h;
  dotPlot($("#dots"), [...rows].sort((a, b) => a.test.qlike_cal - b.test.qlike_cal).map((r) => ({ label: lab(r.group), value: r.test.qlike_cal, color: colorOf(r.group), dashed: ["HAR", "HARPLUS", "GBM"].includes(r.group) })), { aria: "calibrated qlike by model" });
  lineChart($("#mc"), {
    series: rows.slice(0, 5).map((r) => ({ label: lab(r.group).split(",")[0].trim(), color: colorOf(r.group), dash: ["HAR", "HARPLUS", "GBM"].includes(r.group), pts: months.map((m, i) => [i, r.monthly[m]]) })),
    cats: months.map((m) => m.slice(2)), height: 280, yfmt: (v) => f(v, 3), tipx: (i) => months[i], rpad: 140, aria: "monthly comparison",
  });
  // chips were drawn before the colours were known: give each the colour its model has in the charts
  document.querySelectorAll(".chip[data-chip]").forEach((el) => { const sw = el.querySelector(".sw:not(.base)"); if (sw) sw.style.background = colorOf(el.dataset.chip); });
  wireCompare(groups, ref, subset, setQ);
}
function wireCompare(groups, ref, subset, setQ) {
  $("#add")?.addEventListener("change", (e) => { if (e.target.value) setQ([...groups, e.target.value], ref, subset); });
  $$("[data-sub]").forEach((b) => b.addEventListener("click", () => setQ(groups, ref, b.dataset.sub)));
  $$("[data-rm]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); const g = groups.filter((x) => x !== a.dataset.rm); setQ(g, ref === a.dataset.rm ? "" : ref, subset); }));
  $$("[data-ref]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); setQ(groups, a.dataset.ref, subset); }));
  $("#clear")?.addEventListener("click", () => setQ([], "", subset));
}

// ------------------------------------------------------------------ page: LLM fine-tunes (LoRA runs and their scores)
async function pageLLM() {
  const d = await api("/api/llm");
  if (!d.tasks.length) { main.innerHTML = `<div class="empty">No LLM fine-tunes found${d.error ? ": " + esc(d.error) : ""}.</div>`; return; }
  const t1 = d.tasks.find((t) => t.task === "t1") || d.tasks[0];
  const full = (t) => t.evals.filter((e) => !e.partial || e.all.n >= 100);
  const tun = (t) => full(t).find((e) => e.tuned);
  const bestBase = (t) => full(t).filter((e) => !e.tuned).sort((a, b) => b.all.statement_acc - a.all.statement_acc)[0];
  const T = tun(t1), B = bestBase(t1);
  const title = T && B ? `Fine-tuned, ${T.label.split(",")[0]} gets ${pct(T.all.statement_acc * 100)} of statements fully right; the best model out of the box gets ${pct(B.all.statement_acc * 100)}` : "LLM fine-tunes";
  let h = `<div class="eyebrow">LLM fine-tunes</div><h1>${esc(title)}</h1>
    <p class="sub">LoRA fine-tunes of small language models for pulling exact numbers out of financial filings.
    A statement counts only when every field is right. Scored on held-back documents, including companies never seen in training.</p>`;
  const mistakes = (e) => Object.entries(e.all.errors || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(", ") || "none";
  for (const t of d.tasks) {
    const tt = tun(t), bb = bestBase(t);
    h += `<section class="panel"><div class="panel-h"><h2>${esc(t.label)}${tt && bb ? `: fine-tuned ${pct(tt.all.statement_acc * 100)}, best untuned ${pct(bb.all.statement_acc * 100)}` : ""}</h2></div>
      <p class="note">${esc(t.detail)}</p><div class="legend"><span><span class="sw" style="background:var(--s-kronos)"></span>fine-tuned</span><span><span class="sw" style="background:var(--s-base)"></span>out of the box</span></div>
      <h3>Statements with every field right</h3><div id="bars-${t.task}"></div>
      <div class="tw" style="margin-top:10px"><table><thead><tr><th>Model</th><th class="n">Examples</th><th class="n">Fully right</th><th class="n">Within 1%</th><th class="n">Fields right</th><th class="n">Valid JSON</th><th class="n">Unseen companies</th><th class="n">Sec / example</th></tr></thead><tbody>
      ${t.evals.map((e) => `<tr class="${e.tuned ? "" : "base"}"><td><span class="lbl"><span class="sw" style="background:${e.tuned ? "var(--s-kronos)" : "var(--s-base)"}"></span>${esc(e.label)}${e.partial ? ' <span class="sig">partial</span>' : ""}</span><div class="dim" style="font-size:12px;margin-left:18px">mistakes: ${esc(mistakes(e))}</div></td>
        <td class="n">${e.all.n}</td><td class="n mark">${pct(e.all.statement_acc * 100)}</td><td class="n">${pct(e.all.within_1pct * 100)}</td><td class="n">${pct(e.all.field_acc * 100)}</td>
        <td class="n">${pct(e.all.valid_json * 100, 0)}</td><td class="n">${e.unseen ? pct(e.unseen.statement_acc * 100) : "n/a"}</td><td class="n">${e.run && e.run.sec_per_example ? e.run.sec_per_example.toFixed(2) : "n/a"}</td>
</tr>`).join("")}</tbody></table></div>
      <p class="foot">Partial: the run stopped early and is scored on the examples it finished. Scale: right digits, wrong units. Made up: a number that is not in the text.</p></section>`;
    for (const r of t.runs) {
      const c = r.card, vh = c.val_history || [];
      h += `<section class="panel"><div class="panel-h"><h2>Training ${esc(r.name)}: validation loss ${f(vh[0] && vh[0].loss, 4)} to ${f(vh.length ? vh[vh.length - 1].loss : null, 4)}</h2><span class="dim">${esc(c.started || "")}</span></div>
        <p class="note">${esc(c.note || "")}</p>
        <div class="kpis" style="margin:8px 0 14px"><div class="kpi"><div class="l">Examples</div><div class="v">${int(c.examples)}</div><div class="d">${int(c.tokens)} tokens</div></div>
          <div class="kpi"><div class="l">Updates</div><div class="v">${int(c.steps)}</div><div class="d">${c.epochs} epochs, accum ${c.accum}</div></div>
          <div class="kpi"><div class="l">Trainable</div><div class="v">${(c.params_trainable / 1e6).toFixed(1)}M</div><div class="d">${c.trainable_pct}% of ${(c.params_total / 1e6).toFixed(0)}M, LoRA r${c.lora_rank}</div></div>
          <div class="kpi"><div class="l">Train time</div><div class="v">${dur(c.train_minutes * 60)}</div><div class="d">${c.tokens_per_s} tokens/s</div></div>
          <div class="kpi"><div class="l">Peak GPU</div><div class="v">${c.gpu_peak_gb} GB</div><div class="d">${esc((c.safety && c.safety.gpu) || "")}</div></div>
          <div class="kpi"><div class="l">Adapter</div><div class="v">${c.adapter_mb} MB</div><div class="d">lr ${c.lr}, max ${int(c.max_len)} tokens</div></div></div>
        <div class="grid2"><div><h3>Training loss per update (raw and smoothed)</h3><div id="ll-${r.name}"></div></div>
          <div><h3>Validation loss</h3><div id="lv-${r.name}"></div><h3 style="margin-top:8px">Validation exact match</h3><div id="lx-${r.name}"></div></div></div>
        <div class="tw" style="margin-top:12px"><table><thead><tr><th class="n">Step</th><th class="n">Val loss</th><th class="n">Token accuracy</th><th class="n">Exact match</th></tr></thead><tbody>
          ${vh.map((v) => `<tr><td class="n">${v.step}</td><td class="n">${f(v.loss, 4)}</td><td class="n">${pct(v.token_acc * 100, 2)}</td><td class="n">${pct(v.exact * 100)}</td></tr>`).join("")}</tbody></table></div></section>`;
    }
  }
  main.innerHTML = h;
  for (const t of d.tasks) {
    hbars($(`#bars-${t.task}`), t.evals.map((e) => ({ label: e.label + (e.partial ? ` (partial, n=${e.all.n})` : ""), value: e.all.statement_acc, color: e.tuned ? "var(--s-kronos)" : "var(--s-base)" })), { aria: "statements fully right" });
    for (const r of t.runs) {
      const L = r.loss.filter((x) => x.loss != null);
      const ema = emaOf(L.map((x) => [x.step, x.loss]));
      lineChart($(`#ll-${r.name}`), { series: [{ arm: "baseline", label: "raw", pts: L.map((x) => [x.step, x.loss]), width: 1, opacity: 0.6, nolabel: true }, { arm: "kronos", label: "smoothed", pts: ema }],
        height: 260, yfmt: (v) => f(v, 3), xfmt: (v) => int(v), tipx: (v) => `update ${int(v)}`, rpad: 80, aria: "llm train loss" });
      const vh = r.card.val_history || [];
      lineChart($(`#lv-${r.name}`), { series: [{ arm: "kronos", label: "val loss", pts: vh.map((v) => [v.step, v.loss]), dots: true, nolabel: true }],
        height: 120, yfmt: (v) => f(v, 3), xfmt: (v) => int(v), tipx: (v) => `step ${v}`, rpad: 16, yticks: 3, aria: "val loss" });
      lineChart($(`#lx-${r.name}`), { series: [{ color: "var(--s-other)", label: "exact match", pts: vh.map((v) => [v.step, v.exact * 100]), dots: true, nolabel: true }],
        height: 120, ymax: 100, yfmt: (v) => v.toFixed(0) + "%", xfmt: (v) => int(v), tipx: (v) => `step ${v}`, rpad: 16, yticks: 3, aria: "exact match" });
    }
  }
}

// ------------------------------------------------------------------ page: takes
async function pageTakes() {
  const d = await api("/api/takes");
  main.innerHTML = `<div class="eyebrow">Takes</div><h1>${d.takes.length} filmed takes of real training runs</h1>
    <p class="sub">Screen recordings (.mp4) found in the takes folder.</p>
    <div class="takes">${d.takes.map((t) => `<div class="take"><video controls preload="metadata" src="/takes/${encodeURIComponent(t.file)}"></video>
      <div class="b"><div class="nm">${esc(t.file)}</div><div class="d">${t.seconds ? dur(t.seconds) : ""} · ${t.mb.toFixed(1)} MB · ${ago(t.made)}</div>
      ${t.events[0] ? `<div class="d mono" style="font-size:12px">${esc(t.events[0].event.replace(/^type: /, "$ "))}</div>` : ""}</div></div>`).join("")}</div>`;
}

// ------------------------------------------------------------------ final tests
const FT_ARM = { buy_hold: "Buy and hold", HAR: "HAR-RV sizing", HARPLUS: "HAR+extras sizing", control: "Control sizing (no Kronos)", kronos: "Kronos fine-tune sizing",
  YDAY: "Yesterday's RV", GBM: "LightGBM" };
const FT_SERIES = { buy_hold: { arm: "other" }, HAR: { arm: "baseline" }, control: { arm: "control" }, kronos: { arm: "kronos" } };
async function pageFinal() {
  const d = await api("/api/final");
  const t1 = d.t1, k = d.has_kronos;
  const bk = (arm) => t1?.results.find((r) => r.name === "BASKET" && r.arm === arm);
  let title = "Final tests: baselines measured, the fine-tune runs last";
  if (k && bk("kronos") && bk("HAR")) {
    const a = bk("kronos"), b = bk("HAR");
    title = `Sized by the fine-tune, the ${t1.names.length} names earn a Sharpe of ${f(a.sharpe, 2)} against ${f(b.sharpe, 2)} with HAR-RV sizing; risk steadiness ${f(a.risk_cv, 2)} against ${f(b.risk_cv, 2)}`;
  }
  let h = `<div class="eyebrow">Final tests</div><h1>${esc(title)}</h1>
    <p class="sub">Checks on the final model: a volatility-targeting backtest, accuracy against the simple rules, fresh sessions, an LLM agent test and a forward ledger. The rules are fixed before any result exists.</p>`;
  if (t1) {
    const arms = Object.keys(t1.series).filter((a) => FT_SERIES[a]);
    const cats = t1.dates;
    const mk = (key) => arms.map((a) => ({ arm: FT_SERIES[a].arm, label: FT_ARM[a], pts: t1.series[a][key].map((v, i) => [i, v]).filter((p) => p[1] != null) }));
    const rows = T1_ROWS.filter((a) => bk(a)).map((a) => { const r = bk(a); return `<tr><td><span class="lbl">${esc(FT_ARM[a] || a)}</span></td>
      <td class="n mono">${pct(r.return * 100)}</td><td class="n mono">${pct(r.vol * 100)}</td><td class="n mono">${f(r.sharpe, 2)}</td>
      <td class="n mono">${pct(r.max_dd * 100)}</td><td class="n mono">${f(r.risk_cv, 3)}</td><td class="n mono">${f(r.turnover, 1)}</td></tr>`; }).join("");
    h += `<section class="panel"><div class="panel-h"><h2>Test 1: ${t1.names.length} names, sized by each forecast</h2><span class="dim">${esc(t1.names.join(", "))}</span></div>
      <p class="note">Long only. Each day's weight is the square root of the forecast's own past median over today's forecast, capped at 2, held from the next open to the one after, 5 bp per unit of weight traded. Equal-weight basket. Only the ratio of a forecast to its own past matters, so forecasting high or low gains nothing.</p>
      <div class="legend">${arms.map((a) => `<span>${swatch(FT_SERIES[a].arm)}${esc(FT_ARM[a])}</span>`).join("")}</div>
      <h3 class="h3">Growth of 1</h3><div id="ft-eq"></div>
      <h3 class="h3">Risk over time: 21-day volatility of the basket (a flatter line is steadier risk)</h3><div id="ft-vol"></div>
      <div class="tw"><table><thead><tr><th>Arm</th><th class="n">Return</th><th class="n">Volatility</th><th class="n">Sharpe</th><th class="n">Max drawdown</th><th class="n">Risk steadiness</th><th class="n">Turnover</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="foot">Risk steadiness: std over mean of the 21-day volatility, lower is steadier. Turnover: weight traded per year.</p></section>`;
    main.innerHTML = h; h = "";
    lineChart($("#ft-eq"), { series: mk("equity"), cats, height: 280, yfmt: (v) => f(v, 2), tipx: (i) => cats[i], aria: "growth of 1" });
    lineChart($("#ft-vol"), { series: mk("vol21"), cats, height: 240, yfmt: (v) => pct(v * 100, 0), tipx: (i) => cats[i], aria: "21-day volatility" });
  }
  if (d.t34) {
    for (const key of ["heldout", "zeroshot", "seen", "study", "etf", "fresh"]) {
      const t = d.t34[key];
      if (!t) continue;
      const har = t.arms.find((r) => r.arm === "HAR");
      const pic = t.partial_ic || {};
      const tr = t.arms.map((r) => `<tr class="${r.arm === "kronos" ? "best" : ""}"><td><span class="lbl">${esc(FC_LBL[r.arm] || r.arm)}</span></td>
        <td class="n mono">${f(r.cal_QLIKE)}</td><td class="n mono">${pct(gain(r.cal_QLIKE, har?.cal_QLIKE))}</td><td class="n mono">${r.arm === "HAR" ? "" : `${sgn(r.t_vs_HAR)} ${sigBadge(r.t_vs_HAR)}`}</td>
        <td class="n mono">${f(r.R2, 3)}</td><td class="n mono">${r.rank_IC == null ? "" : f(r.rank_IC, 3)}</td>
        <td class="n mono">${pic[r.arm]?.mean == null ? "" : `${sgn(pic[r.arm].mean, 3)} <span class="dim">t ${sgn(pic[r.arm].t, 1)}</span>`}</td></tr>`).join("");
      h += `<section class="panel"><div class="panel-h"><h2>${esc(key === "fresh" ? "Test 4: " : key === "etf" || key === "zeroshot" ? "" : "Test 3: ")}${esc(t.label)}</h2>
        <span class="dim">${t.rows.toLocaleString()} rows, ${t.names} names, ${t.sessions} sessions, ${esc(t.start)} to ${esc(t.end)}</span></div>
        <div class="tw"><table><thead><tr><th>Forecast</th><th class="n">Cal. QLIKE</th><th class="n">Lower loss than HAR</th><th class="n">DM t vs HAR</th><th class="n">R²</th><th class="n">Rank IC</th><th class="n">Rank IC left after HAR</th></tr></thead><tbody>${tr}</tbody></table></div>
        ${key === "fresh" ? `<p class="foot">Few sessions: a check that nothing broke after the study window, not proof on its own.</p>` : ""}
        <p class="foot">DM t: Diebold-Mariano on the daily mean calibrated QLIKE against HAR-RV, Newey-West 5 lags; negative is a lower loss. Rank IC left after HAR: per session, the rank correlation of the forecast with realised RV once both are regressed on HAR-RV across stocks.</p></section>`;
    }
  }
  if (d.t6) {
    const sp = d.t6.splits, ls = sp.last_share;
    const top = ls?.buckets[ls.buckets.length - 1];
    const ttl = top ? `The edge comes from the end of the session: when the last hour carries the most of the day's variance, the fine-tune's loss is ${pct(-top.gap_pct)} below the control's` : "Where the edge comes from";
    h += `<section class="panel"><div class="panel-h"><h2>${esc(ttl)}</h2><span class="dim">${d.t6.rows.toLocaleString()} held-out rows, mean ${pct(-d.t6.gap_pct)} lower loss than the control</span></div>
      <p class="note">Calibrated QLIKE of the fine-tune against the same head without Kronos, split into fifths by what today looked like. A bar to the left is a lower loss for the fine-tune. The control already sees the last hour's variance as a number, so a gain there comes from the shape of the late candles.</p>
      <div class="grid2" style="grid-template-columns:1fr 1fr">${Object.entries(sp).map(([k, v]) => `<div><h3 class="h3">${esc(v.label)}</h3><div id="edge-${esc(k)}"></div></div>`).join("")}</div>
      <h3 class="h3">By month</h3><div id="edge-month"></div></section>`;
  }
  const agents = (d.t2s || []).map((t) => [t, t.model || "an LLM agent"]);   // same prompts, one file per model
  for (const [t, who] of agents) {
    const rows = t.arms.map((r) => `<tr><td><span class="lbl">${esc({ none: "No forecast", real: "Real fine-tune forecast", fake: "Fake forecast (another stock, another date)" }[r.arm])}</span></td>
      <td class="n mono">${f(r.mean_size, 1)}%</td><td class="n mono">${r.follow == null ? "" : sgn(r.follow, 3)}</td><td class="n mono">${f(r.sharpe, 2)}</td><td class="n mono">${pct(r.vol * 100)}</td><td class="n mono">${pct(r.worst_5pct * 100, 2)}</td></tr>`).join("");
    h += `<section class="panel"><div class="panel-h"><h2>Test 2: ${esc(who)} sizing a position, with the real forecast, a fake one, or none</h2><span class="dim">${t.decisions.toLocaleString()} decisions</span></div>
      <div class="tw"><table><thead><tr><th>Arm</th><th class="n">Mean size</th><th class="n">Follows the shown forecast</th><th class="n">Sharpe</th><th class="n">Volatility</th><th class="n">Worst 5% day</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="foot">Follows: rank correlation of the size chosen with the forecast shown. Real and fake read alike in the prompt.${t.planned && t.decisions < t.planned ? ` Partial: ${t.decisions.toLocaleString()} of ${t.planned.toLocaleString()} planned decisions.` : ""}${t.note ? ` ${esc(t.note)}.` : ""}</p></section>`;
  }
  if (!agents.length) h += `<section class="panel"><div class="panel-h"><h2>Test 2: the agent test</h2></div><p class="note">Runs on the final model's forecasts. Not run yet.</p></section>`;
  if (d.ledger) {
    h += `<section class="panel"><div class="panel-h"><h2>Test 5: forward ledger, ${d.ledger_n} forecasts logged after the close</h2><span class="dim">each line carries the hash of the one before</span></div>
      <div class="tw"><table><thead><tr><th>As of</th><th>Name</th><th class="n">Fine-tune, annualised</th><th class="n">HAR-RV</th><th class="n">Realised</th></tr></thead><tbody>${d.ledger.slice().reverse().map((r) => `<tr><td class="mono">${esc(r.as_of)}</td><td>${esc(r.symbol)}</td>
      <td class="n mono">${pct(r.kronos_vol * 100)}</td><td class="n mono">${pct(r.har_vol * 100)}</td><td class="n mono">${r.realised_vol == null ? '<span class="dim">pending</span>' : pct(r.realised_vol * 100)}</td></tr>`).join("")}</tbody></table></div></section>`;
  } else h += `<section class="panel"><div class="panel-h"><h2>Test 5: forward ledger</h2></div><p class="note">Starts with the released model. Not started yet.</p></section>`;
  main.insertAdjacentHTML("beforeend", h);
  if (d.t6) {
    const qn = ["lowest fifth", "2nd", "3rd", "4th", "highest fifth"];
    for (const [k, v] of Object.entries(d.t6.splits)) divBars($(`#edge-${k}`), v.buckets.map((b) => ({ label: qn[b.q - 1], value: b.gap_pct, tip: `${b.n.toLocaleString()} rows, ${f(b.share, 0)}% of the total gain` })));
    divBars($("#edge-month"), d.t6.months.map((m) => ({ label: m.month, value: m.gap_pct, tip: `${m.n} rows` })), { compact: true });
  }
}
/* Signed bars around zero, horizontal. Negative (a lower loss for the fine-tune) in the Kronos colour, positive in grey. */
function divBars(el, rows, o = {}) {
  el.classList.add("chart");
  const draw = () => {
    const W = Math.max(el.clientWidth, 260), rh = o.compact ? 17 : 30, H = rows.length * rh + 24;
    const m = { l: o.compact ? 64 : 104, r: 52, t: 4, b: 20 };
    const iw = W - m.l - m.r, mx = Math.max(1, ...rows.map((r) => Math.abs(r.value)));
    const X = (v) => m.l + iw / 2 + (v / mx) * (iw / 2);
    let g = `<line x1="${X(0)}" x2="${X(0)}" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--text-2);stroke-width:1"/>`;
    for (const t of [-mx, mx]) g += `<text class="ax" x="${X(t)}" y="${H - 5}" text-anchor="middle">${esc((t > 0 ? "+" : MINUS) + Math.abs(t).toFixed(0) + "%")}</text>`;
    rows.forEach((r, i) => {
      const y = m.t + i * rh, bh = Math.min(16, rh - 4), yy = y + (rh - bh) / 2, x0 = Math.min(X(0), X(r.value)), w = Math.max(Math.abs(X(r.value) - X(0)), 1.5);
      g += `<g><title>${esc(r.label + ": " + (r.value > 0 ? "+" : MINUS) + Math.abs(r.value).toFixed(1) + "%" + (r.tip ? ", " + r.tip : ""))}</title>
        <text class="ax" x="${m.l - 8}" y="${yy + bh / 2 + 4}" text-anchor="end" style="fill:var(--text-2)">${esc(r.label)}</text>
        <rect x="${x0}" y="${yy}" width="${w}" height="${bh}" rx="3" style="fill:${r.value < 0 ? "var(--s-kronos)" : "var(--s-base)"}"/>
        <text class="ax" x="${r.value < 0 ? X(0) + 6 : x0 + w + 5}" y="${yy + bh / 2 + 4}" text-anchor="start" style="fill:var(--text)">${esc((r.value > 0 ? "+" : MINUS) + Math.abs(r.value).toFixed(1) + "%")}</text></g>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(o.aria || "signed bars")}">${g}</svg>`;
  };
  draw();
  new ResizeObserver(draw).observe(el);
}
const T1_ROWS = ["buy_hold", "HAR", "HARPLUS", "control", "kronos"];
const FC_LBL = { zeroshot: "Kronos as shipped (zero-shot)", YDAY: "Yesterday's RV", HAR: "HAR-RV", HARPLUS: "HAR + extra features", GBM: "LightGBM", control: "Control: same head, no Kronos", kronos: "Kronos fine-tune" };

// ------------------------------------------------------------------ router
async function route() {
  timers.forEach(clearInterval); timers = []; hideTip();
  const h = location.hash || "#/";
  const [path, qs] = h.slice(1).split("?");
  const q = new URLSearchParams(qs || "");
  const navKey = path.startsWith("/job") ? "jobs" : path === "/" || path === "" ? "exp" : path.slice(1);
  $$(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === navKey));
  try {
    if (path === "/" || path === "") await pageExp();
    else if (path === "/jobs") await pageJobs();
    else if (path.startsWith("/job/")) { if (q.get("tab")) JOBTAB.tab = q.get("tab"); await pageJob(decodeURIComponent(path.slice(5))); }
    else if (path === "/compare") await pageCompare(q);
    else if (path === "/takes") await pageTakes();
    else if (path === "/final") await pageFinal();
    else if (path === "/llm") await pageLLM();
    else main.innerHTML = `<div class="empty">Nothing here.</div>`;
  } catch (e) { showErr(e); }
}

// theme
(function () {
  let t = "dark";
  try { t = localStorage.getItem("lf-theme") || "dark"; } catch (e) { /* storage blocked */ }
  document.documentElement.dataset.theme = t;
  $("#theme").addEventListener("click", () => {
    const n = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = n;
    try { localStorage.setItem("lf-theme", n); } catch (e) { /* ignore */ }
  });
})();
// synthetic demo folder (make_demo.py): say so on every page
api("/api/info").then((d) => { if (d.demo) $("#demo").hidden = false; }).catch(() => {});
window.addEventListener("hashchange", route);
route();
sys(); badge();
setInterval(sys, 5000);
setInterval(badge, 15000);
setInterval(() => { META = null; }, 60000);
