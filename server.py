"""LooksFine: a local dashboard for fine-tune runs (built for Kronos-Vol, the Kronos-small volatility fine-tune).

    python server.py --root PATH            then open http://127.0.0.1:8765
    python server.py --runs PATH/runs       the same: the root is the folder above runs/
    LOOKSFINE_ROOT=PATH python server.py    or set the root in the environment

Read only. What it reads under the root (each folder can be moved with its own flag or environment variable):
  runs/<run>/card.json, test_preds.parquet   finished runs (args, VAL history, train loss, TEST forecasts)
  runs/<run>/live.json                       a run in progress (step, train loss, VAL history so far)
  data/<panel>.parquet                       realised variance (the truth), to score forecasts and fit HAR baselines
  *.log, queue_*.sh in the root              optional: run logs (window counts, tail) and queued jobs
  final_tests/                               optional: outputs of the final-test scripts
  llm/                                       optional: LLM fine-tunes (evals/*/score.json, runs/*/run_card.json)
  rec/takes/*.mp4                            optional: screen recordings
Every score is computed here from the saved forecasts with one ruler:
  QLIKE on variance, calibrated QLIKE (each model's own best level shift removed), R2 on log variance,
  cross-sectional rank IC, and Diebold-Mariano t on the daily cross-sectional mean loss (Newey-West, 5 lags).
Binds to 127.0.0.1 only.
"""
import argparse
import ctypes
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
# set by configure() from the command line or the environment
ROOT = RUNS = DATA = TAKES = FT = FF = Path(".")


def configure(root=None, runs=None, data=None, final=None, llm=None, takes=None):
    """Folders the dashboard reads. Precedence: argument, environment variable, default under the root."""
    global ROOT, RUNS, DATA, TAKES, FT, FF
    env = os.environ.get
    runs = runs or env("LOOKSFINE_RUNS")
    root = root or env("LOOKSFINE_ROOT") or (Path(runs).resolve().parent if runs else Path.cwd())
    ROOT = Path(root).resolve()
    RUNS = Path(runs or ROOT / "runs").resolve()
    DATA = Path(data or env("LOOKSFINE_DATA") or ROOT / "data").resolve()
    FT = Path(final or env("LOOKSFINE_FINAL") or ROOT / "final_tests").resolve()
    FF = Path(llm or env("LOOKSFINE_LLM") or ROOT / "llm").resolve()
    TAKES = Path(takes or env("LOOKSFINE_TAKES") or ROOT / "rec" / "takes").resolve()
    if not RUNS.is_dir():
        raise SystemExit(f"no runs folder at {RUNS}: pass --root or --runs (or set LOOKSFINE_ROOT)")


F = ["lrv_d", "lrv_w", "lrv_m", "l_on", "l_last", "ret_d", "spy_lrv_d", "spy_lrv_w", "spy_lrv_m"]
TRAIN = ("2021-07-01", "2023-12-31")
TE = ("2024-07-01", "2026-11-10")
TE_DAILY = ("2024-07-01", "2026-11-09")                  # the weekly daily-bar experiments end one session earlier
LOCK = threading.RLock()
SAFE = re.compile(r"^[A-Za-z0-9_-]+$")                     # run names from the URL: no path separators
CACHE = {}

FAMILIES = {
    "daily": {"label": "Daily bars, S&P 500", "short": "Daily, S&P 500", "detail": "Next-week variance from 128 daily bars. Point-in-time S&P 500 members.", "order": 5},
    "intra60": {"label": "5-minute bars, 60 stocks", "short": "5-min, 60 stocks", "detail": "Next-day realised variance from the last 256 five-minute bars. 60 most liquid S&P 500 names.", "order": 4},
    "u200": {"label": "5-minute bars, 199 stocks, 40 held out", "short": "5-min, 159 + 40 held out", "detail": "159 training stocks. 40 random stocks never seen in training or validation.", "order": 1},
    "u200n60": {"label": "Data efficiency, 60 training stocks", "short": "5-min, 60 + 40 held out", "detail": "Same 40 held-out stocks. Training on 60 of the 159.", "order": 2},
    "u200n20": {"label": "Data efficiency, 20 training stocks", "short": "5-min, 20 + 40 held out", "detail": "Same 40 held-out stocks. Training on 20 of the 159 (a subset of the 60).", "order": 3},
}
ARMS = {
    "kronos": "Kronos-small, pretrained, full fine-tune",
    "randinit": "Same transformer, random weights",
    "control": "Head only, no Kronos (control)",
    "frozen": "Kronos frozen, head trained",
    "lora": "Kronos with LoRA adapters",
}


# ------------------------------------------------------------------ run metadata
def families():
    """FAMILIES, updated from ROOT/families.json when the root has one: {"tag": {"label", "short", "detail", "order"}}."""
    f = ROOT / "families.json"
    if not f.exists():
        return FAMILIES
    try:
        return {**FAMILIES, **json.loads(f.read_text(encoding="utf-8"))}
    except (ValueError, OSError):
        return FAMILIES


def describe(name, card):
    a = card.get("args", {})
    daily = name.startswith("vol_")
    mode = a.get("mode", "?")
    arm = "control" if mode == "none" else ("randinit" if a.get("random_init") else {"full": "kronos"}.get(mode, mode))
    if daily:
        fam = "daily"
    else:
        tag = a.get("tag", "")
        fam = {"": "intra60", "_u200": "u200"}.get(tag, tag.strip("_"))
    variant = []
    if a.get("loss", "mse") != "mse":
        variant.append(f"{a['loss']} loss")
    if a.get("pool", "last") != "last":
        variant.append(f"pool {a['pool']}")
    if a.get("model", "small") not in ("small", "NeoQuasar/Kronos-small"):
        variant.append("Kronos-base" if a["model"] in ("base", "NeoQuasar/Kronos-base") else str(a["model"]).replace("NeoQuasar/", ""))
    if a.get("L") and not daily and a["L"] != 256:
        variant.append(f"L{a['L']}")
    seed = a.get("seed", 0)
    group = re.sub(r"_s\d+$", "", name)
    panel = "daily" if daily else a.get("panel", "intra_daily.parquet")
    lr = a.get("lr_bb") if mode == "full" else (a.get("lr_lora") if mode == "lora" else None)
    return {"name": name, "family": fam, "arm": arm, "variant": ", ".join(variant), "seed": seed, "group": group,
            "panel": panel, "target": a.get("target", "lfv5"), "mode": mode, "random_init": bool(a.get("random_init")),
            "holdout": a.get("holdout", 0), "train_stocks": a.get("train_stocks", 0), "epochs": a.get("epochs"),
            "bs": a.get("bs"), "L": a.get("L"), "lr_backbone": lr, "lr_head": a.get("lr_head"),
            "base_model": "Random weights" if a.get("random_init") else ("None" if mode == "none" else
                          ("Kronos-base" if a.get("model") == "base" else "Kronos-small")),
            "arm_label": ARMS.get(arm, arm) + (f" ({', '.join(variant)})" if variant else "")}


# ------------------------------------------------------------------ truth and baselines
def truth(panel, target):
    key = ("truth", panel, target)
    if key in CACHE:
        return CACHE[key]
    if panel == "daily":
        W = pd.read_parquet(DATA / "weekly_rows.parquet")
        B = pd.read_parquet(DATA / "baseline_preds.parquet")
        T = W.merge(B, on=["date", "symbol"])
        T = T[(T.date >= TE_DAILY[0]) & (T.date <= TE_DAILY[1])]
        T = T[["date", "symbol", target, "f_har", "f_gbm"]].rename(columns={target: "y", "f_har": "HAR", "f_gbm": "GBM"})
        tr_n = None
    else:
        D = pd.read_parquet(DATA / panel).replace([np.inf, -np.inf], np.nan).dropna(subset=F + [target])
        tr = D[(D.date >= TRAIN[0]) & (D.date <= TRAIN[1])]
        for k, cols in {"HAR": ["lrv_d", "lrv_w", "lrv_m"], "HARPLUS": F}.items():
            D[k] = LinearRegression().fit(tr[cols], tr[target]).predict(D[cols])
        T = D[(D.date >= TE[0]) & (D.date <= TE[1])][["date", "symbol", target, "HAR", "HARPLUS"]].rename(columns={target: "y"})
        tr_n = int(tr.symbol.nunique())
    T = T.dropna(subset=["y"]).reset_index(drop=True)
    CACHE[key] = (T, tr_n)
    return CACHE[key]


def baselines_for(panel):
    return ["HAR", "GBM"] if panel == "daily" else ["HAR", "HARPLUS"]


BASE_LABEL = {"HAR": "HAR-RV", "HARPLUS": "HAR + extras (linear, same 9 features)", "GBM": "LightGBM on HAR features"}


# ------------------------------------------------------------------ scoring (one ruler)
def qlike_parts(y, p):
    yv, fv = np.exp(y), np.exp(p)
    r = yv / fv
    ql = r - np.log(r) - 1
    c = np.log(r.mean())
    rc = r / np.exp(c)
    qlc = rc - np.log(rc) - 1
    return ql, qlc, c


def stats(x):
    if len(x) == 0:
        return None
    ql, qlc, c = qlike_parts(x.y.values, x.p.values)
    se = (x.y - x.p) ** 2
    r2 = 1 - se.sum() / ((x.y - x.y.mean()) ** 2).sum()
    rk = x[["y", "p"]].groupby(x.date).rank()
    rk = rk - rk.groupby(x.date).transform("mean")
    num = (rk.y * rk.p).groupby(x.date).sum()
    den = np.sqrt((rk.y ** 2).groupby(x.date).sum() * (rk.p ** 2).groupby(x.date).sum())
    ic = (num / den.replace(0, np.nan)).mean()
    return {"n": int(len(x)), "stocks": int(x.symbol.nunique()), "days": int(x.date.nunique()), "qlike": float(ql.mean()),
            "qlike_cal": float(qlc.mean()), "level_shift": float(c), "bias": float((x.p - x.y).mean()), "r2": float(r2),
            "ic": float(ic), "mse": float(se.mean())}


def nw_se(d, lags=5):
    d = d - d.mean()
    n = len(d)
    g = [np.dot(d[k:], d[:n - k]) / n for k in range(lags + 1)]
    v = g[0] + 2 * sum((1 - k / (lags + 1)) * g[k] for k in range(1, lags + 1))
    return np.sqrt(v / n)


def dm(xa, xb):
    """t-stats of loss(a) - loss(b): QLIKE, calibrated QLIKE, squared error. Negative = a is better."""
    m = xa.merge(xb, on=["date", "symbol", "y"], suffixes=("_a", "_b"))
    if len(m) < 100:
        return None
    qa, qca, _ = qlike_parts(m.y.values, m.p_a.values)
    qb, qcb, _ = qlike_parts(m.y.values, m.p_b.values)
    out = {}
    for k, d in (("qlike", qa - qb), ("qlike_cal", qca - qcb), ("se", (m.y - m.p_a) ** 2 - (m.y - m.p_b) ** 2)):
        s = pd.Series(np.asarray(d), index=m.date).groupby(level=0).mean().values
        se_ = nw_se(s)
        out[k] = float(s.mean() / se_) if se_ > 0 else 0.0
    out["n"] = int(len(m))
    return out


def robustness(x):
    """On the held-out stocks when there are any: calibrated QLIKE per TEST year for the model and for HAR, and the share
    of stocks where the model's own calibrated QLIKE is below HAR's."""
    if "holdout" in x and x.holdout.any():
        x = x[x.holdout == True]  # noqa: E712
    yrs = {}
    for y, g in x.groupby(x.date.dt.year):
        yrs[str(y)] = {"model": float(qlike_parts(g.y.values, g.p.values)[1].mean()), "HAR": float(qlike_parts(g.y.values, g.HAR.values)[1].mean())}
    wins, n = 0, 0
    for _, g in x.groupby("symbol"):
        n += 1
        wins += qlike_parts(g.y.values, g.p.values)[1].mean() < qlike_parts(g.y.values, g.HAR.values)[1].mean()
    return {"years": yrs, "stock_wins_har": wins / n if n else None, "stocks": n}


def monthly(x):
    _, qlc, _ = qlike_parts(x.y.values, x.p.values)
    s = pd.Series(qlc, index=x.date).groupby(pd.DatetimeIndex(x.date).to_period("M")).mean()
    return {str(k): float(v) for k, v in s.items()}


def calibration(x, bins=15):
    q = pd.qcut(x.p, bins, labels=False, duplicates="drop")
    g = x.groupby(q).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return [{"pred": float(r.p), "real": float(r.y), "n": int(r.n)} for r in g.itertuples()]


# ------------------------------------------------------------------ runs
def card_of(name):
    f = RUNS / name / "card.json"
    return json.loads(f.read_text()) if f.exists() else None


def preds_of(name):
    f = RUNS / name / "test_preds.parquet"
    key = ("preds", name, f.stat().st_mtime if f.exists() else 0)
    if key not in CACHE:
        CACHE[key] = pd.read_parquet(f) if f.exists() else None
    return CACHE[key]


def frame(preds, T):
    x = T.merge(preds[["date", "symbol", "pred"] + (["holdout"] if "holdout" in preds else [])].rename(columns={"pred": "p"}),
                on=["date", "symbol"])
    return x.dropna(subset=["p", "y"])


def ensemble_preds(names):
    parts = [preds_of(n) for n in names]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    hold = parts[0][["date", "symbol", "holdout"]] if "holdout" in parts[0] else None
    m = pd.concat([p.set_index(["date", "symbol"]).pred for p in parts], axis=1).mean(axis=1).rename("pred").reset_index()
    return m.merge(hold, on=["date", "symbol"], how="left") if hold is not None else m


def scan_logs():
    """Run name -> what the logs say: window counts, step lines, last write."""
    key = ("logs", tuple(sorted((f.name, f.stat().st_mtime) for f in ROOT.glob("*.log"))))
    if key in CACHE:
        return CACHE[key]
    head = re.compile(r"^(\S+): train ([\d,]+)\s+val ([\d,]+)\s+test ([\d,]+) windows", re.M)
    step = re.compile(r"step (\d+)/(\d+)\s+(\d+)s\s+val r2 ([-\d.]+)\s+qlike ([-\d.]+)\s+ic ([-\d.]+)")
    out = {}
    for f in ROOT.glob("*.log"):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        m = head.search(txt)
        if not m:
            continue
        name, mt = m.group(1), f.stat().st_mtime
        if name in out and out[name]["mtime"] > mt:
            continue
        steps = [{"step": int(s[0]), "steps": int(s[1]), "secs": int(s[2]), "val_r2": float(s[3]), "val_qlike": float(s[4]),
                  "val_ic": float(s[5])} for s in step.findall(txt)]
        tail = [ln for ln in txt.splitlines() if ln.strip() and "Warning" not in ln and "warn" not in ln.lower()
                and "WinError" not in ln and "result = getattr" not in ln and "LOKY" not in ln][-40:]
        out[name] = {"log": f.name, "mtime": mt, "windows": {k: int(v.replace(",", "")) for k, v in
                     zip(("train", "val", "test"), m.groups()[1:])}, "steps": steps, "tail": tail}
    CACHE.clear() if len(CACHE) > 4000 else None
    CACHE[key] = out
    return out


def run_names():
    return sorted(p.name for p in RUNS.iterdir() if p.is_dir())


def live_of(name):
    f = RUNS / name / "live.json"
    if not f.exists():
        return None
    try:
        v = json.loads(f.read_text())
        v["mtime"] = f.stat().st_mtime
        return v
    except Exception:                                 # caught mid-write: next poll gets it
        return None


def status_of(name, logs):
    card_f = RUNS / name / "card.json"
    lg = logs.get(name)
    lv = RUNS / name / "live.json"
    if not card_f.exists() and lv.exists() and time.time() - lv.stat().st_mtime < 900:
        return "running"
    if card_f.exists():
        if lg and lg["mtime"] > card_f.stat().st_mtime + 120 and time.time() - lg["mtime"] < 1200:
            return "running"                          # re-run in progress over an older result
        return "completed"
    if lg and time.time() - lg["mtime"] < 1200:
        return "running"
    if (RUNS / name).exists() and time.time() - (RUNS / name).stat().st_mtime < 900:
        return "starting"
    return "stopped"


def run_summary(name, logs):
    card = card_of(name)
    lg = logs.get(name)
    st = status_of(name, logs)
    if card is None:
        args = describe(name, {"args": guess_args(name)})
        prog = None
        lv = live_of(name)
        if lv:
            h = lv.get("history") or []
            prog = {"step": lv["step"], "steps": lv["steps"], "secs": lv["secs"],
                    "eta": lv["secs"] / max(lv["step"], 1) * (lv["steps"] - lv["step"]),
                    "val_qlike": h[-1]["val_qlike"] if h else None,
                    "train_loss": lv["train_loss"][-1][1] if lv.get("train_loss") else None}
        elif lg and lg["steps"]:
            s = lg["steps"][-1]
            prog = {"step": s["step"], "steps": s["steps"], "secs": s["secs"],
                    "eta": s["secs"] / max(s["step"], 1) * (s["steps"] - s["step"]), "val_qlike": s["val_qlike"]}
        return {**args, "status": st, "progress": prog, "windows": lg["windows"] if lg else None,
                "updated": lg["mtime"] if lg else (RUNS / name).stat().st_mtime}
    m = describe(name, card)
    f = RUNS / name / "test_preds.parquet"
    key = ("summary", name, f.stat().st_mtime if f.exists() else 0, (RUNS / name / "card.json").stat().st_mtime)
    if key not in CACHE:
        T, _ = truth(m["panel"], m["target"])
        x = frame(preds_of(name), T)
        s = stats(x)
        har = x.assign(p=x.HAR)
        sh = stats(har)
        out = {"test": s, "har": sh, "vs_har_pct": (sh["qlike_cal"] - s["qlike_cal"]) / sh["qlike_cal"] * 100}
        if "holdout" in x and x.holdout.any():
            hx = x[x.holdout == True]  # noqa: E712
            out["heldout_test"] = stats(hx)
            out["heldout_har"] = stats(hx.assign(p=hx.HAR))
        CACHE[key] = out
    res = CACHE[key]
    return {**m, **res, "status": st, "minutes": card.get("minutes"), "best_step": card.get("best_step"),
            "windows": lg["windows"] if lg else None, "updated": (RUNS / name / "card.json").stat().st_mtime,
            "has_train_loss": bool(card.get("train_loss"))}


def guess_args(name):
    """For a run that has not written its card yet: read what the name says."""
    a = {"mode": "full", "tag": "", "seed": 0}
    v = re.match(r"vol_(full|none|lora|frozen)(_randinit)?_L(\d+)_\w+?_s(\d+)$", name)
    if v:
        a.update(mode=v.group(1), random_init=bool(v.group(2)), L=int(v.group(3)), seed=int(v.group(4)), target="lfv5")
        return a
    m = re.match(r"(intra|vol)_(\w+?)_(full|none|lora|frozen)(_randinit)?_L(\d+)(.*)_s(\d+)$", name)
    if m:
        a.update(mode=m.group(3), random_init=bool(m.group(4)), L=int(m.group(5)), seed=int(m.group(7)), target=m.group(2))
        rest = m.group(6)
        t = re.search(r"(_u200\w*)$", rest)
        a["tag"] = t.group(1) if t else ""
        for k, pat in (("loss", r"_(qlike|mix)"), ("pool", r"_pool(day|both)")):
            mm = re.search(pat, rest)
            if mm:
                a[k] = mm.group(1)
        if "_base" in rest:
            a["model"] = "base"
        if a["tag"].startswith("_u200"):
            a["panel"] = "intra_daily200.parquet"; a["holdout"] = 40
    return a


def run_detail(name):
    logs = scan_logs()
    summ = run_summary(name, logs)
    card = card_of(name)
    lg = logs.get(name)
    out = {"summary": summ, "card": card, "log": lg}
    if card is None:
        out["live"] = live_of(name)
        return out
    m = describe(name, card)
    T, tr_n = truth(m["panel"], m["target"])
    x = frame(preds_of(name), T)
    out["monthly"] = {"model": monthly(x)}
    out["calibration"] = calibration(x)
    out["baselines"] = {}
    for b in baselines_for(m["panel"]):
        bx = x.assign(p=x[b])
        out["baselines"][b] = {"label": BASE_LABEL[b], "test": stats(bx), "dm": dm(x[["date", "symbol", "y", "p"]], bx[["date", "symbol", "y", "p"]])}
        out["monthly"][b] = monthly(bx)
    # the family's head-only control, averaged over its seeds
    ctrl = [n for n in run_names() if (RUNS / n / "card.json").exists() and describe(n, card_of(n))["family"] == m["family"]
            and describe(n, card_of(n))["arm"] == "control" and not describe(n, card_of(n))["variant"]]
    if ctrl and m["arm"] != "control":
        cx = frame(ensemble_preds(ctrl), T)
        out["control"] = {"runs": ctrl, "test": stats(cx), "dm": dm(x[["date", "symbol", "y", "p"]], cx[["date", "symbol", "y", "p"]])}
        out["monthly"]["control"] = monthly(cx)
    if "holdout" in x and x.holdout.any():
        out["subsets"] = {k: stats(x[x.holdout == v]) for k, v in (("seen", False), ("held-out", True))}
    out["by_year"] = {str(y): stats(g) for y, g in x.groupby(x.date.dt.year)}
    per = []
    for sym, g in x.groupby("symbol"):
        _, qm, _ = qlike_parts(g.y.values, g.p.values)
        _, qh, _ = qlike_parts(g.y.values, g.HAR.values)
        per.append({"symbol": sym, "n": int(len(g)), "heldout": bool(g.holdout.iloc[0]) if "holdout" in g else False,
                    "qlike_cal": float(qm.mean()), "har": float(qh.mean()), "gain": float((qh.mean() - qm.mean()) / qh.mean() * 100)})
    out["stocks"] = sorted(per, key=lambda r: -r["gain"])
    out["truth_note"] = {"rows": int(len(x)), "har_train_stocks": tr_n}
    return out


# ------------------------------------------------------------------ experiments (seed ensembles)
def experiments():
    names = [n for n in run_names() if (RUNS / n / "card.json").exists() and (RUNS / n / "test_preds.parquet").exists()]
    key = ("exp", tuple((n, (RUNS / n / "test_preds.parquet").stat().st_mtime) for n in names))
    if key in CACHE:
        return CACHE[key]
    meta = {n: describe(n, card_of(n)) for n in names}
    fams = {}
    for n, m in meta.items():
        fams.setdefault(m["family"], {}).setdefault(m["group"], []).append(n)
    out = []
    for fam, groups in fams.items():
        any_m = meta[next(iter(groups.values()))[0]]
        T, tr_n = truth(any_m["panel"], any_m["target"])
        ens = {g: frame(ensemble_preds(ns), T) for g, ns in groups.items()}
        ctrl_g = next((g for g, ns in groups.items() if meta[ns[0]]["arm"] == "control" and not meta[ns[0]]["variant"]), None)
        # a variant is compared with the control trained the same way (same loss) when that control exists
        ctrl_var = {meta[ns[0]]["variant"]: g for g, ns in groups.items() if meta[ns[0]]["arm"] == "control"}
        rows = []
        for g, ns in groups.items():
            x = ens[g]
            m0 = meta[ns[0]]
            row = {"group": g, "arm": m0["arm"], "arm_label": m0["arm_label"], "variant": m0["variant"], "runs": sorted(ns),
                   "seeds": len(ns), "all": stats(x)}
            per = []
            for n in sorted(ns):
                xi = frame(preds_of(n), T)
                hi = xi[xi.holdout == True] if "holdout" in xi and xi.holdout.any() else None  # noqa: E712
                per.append({"run": n, "seed": meta[n]["seed"], "qlike_cal": stats(xi)["qlike_cal"],
                            "heldout_cal": stats(hi)["qlike_cal"] if hi is not None and len(hi) else None})
            row["per_seed"] = per
            if "holdout" in x and x.holdout.notna().any() and x.holdout.any():
                row["seen"] = stats(x[x.holdout == False])  # noqa: E712
                row["heldout"] = stats(x[x.holdout == True])  # noqa: E712
            cg = ctrl_var.get(m0["variant"], ctrl_g)
            if cg and g != cg:
                c = ens[cg]
                row["control_used"] = cg
                row["dm_vs_control"] = dm(x[["date", "symbol", "y", "p"]], c[["date", "symbol", "y", "p"]])
                if "heldout" in row:
                    row["dm_vs_control_heldout"] = dm(x[x.holdout == True][["date", "symbol", "y", "p"]],  # noqa: E712
                                                      c[c.holdout == True][["date", "symbol", "y", "p"]])  # noqa: E712
            har = x.assign(p=x.HAR)
            row["dm_vs_har"] = dm(x[["date", "symbol", "y", "p"]], har[["date", "symbol", "y", "p"]])
            row["robust"] = robustness(x)
            rows.append(row)
        # recipe search: a variant against the default recipe of its arm, with the same seeds, on held-out stocks if any
        for row in rows:
            if not row["variant"]:
                continue
            b0 = next((r for r in rows if r["arm"] == row["arm"] and not r["variant"]), None)
            if not b0:
                continue
            seeds = sorted({meta[n]["seed"] for n in row["runs"]})
            bruns = [n for n in b0["runs"] if meta[n]["seed"] in seeds] or b0["runs"]
            bx = frame(ensemble_preds(bruns), T)
            x = ens[row["group"]]
            sub = "holdout" in x and x.holdout.any()
            if sub:
                x, bx = x[x.holdout == True], bx[bx.holdout == True]  # noqa: E712
            sx, sb = stats(x), stats(bx)
            d = dm(x[["date", "symbol", "y", "p"]], bx[["date", "symbol", "y", "p"]])
            row["vs_default"] = {"default_group": b0["group"], "default_runs": bruns, "seeds": seeds,
                                 "subset": "held-out" if sub else "all", "model": sx["qlike_cal"], "default": sb["qlike_cal"],
                                 "r2": sx["r2"], "default_r2": sb["r2"], "bias": sx["bias"], "default_bias": sb["bias"],
                                 "t": d["qlike_cal"] if d else None, "n": d["n"] if d else None}
        # baselines on the family's rows (the union of what the runs scored)
        x0 = ens[next(iter(groups))]
        base = []
        for b in baselines_for(any_m["panel"]):
            bx = x0.assign(p=x0[b])
            r = {"group": b, "arm": "baseline", "arm_label": BASE_LABEL[b], "all": stats(bx), "seeds": 0}
            if "holdout" in bx and bx.holdout.any():
                r["seen"] = stats(bx[bx.holdout == False])  # noqa: E712
                r["heldout"] = stats(bx[bx.holdout == True])  # noqa: E712
            base.append(r)
        order = {"kronos": 0, "randinit": 1, "lora": 2, "frozen": 3, "control": 4}
        rows.sort(key=lambda r: (order.get(r["arm"], 9), r["variant"]))
        info = families().get(fam, {"label": fam, "detail": "", "order": 9})
        n_train = any_m["train_stocks"] or ((tr_n - (any_m["holdout"] or 0)) if tr_n else None)
        out.append({"family": fam, **info, "target": any_m["target"], "groups": rows, "baselines": base,
                    "control_group": ctrl_g, "train_stocks": n_train})
    out.sort(key=lambda f: f["order"])
    CACHE[key] = out
    return out


def stock_series(name, sym):
    m = describe(name, card_of(name))
    T, _ = truth(m["panel"], m["target"])
    x = frame(preds_of(name), T)
    g = x[x.symbol == sym].sort_values("date")
    out = {"symbol": sym, "dates": [d.strftime("%Y-%m-%d") for d in g.date], "real": g.y.tolist(), "model": g.p.tolist(), "HAR": g.HAR.tolist()}
    ctrl = [n for n in run_names() if (RUNS / n / "card.json").exists() and describe(n, card_of(n))["family"] == m["family"]
            and describe(n, card_of(n))["arm"] == "control" and not describe(n, card_of(n))["variant"]]
    if ctrl and m["arm"] != "control":
        c = frame(ensemble_preds(ctrl), T)
        c = c[c.symbol == sym].set_index("date").p.reindex(g.date)
        out["control"] = [None if not np.isfinite(v) else float(v) for v in c.values]
    return out


def queue():
    """Jobs waiting in queue_*.sh files (ft_intra.py lines), with the run name each will write."""
    items = []
    for f in sorted(ROOT.glob("queue_*.sh")):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        var = dict(re.findall(r'^(\w+)="([^"]*)"', txt, re.M))
        for ln in txt.splitlines():
            if "ft_intra.py" not in ln or ln.strip().startswith("#"):
                continue
            cmd = ln.split(">")[0]
            for k, v in var.items():
                cmd = cmd.replace("$" + k, v)
            toks = cmd.split()
            a = {"mode": "full", "target": "lrv1", "L": 256, "tag": "", "seed": 0, "loss": "mse", "pool": "last", "model": "small", "random_init": False}
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == "--random_init":
                    a["random_init"] = True
                elif t.startswith("--") and i + 1 < len(toks) and not toks[i + 1].startswith("--"):   # --save etc. take no value
                    a[t[2:]] = toks[i + 1]; i += 1
                i += 1
            xtra = (f"_{a['loss']}" if a["loss"] != "mse" else "") + (f"_pool{a['pool']}" if a["pool"] != "last" else "") + ("_base" if a["model"] == "base" else "")
            name = f"intra_{a['target']}_{a['mode']}{'_randinit' if a['random_init'] else ''}_L{a['L']}{xtra}{a['tag']}_s{a['seed']}"
            done = (RUNS / name / "card.json").exists()
            st = "done" if done else ("running" if (RUNS / name).exists() else "queued")
            j = next((k for k, t in enumerate(toks) if t.endswith("ft_intra.py")), 0)
            items.append({"queue": f.name, "run": name, "status": st, "cmd": " ".join(toks[j:])})
    return items


def compare(groups, ref, subset):
    ms = []
    for g in groups:
        names = g.split("+")
        if names[0] in ("HAR", "HARPLUS", "GBM"):
            continue
        ms.append(describe(names[0], card_of(names[0])))
    if not ms:
        return {"error": "pick at least one run"}
    panel, target = ms[0]["panel"], ms[0]["target"]
    T, _ = truth(panel, target)
    frames = {}
    for g in groups:
        names = g.split("+")
        if names[0] in ("HAR", "HARPLUS", "GBM"):
            continue
        x = frame(ensemble_preds(names), T)
        if subset != "all" and "holdout" in x:
            x = x[x.holdout == (subset == "held-out")]
        frames[g] = x
    keys = None
    for x in frames.values():
        k = x[["date", "symbol"]]
        keys = k if keys is None else keys.merge(k, on=["date", "symbol"])
    for g in groups:
        if g in ("HAR", "HARPLUS", "GBM"):
            frames[g] = T.merge(keys, on=["date", "symbol"]).assign(p=lambda d, g=g: d[g])
    frames = {g: x.merge(keys, on=["date", "symbol"]) for g, x in frames.items()}
    ref = ref if ref in frames else groups[0]
    R = frames[ref][["date", "symbol", "y", "p"]]
    rows = []
    for g in groups:
        x = frames[g]
        rows.append({"group": g, "test": stats(x), "dm": None if g == ref else dm(x[["date", "symbol", "y", "p"]], R),
                     "monthly": monthly(x), "by_year": {str(y): stats(gg)["qlike_cal"] for y, gg in x.groupby(x.date.dt.year)}})
    return {"panel": panel, "target": target, "ref": ref, "subset": subset, "rows": rows, "n_common": int(len(keys))}


# ------------------------------------------------------------------ LLM fine-tunes (LoRA runs and their scores, folder FF)
TASKS = {
    "t1": {"label": "8 numbers from a 10-K statement", "detail": "Balance sheet, income or cash-flow statement text in, 8 fields out as JSON. Scored on 286 held-back statements; 50 from companies never seen in training."},
    "t4": {"label": "GAAP numbers from an earnings release", "detail": "Earnings-release text in, the GAAP figures out as JSON. Scored on 300 held-back releases."},
}
EXTERNAL = {"gemma4_31b": "Gemma 4 31B, out of the box (local)", "gemini_flash": "Gemini Flash, out of the box (API)"}


def llm_label(name, score):
    tail = name.split("_", 1)[1] if "_" in name else name
    if tail in EXTERNAL:
        return EXTERNAL[tail], False
    r = score.get("run", {})
    m = (r.get("model") or "").split("/")[-1].replace("Qwen3.5-", "Qwen3.5 ")
    tuned = bool(r.get("adapter")) or "tuned" in name
    where = f", {r['machine']}" if r.get("machine") else (", CUDA run" if name.endswith("_cuda") else "")   # same recipe, two machines
    return f"{m}, {'fine-tuned (LoRA)' if tuned else 'out of the box'}{where}", tuned


def llm():
    if not FF.exists():
        return {"tasks": [], "error": f"no LLM folder at {FF} (pass --llm)"}
    key = ("llm", tuple(sorted((str(f), f.stat().st_mtime) for f in list(FF.glob("evals/*/score*.json")) + list(FF.glob("runs/*/run_card.json")))))
    if key in CACHE:
        return CACHE[key]
    tasks = {k: {"task": k, **v, "evals": [], "runs": []} for k, v in TASKS.items()}
    for d in sorted(FF.glob("evals/*")):
        if d.name.startswith("_") or not d.is_dir():
            continue
        f = d / "score.json"
        partial = not f.exists()
        if partial:
            f = d / "score_partial.json"
        if not f.exists():
            continue
        sc = json.loads(f.read_text())
        t = d.name.split("_")[0]
        if t not in tasks:
            continue
        label, tuned = llm_label(d.name, sc)
        tasks[t]["evals"].append({"name": d.name, "label": label, "tuned": tuned, "partial": partial, "all": sc.get("all", {}),
                                  "seen": sc.get("seen_companies"), "unseen": sc.get("unseen_companies"), "run": sc.get("run", {}),
                                  "parts": {k: v for k, v in sc.items() if k not in ("all", "run", "seen_companies", "unseen_companies")}})
    for d in sorted(FF.glob("runs/*")):
        c = d / "run_card.json"
        if d.name.startswith("_") or not c.exists():
            continue
        card = json.loads(c.read_text())
        if not card.get("model") or "kronos" in str(card.get("model")).lower():   # LLM runs only
            continue
        t = d.name.split("_")[0]
        loss = []
        lf = d / "loss.csv"
        if lf.exists() and lf.stat().st_size > 0:
            L = pd.read_csv(lf)
            loss = L[[c_ for c_ in ("step", "epoch", "loss", "token_acc", "lr", "tokens_per_s", "gpu_peak_gb") if c_ in L]].to_dict("records")
        card.pop("lora_kinds", None)
        tasks.setdefault(t, {"task": t, "label": t, "detail": "", "evals": [], "runs": []})["runs"].append({"name": d.name, "card": card, "loss": loss})
    for t in tasks.values():
        t["evals"].sort(key=lambda e: -(e["all"].get("statement_acc") or 0))
    out = {"tasks": [t for t in tasks.values() if t["evals"] or t["runs"]], "source": FF.name}
    CACHE[key] = out
    return out


# ------------------------------------------------------------------ machine
class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


_sys = {"t": 0, "v": None}


def system():
    if time.time() - _sys["t"] < 4 and _sys["v"]:
        return _sys["v"]
    v = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        nm, util, used, tot, temp = [s.strip() for s in r.stdout.strip().split(",")]
        v["gpu"] = {"name": nm, "util": float(util), "mem_used": float(used) / 1024, "mem_total": float(tot) / 1024, "temp": float(temp)}
    except Exception:
        v["gpu"] = None
    try:
        ms = MEMSTAT(); ms.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        v["ram"] = {"used": (ms.ullTotalPhys - ms.ullAvailPhys) / 2 ** 30, "total": ms.ullTotalPhys / 2 ** 30}
    except Exception:                                 # not Windows: POSIX page counts where available
        try:
            pg = os.sysconf("SC_PAGE_SIZE")
            tot, av = os.sysconf("SC_PHYS_PAGES") * pg, os.sysconf("SC_AVPHYS_PAGES") * pg
            v["ram"] = {"used": (tot - av) / 2 ** 30, "total": tot / 2 ** 30}
        except (AttributeError, ValueError, OSError):
            v["ram"] = None
    du = shutil.disk_usage(ROOT.anchor)
    v["disk"] = {"free": du.free / 2 ** 30, "total": du.total / 2 ** 30, "drive": ROOT.anchor}
    _sys.update(t=time.time(), v=v)
    return v


T1_ORDER = ["buy_hold", "HAR", "HARPLUS", "control", "kronos"]


def final_tests():
    """Outputs of the final-test scripts, read as they are: t1 (backtest), t34 (accuracy), t6 (edge), t2* (agent), ledger."""
    out = {}
    f1 = FT / "t1_daily.parquet"
    if f1.exists():
        key = ("final_t1", f1.stat().st_mtime)
        if key not in CACHE:
            X = pd.read_parquet(f1)
            B = X.groupby(["arm", "date"]).pnl.mean().unstack(0).sort_index()
            ser = {}
            for arm in [a for a in T1_ORDER if a in B]:
                pnl = B[arm].fillna(0.0)
                eq = (1 + pnl).cumprod()
                rv = pnl.rolling(21).std() * np.sqrt(252)
                ser[arm] = {"equity": [round(float(v), 5) for v in eq], "vol21": [None if not np.isfinite(v) else round(float(v), 5) for v in rv]}
            res = pd.read_csv(FT / "t1_results.csv")
            res = res.astype(object).where(res.notna(), None)
            CACHE[key] = {"dates": [d.strftime("%Y-%m-%d") for d in B.index], "series": ser, "results": res.to_dict("records"),
                          "names": sorted(X.symbol.unique())}
        out["t1"] = CACHE[key]
    for nm, k in (("t34_results.json", "t34"), ("t6_results.json", "t6")):
        f = FT / nm
        if f.exists():
            out[k] = json.loads(f.read_text(encoding="utf-8"))
    # agent tests: one file per model (t2_results*.json, each with a "model" field); complete runs first
    t2 = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(FT.glob("t2_results*.json"))]
    out["t2s"] = sorted(t2, key=lambda t: -(t.get("decisions", 0) / max(t.get("planned") or t.get("decisions", 1), 1)))
    led = FT / "ledger.jsonl"
    if led.exists():
        rows = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines() if x.strip()]
        rf = FT / "ledger_realised.json"                  # realised values live apart: the ledger is never rewritten
        real = json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else {}
        for r in rows:
            r["realised_vol"] = real.get(f"{r['symbol']}|{r['as_of']}", {}).get("realised_vol")
        out["ledger"] = rows[-60:]
        out["ledger_n"] = len(rows)
    out["has_kronos"] = any(r["arm"] == "kronos" for r in out.get("t1", {}).get("results", []))
    return out


def takes():
    out = []
    for f in sorted(TAKES.glob("*.mp4")):
        ev = f.with_suffix(".events.json")
        e = json.loads(ev.read_text()) if ev.exists() else {}
        out.append({"file": f.name, "mb": f.stat().st_size / 2 ** 20, "seconds": e.get("seconds"), "events": e.get("events", []),
                    "made": f.stat().st_mtime})
    return out


# ------------------------------------------------------------------ http
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(clean(obj)).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, f, ctype=None):
        size = f.stat().st_size
        ctype = ctype or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(f, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = unquote(u.path)
        try:
            if p == "/" or p == "/index.html":
                return self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
            if p.startswith("/static/"):
                f = (STATIC / p[len("/static/"):]).resolve()
                if STATIC.resolve() in f.parents and f.is_file():
                    return self.send_file(f)
                return self.send_json({"error": "not found"}, 404)
            if p.startswith("/takes/"):
                f = (TAKES / p[len("/takes/"):]).resolve()
                if TAKES.resolve() in f.parents and f.is_file() and f.suffix == ".mp4":
                    return self.send_file(f, "video/mp4")
                return self.send_json({"error": "not found"}, 404)
            with LOCK:
                if p == "/api/runs":
                    logs = scan_logs()
                    names = set(run_names()) | {n for n in logs if (RUNS / n).exists()}
                    return self.send_json({"runs": [run_summary(n, logs) for n in sorted(names)], "families": families(), "arms": ARMS,
                                           "now": time.time()})
                if p.startswith("/api/run/"):
                    name = p[len("/api/run/"):]
                    if not SAFE.match(name) or not (RUNS / name).is_dir():
                        return self.send_json({"error": "no such run"}, 404)
                    return self.send_json(run_detail(name))
                if p == "/api/experiments":
                    return self.send_json({"families": experiments()})
                if p == "/api/compare":
                    return self.send_json(compare(q.get("g", []), q.get("ref", [""])[0], q.get("subset", ["all"])[0]))
            if p == "/api/queue":
                return self.send_json({"items": queue()})
            if p == "/api/stock":
                run = q.get("run", [""])[0]
                if not SAFE.match(run) or not (RUNS / run / "card.json").exists():
                    return self.send_json({"error": "no such run"}, 404)
                with LOCK:
                    return self.send_json(stock_series(run, q.get("sym", [""])[0]))
            if p == "/api/llm":
                return self.send_json(llm())
            if p == "/api/system":
                return self.send_json(system())
            if p == "/api/final":
                with LOCK:
                    return self.send_json(final_tests())
            if p == "/api/takes":
                return self.send_json({"takes": takes()})
            if p == "/api/info":
                return self.send_json({"root": ROOT.name, "demo": (ROOT / "DEMO.txt").exists()})
            return self.send_json({"error": "not found"}, 404)
        except Exception as e:  # keep the dashboard up; show the error in the page
            import traceback
            traceback.print_exc()
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


def warm():
    with LOCK:
        try:
            logs = scan_logs()
            for n in run_names():
                run_summary(n, logs)
            experiments()
        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LooksFine: a local dashboard for fine-tune runs")
    ap.add_argument("--root", help="folder with runs/ and data/ (env LOOKSFINE_ROOT; default: the current folder)")
    ap.add_argument("--runs", help="runs folder (env LOOKSFINE_RUNS; default: ROOT/runs)")
    ap.add_argument("--data", help="folder with the panel parquet files (env LOOKSFINE_DATA; default: ROOT/data)")
    ap.add_argument("--final", help="final-test outputs (env LOOKSFINE_FINAL; default: ROOT/final_tests)")
    ap.add_argument("--llm", help="LLM fine-tunes folder (env LOOKSFINE_LLM; default: ROOT/llm)")
    ap.add_argument("--takes", help="screen recordings (env LOOKSFINE_TAKES; default: ROOT/rec/takes)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LOOKSFINE_PORT", 8765)))
    a = ap.parse_args()
    configure(a.root, a.runs, a.data, a.final, a.llm, a.takes)
    threading.Thread(target=warm, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"LooksFine on http://127.0.0.1:{a.port}  (runs: {RUNS})", flush=True)
    srv.serve_forever()
