"""Build a small synthetic root folder so LooksFine can be tried without any market data or GPU.
Every number in it is simulated (40 made-up stocks, DEMO01 to DEMO40); the dashboard says so on every page.
    python make_demo.py                 writes demo/ (about 8 MB) next to this file
    python make_demo.py --live          the same, then plays a training job for about 3 minutes (Jobs page, live charts)
    python server.py --root demo        then open http://127.0.0.1:8765
Layout written, the same one ft_intra.py and the Kronos-Vol scripts produce:
    data/intra_daily200.parquet         one row per stock and session: next-day log realised variance and the 9 features
    runs/<run>/card.json, test_preds.parquet, live.json      31 finished runs in 3 experiments (seeds, controls, recipes)
    run_<run>.log, queue_demo.sh        logs and a queue file, as a training session leaves them
    final_tests/, llm/                  outputs for the Final tests and LLM fine-tunes pages
"""
import argparse
import json
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "demo"))
ap.add_argument("--live", action="store_true", help="after building, play a running job (about 3 minutes)")
a = ap.parse_args()
OUT = Path(a.out)
if OUT.exists():
    if not (OUT / "DEMO.txt").exists():
        raise SystemExit(f"{OUT} exists and is not a demo folder: pick another --out")
    shutil.rmtree(OUT)
(OUT / "data").mkdir(parents=True)
(OUT / "DEMO.txt").write_text("Synthetic data made by make_demo.py. No number here comes from a real market or a real model.\n")
rng = np.random.default_rng(7)
F = ["lrv_d", "lrv_w", "lrv_m", "l_on", "l_last", "ret_d", "spy_lrv_d", "spy_lrv_w", "spy_lrv_m"]
TRAIN, VAL, TEST = ("2021-07-01", "2023-12-31"), ("2024-01-01", "2024-06-30"), ("2024-07-01", "2026-11-10")

# ------------------------------------------------------------------ a panel of simulated stocks
dates = pd.bdate_range("2021-06-01", "2026-11-17")
T_ = len(dates)
syms = [f"DEMO{i:02d}" for i in range(1, 41)]
HOLD = set(syms[30:])                                     # 10 stocks never trained on
m = np.zeros(T_)
for t in range(1, T_):                                    # market log-variance factor, persistent
    m[t] = 0.96 * m[t - 1] + 0.18 * rng.standard_normal()


def har_feats(lrv):
    rv = pd.Series(np.exp(lrv))
    return np.log(rv), np.log(rv.rolling(5).mean()), np.log(rv.rolling(22).mean())


spy_l = -9.8 + m + 0.35 * rng.standard_normal(T_)
spy_d, spy_w, spy_m = har_feats(spy_l)
rows, latent_next = [], []
for s in syms:
    a_ = np.zeros(T_)
    for t in range(1, T_):                                # stock-specific factor
        a_[t] = 0.92 * a_[t - 1] + 0.22 * rng.standard_normal()
    mu = rng.normal(-8.5, 0.35)
    h = mu + m + a_                                       # latent log variance
    late = 0.25 * rng.standard_normal(T_)                 # how the last hour traded: part of it carries into tomorrow
    lrv = h + 0.45 * rng.standard_normal(T_) + 0.4 * np.r_[0.0, late[:-1]]    # what 5-minute bars would measure
    d, w, mo = har_feats(lrv)
    df = pd.DataFrame({"date": dates, "symbol": s, "lrv_d": d, "lrv_w": w, "lrv_m": mo,
                       "l_on": lrv - 2.5 + 2.4 * rng.standard_normal(T_), "l_last": lrv - 2.4 + late,
                       "ret_d": np.exp(lrv / 2) * rng.standard_normal(T_) * 1.1, "spy_lrv_d": spy_d.values,
                       "spy_lrv_w": spy_w.values, "spy_lrv_m": spy_m.values, "lrv1": pd.Series(lrv).shift(-1).values})
    df["oracle"] = mu + 0.96 * m + 0.92 * a_ + 0.4 * late   # the best possible forecast of tomorrow
    rows.append(df)
P = pd.concat(rows, ignore_index=True).dropna(subset=F)
P[["date", "symbol"] + F + ["lrv1"]].astype({c: "float32" for c in F + ["lrv1"]}).to_parquet(OUT / "data" / "intra_daily200.parquet", index=False)

tr = P[(P.date >= TRAIN[0]) & (P.date <= TRAIN[1])].dropna(subset=["lrv1"])
X = np.c_[np.ones(len(tr)), tr[["lrv_d", "lrv_w", "lrv_m"]].values]
beta = np.linalg.lstsq(X, tr.lrv1.values, rcond=None)[0]
P["HAR"] = np.c_[np.ones(len(P)), P[["lrv_d", "lrv_w", "lrv_m"]].values] @ beta
TE = P[(P.date >= TEST[0]) & (P.date <= TEST[1])].reset_index(drop=True)
VA = P[(P.date >= VAL[0]) & (P.date <= VAL[1])].reset_index(drop=True)

# ------------------------------------------------------------------ runs
# (tag, train stocks, epochs, [(arm, variant, quality, seeds)]): quality is how far a forecast moves from HAR to the oracle
PLAN = [
    ("_u200", 0, 3, [("kronos", "", 0.17, 3), ("kronos", "mix", 0.21, 3), ("randinit", "", 0.18, 3), ("control", "", 0.10, 3),
                     ("control", "mix", 0.10, 3)]),
    ("_u200n60", 20, 5, [("kronos", "", 0.16, 2), ("kronos", "mix", 0.20, 2), ("kronos", "qlike", 0.20, 1), ("kronos", "poolday", 0.18, 1),
                         ("randinit", "", 0.12, 2), ("control", "", 0.10, 2)]),
    ("_u200n20", 10, 8, [("kronos", "", 0.15, 2), ("randinit", "", 0.09, 2), ("control", "", 0.09, 2)]),
]


def run_name(arm, variant, tag, seed):
    mode = "none" if arm == "control" else "full"
    xtra = {"": "", "mix": "_mix", "qlike": "_qlike", "poolday": "_poolday"}[variant]
    return f"intra_lrv1_{mode}{'_randinit' if arm == 'randinit' else ''}_L256{xtra}{tag}_s{seed}"


def args_of(arm, variant, tag, seed, n_train, epochs):
    return {"mode": "none" if arm == "control" else "full", "L": 256, "epochs": float(epochs), "bs": 64, "lr_head": 0.001,
            "lr_bb": 0.0001 if arm == "randinit" else 1e-05, "lr_lora": 0.0002, "target": "lrv1", "seed": seed, "evals": 12,
            "random_init": arm == "randinit", "panel": "intra_daily200.parquet", "holdout": 10, "save": True,
            "loss": variant if variant in ("mix", "qlike") else "mse", "pool": "day" if variant == "poolday" else "last",
            "model": "small", "train_stocks": n_train, "tag": tag, "show": 0}


def curve(steps, q0, q1, r, bias):
    """VAL history (12 checks) and the train loss every 25 steps."""
    hist = []
    for k in range(12):
        st = int(steps * (k + 1) / 12)
        q = q1 + (q0 - q1) * np.exp(-3 * (k + 1) / 12) + abs(r.normal(0, 0.003))
        hist.append({"step": st, "val_n": int(len(VA)), "val_r2": round(0.47 + 0.05 * (1 - np.exp(-(k + 1) / 4)) + r.normal(0, 0.006), 4),
                     "val_mse": round(0.33 - 0.02 * (k / 11), 4), "val_qlike": round(q, 4), "val_cs_ic": round(0.735 + r.normal(0, 0.003), 4)})
    tl = [(s, round(float((0.62 if bias else 0.32) * (0.75 + 0.35 * np.exp(-s / (steps / 6))) * np.exp(r.normal(0, 0.12))), 5))
          for s in range(25, steps + 1, 25)]
    return hist, tl


def write_run(name, args, pred_te, minutes, r):
    d = OUT / "runs" / name
    d.mkdir(parents=True)
    n_train = int(26_000 * (args["train_stocks"] or 30) / 30)
    steps = int(args["epochs"] * n_train / 64)
    hist, tl = curve(steps, 0.175, 0.150 if args["mode"] != "none" else 0.157, r, args["loss"] != "mse")
    best = min(hist, key=lambda x: x["val_qlike"])
    card = {"run": name, "args": args, "best_step": best["step"], "minutes": minutes, "history": hist, "train_loss": tl}
    (d / "card.json").write_text(json.dumps(card, indent=1))
    (d / "live.json").write_text(json.dumps({"step": steps, "steps": steps, "secs": minutes * 60, "train_loss": tl, "history": hist, "args": args}))
    TE.assign(pred=pred_te.astype("float32"), holdout=TE.symbol.isin(HOLD))[["date", "symbol", "pred", "holdout"]].to_parquet(d / "test_preds.parquet", index=False)
    lines = [f"{name}: train {n_train:,}  val {len(VA):,}  test {len(TE):,} windows"]
    lines += [f"  step {h_['step']}/{steps}  {int(minutes * 60 * h_['step'] / steps)}s  val r2 {h_['val_r2']}  qlike {h_['val_qlike']}  ic {h_['val_cs_ic']}" for h_ in hist]
    lines += ["", f"{name}   (R2 log | QLIKE | cross-sectional IC)        VAL                      TEST", "  (demo run: synthetic numbers)"]
    (OUT / f"run_{name}.log").write_text("\n".join(lines) + "\n")


def forecast(df, q, r, hold_q=None):
    qq = np.where(df.symbol.isin(HOLD), hold_q if hold_q is not None else q, q)
    return df.HAR.values + qq * (df.oracle.values - df.HAR.values) + r.normal(0.04, 0.02) + r.normal(0, 0.06, len(df))


ens = {}
for tag, n_train, epochs, arms in PLAN:
    for arm, variant, q, seeds in arms:
        for seed in range(seeds):
            r = np.random.default_rng(zlib.crc32(f"{tag}|{arm}|{variant}|{seed}".encode()))
            hq = q * (0.85 if arm == "randinit" and n_train else 1.0)
            p = forecast(TE, q, r, hq) + (0.08 if variant == "qlike" else 0.0)
            name = run_name(arm, variant, tag, seed)
            mins = 1.1 if arm == "control" else round(47 * (n_train or 30) / 30 * epochs / 3 * r.uniform(0.95, 1.05), 1)
            write_run(name, args_of(arm, variant, tag, seed, n_train, epochs), p, mins, r)
            ens.setdefault((tag, arm, variant), []).append(p)
print(f"{sum(len(v) for v in ens.values())} runs written")

(OUT / "queue_demo.sh").write_text("#!/bin/bash\n# demo queue: one run done, one waiting\n"
                                    'B="--target lrv1 --panel intra_daily200.parquet --loss mix --holdout 10 --epochs 3 --tag _u200 --save"\n'
                                    "py -3.11 ft_intra.py --mode full $B --seed 2 > run_a.log 2>&1\n"
                                    "py -3.11 ft_intra.py --mode full $B --seed 3 > run_b.log 2>&1\n")
(OUT / "families.json").write_text(json.dumps({
    "u200": {"label": "Demo: 40 stocks, 10 held out", "short": "demo, 30 + 10 held out", "detail": "Synthetic. 30 training stocks, 10 never seen in training or validation.", "order": 1},
    "u200n60": {"label": "Demo: data efficiency, 20 training stocks", "short": "demo, 20 + 10 held out", "detail": "Synthetic. Same 10 held-out stocks, 20 training stocks.", "order": 2},
    "u200n20": {"label": "Demo: data efficiency, 10 training stocks", "short": "demo, 10 + 10 held out", "detail": "Synthetic. Same 10 held-out stocks, 10 training stocks.", "order": 3},
    "u200live": {"label": "Demo: live job", "short": "demo, live", "detail": "Synthetic job played by make_demo.py --live.", "order": 4}}, indent=1))


# ------------------------------------------------------------------ final tests (computed on the demo forecasts)
def qparts(y, p):
    r_ = np.exp(y - p)
    c = np.log(r_.mean())
    rc = np.exp(y - p - c)
    return r_ - np.log(r_) - 1, rc - np.log(rc) - 1


def nw_t(s, lags=5):
    s = np.asarray(s, float); s = s[np.isfinite(s)]; n = len(s); e = s - s.mean(); v = (e @ e) / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(s.mean() / np.sqrt(v / n))


FTD = OUT / "final_tests"
FTD.mkdir()
R = TE.assign(y=TE.lrv1, YDAY=TE.lrv_d, kronos=np.mean(ens[("_u200", "kronos", "mix")], 0),
              control=np.mean(ens[("_u200", "control", "mix")], 0)).dropna(subset=["y"])
t34 = {}
for key, label, mask in (("heldout", "TEST, held-out stocks (never trained on)", R.symbol.isin(HOLD)),
                         ("seen", "TEST, training stocks, later dates", ~R.symbol.isin(HOLD))):
    x = R[mask]
    _, hc = qparts(x.y.values, x.HAR.values)
    arms = []
    for arm in ("YDAY", "HAR", "control", "kronos"):
        q, qc = qparts(x.y.values, x[arm].values)
        ic = x.groupby("date").apply(lambda g: g[arm].corr(g.y, method="spearman"), include_groups=False).mean()
        dd = pd.Series(qc - hc, index=x.date.values).groupby(level=0).mean()
        arms.append({"arm": arm, "QLIKE": float(q.mean()), "cal_QLIKE": float(qc.mean()),
                     "R2": float(1 - ((x.y - x[arm]) ** 2).sum() / ((x.y - x.y.mean()) ** 2).sum()), "rank_IC": float(ic),
                     "t_vs_HAR": 0.0 if arm == "HAR" else nw_t(dd.values)})
    t34[key] = {"label": label, "rows": int(len(x)), "names": int(x.symbol.nunique()), "sessions": int(x.date.nunique()),
                "start": str(x.date.min().date()), "end": str(x.date.max().date()), "arms": arms}
(FTD / "t34_results.json").write_text(json.dumps(t34, indent=1))

# where the edge comes from: fine-tune minus control, by fifths of what the day looked like
x = R[R.symbol.isin(HOLD)].copy()
_, x["k"] = qparts(x.y.values, x.kronos.values)
_, x["c"] = qparts(x.y.values, x.control.values)
x["last_share"] = x.l_last - x.lrv_d
x["surprise"] = x.lrv_d - x.lrv_m
tot = (x.k - x.c).sum()
splits = {}
for col, label in (("last_share", "Share of today's variance in the last hour"), ("surprise", "Today's volatility against its own month")):
    x["qb"] = pd.qcut(x[col], 5, labels=False) + 1
    splits[col] = {"label": label, "buckets": [{"q": int(qb), "lo": float(g[col].min()), "hi": float(g[col].max()), "n": int(len(g)),
                                                "gap_pct": float((g.k.mean() - g.c.mean()) / g.c.mean() * 100),
                                                "share": float((g.k - g.c).sum() / tot * 100)} for qb, g in x.groupby("qb")]}
months = [{"month": str(mo), "gap_pct": float((g.k.mean() - g.c.mean()) / g.c.mean() * 100), "n": int(len(g))}
          for mo, g in x.groupby(x.date.dt.to_period("M"))]
(FTD / "t6_results.json").write_text(json.dumps({"rows": int(len(x)), "gap_pct": float((x.k.mean() - x.c.mean()) / x.c.mean() * 100),
                                                 "splits": splits, "months": months}, indent=1))

# a volatility-targeting backtest on 4 demo names: weight = sqrt(past median forecast / today's forecast), capped at 2
names = ["DEMO03", "DEMO11", "DEMO24", "DEMO35"]
bt, res = [], []
for s in names:
    g = R[R.symbol == s].sort_values("date")
    ret = np.exp(g.y.values / 2) * rng.standard_normal(len(g)) + 0.0006
    for arm in ("buy_hold", "HAR", "control", "kronos"):
        if arm == "buy_hold":
            w = np.ones(len(g))
        else:
            fv = np.exp(g[arm].values)
            med = pd.Series(fv).rolling(252, min_periods=20).median().shift(1).bfill().values
            w = np.minimum(np.sqrt(med / fv), 2.0)
        turn = np.abs(np.diff(np.r_[0.0, w]))
        bt.append(pd.DataFrame({"date": g.date.values, "symbol": s, "arm": arm, "w": w, "r": ret, "turn": turn, "pnl": w * ret - 0.0005 * turn}))
B = pd.concat(bt, ignore_index=True)
B.to_parquet(FTD / "t1_daily.parquet", index=False)
for (nm, arm), g in list(B.groupby(["symbol", "arm"])) + [(("BASKET", a_), gg.groupby("date").mean(numeric_only=True)) for a_, gg in B.groupby("arm")]:
    pnl = g.pnl.values
    eq = np.cumprod(1 + pnl)
    v21 = pd.Series(pnl).rolling(21).std().dropna() * np.sqrt(252)
    res.append({"name": nm, "arm": arm, "return": float(eq[-1] ** (252 / len(pnl)) - 1), "vol": float(pnl.std() * np.sqrt(252)),
                "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)), "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                "risk_cv": float(v21.std() / v21.mean()), "turnover": float(g.turn.sum() / len(pnl) * 252), "days": int(len(pnl))})
pd.DataFrame(res).to_csv(FTD / "t1_results.csv", index=False)
(FTD / "t2_results_demo.json").write_text(json.dumps({"model": "a demo agent", "decisions": 600, "planned": 600, "names": names, "note": "synthetic",
    "arms": [{"arm": "none", "mean_size": 40.0, "sharpe": 0.10, "vol": 0.08, "worst_5pct": -0.007, "days": 150, "follow": None},
             {"arm": "real", "mean_size": 43.5, "sharpe": 0.18, "vol": 0.085, "worst_5pct": -0.007, "days": 150, "follow": -0.55},
             {"arm": "fake", "mean_size": 42.8, "sharpe": 0.05, "vol": 0.09, "worst_5pct": -0.008, "days": 150, "follow": -0.41}]}, indent=1))
led = [{"logged_utc": "2026-11-25T21:17:20+00:00", "symbol": s, "as_of": "2026-11-25", "model": "demo", "kronos_vol": round(float(rng.uniform(0.15, 0.45)), 5),
        "har_vol": round(float(rng.uniform(0.15, 0.45)), 5), "prev_hash": "demo"} for s in names]
(FTD / "ledger.jsonl").write_text("".join(json.dumps(r_) + "\n" for r_ in led))

# ------------------------------------------------------------------ LLM fine-tunes page
L = OUT / "llm"
for nm, model, adapter, acc in (("t1_base_demo", "Demo/TinyLM-0.8B", None, 0.02), ("t1_tuned_demo", "Demo/TinyLM-0.8B", "runs/t1_lora_demo/adapter", 0.93),
                                 ("t1_base_large_demo", "Demo/TinyLM-4B", None, 0.41)):
    (L / "evals" / nm).mkdir(parents=True)

    def part(k, acc=acc):
        return {"n": 286 if k == "all" else 50, "field_acc": min(1, acc + 0.04), "statement_acc": acc, "within_1pct": min(1, acc + 0.02),
                "valid_json": 1.0, "errors": {"made_up": int(30 * (1 - acc)), "missed": int(20 * (1 - acc))}}
    (L / "evals" / nm / "score.json").write_text(json.dumps({"all": part("all"), "unseen_companies": part("u"), "run": {
        "model": model, "adapter": adapter, "examples": 286, "sec_per_example": 0.9 if adapter else 1.4}}, indent=1))
(L / "runs" / "t1_lora_demo").mkdir(parents=True)
steps = 240
(L / "runs" / "t1_lora_demo" / "run_card.json").write_text(json.dumps({
    "run": "t1_lora_demo", "note": "synthetic demo run", "model": "Demo/TinyLM-0.8B", "params_total": 800_000_000, "params_trainable": 10_800_000,
    "trainable_pct": 1.35, "lora_rank": 16, "examples": 1900, "tokens": 4_100_000, "max_len": 3072, "epochs": 2, "accum": 16, "steps": steps,
    "lr": 0.0002, "train_minutes": 127.0, "tokens_per_s": 397, "gpu_peak_gb": 3.18, "adapter_mb": 41.5, "started": "demo",
    "safety": {"gpu": "demo GPU"}, "val_history": [{"step": s, "loss": round(0.30 * np.exp(-s / 80) + 0.05, 4), "token_acc": round(0.95 + 0.04 * (1 - np.exp(-s / 60)), 4),
                                                   "exact": round(0.90 * (1 - np.exp(-s / 50)), 4)} for s in range(40, steps + 1, 40)]}, indent=1))
pd.DataFrame({"step": range(1, steps + 1), "epoch": np.arange(1, steps + 1) / (steps / 2),
              "loss": 0.25 * np.exp(-np.arange(steps) / 60) + 0.03 + np.abs(rng.normal(0, 0.01, steps))}).to_csv(L / "runs" / "t1_lora_demo" / "loss.csv", index=False)
mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2 ** 20
print(f"demo folder ready: {OUT} ({mb:.1f} MB). Run: python server.py --root \"{OUT}\"")

# ------------------------------------------------------------------ optional: a job that trains while you watch
if a.live:
    name = run_name("kronos", "mix", "_u200live", 0)
    args = args_of("kronos", "mix", "_u200live", 0, 0, 3)
    d = OUT / "runs" / name
    d.mkdir(parents=True)
    steps, r = 900, np.random.default_rng(1)
    hist_all, tl_all = curve(steps, 0.175, 0.150, r, True)
    log = OUT / f"run_{name}.log"
    log.write_text(f"{name}: train 26,000  val {len(VA):,}  test {len(TE):,} windows\n")
    print(f"playing {name}: open the Jobs page (Ctrl+C stops)", flush=True)
    t0 = time.time()
    for step in range(5, steps + 1, 5):
        time.sleep(1)
        secs = time.time() - t0
        hist = [h_ for h_ in hist_all if h_["step"] <= step]
        tl = [x_ for x_ in tl_all if x_[0] <= step]
        (d / "live.json").write_text(json.dumps({"step": step, "steps": steps, "secs": round(secs, 1), "train_loss": tl, "history": hist, "args": args}))
        if hist and hist[-1]["step"] > step - 5:
            h_ = hist[-1]
            with open(log, "a") as fh:
                fh.write(f"  step {h_['step']}/{steps}  {int(secs)}s  val r2 {h_['val_r2']}  qlike {h_['val_qlike']}  ic {h_['val_cs_ic']}\n")
    shutil.rmtree(d)
    write_run(name, args, forecast(TE, 0.21, r), round((time.time() - t0) / 60, 1), r)
    print("live job finished", flush=True)
