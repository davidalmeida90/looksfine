# LooksFine

A local dashboard for fine-tuning runs. It was built to follow the fine-tune of Kronos-small into
[Kronos-Vol](https://huggingface.co/ddedei/Kronos-Vol), a next-day volatility model: every training job, its live
curves, its checkpoints, and every score against the baselines, computed from the saved forecasts.

It reads the files a training job writes and never touches the job. Read only, it binds to 127.0.0.1, and it needs no
GPU, no database and no build step: a Python standard-library server, pandas for the scores, and one page of vanilla
JavaScript with its own SVG charts.

<!-- screenshots to add: docs/experiments.png, docs/job.png, docs/compare.png, docs/final.png (python shots.py docs ...) -->

## Try it in 1 minute, no data needed

```
pip install -r requirements.txt
python make_demo.py                 # writes demo/ (about 8 MB of synthetic runs)
python server.py --root demo        # then open http://127.0.0.1:8765
```

Every number in the demo is simulated: 40 made-up stocks, 31 runs, a backtest, an agent test and 2 LLM fine-tunes.
Each page says so in the sidebar. `python make_demo.py --live` also plays a training job for about 3 minutes, so the Jobs
page and the live charts move.

## Point it at your runs

```
python server.py --root path/to/kronos_vol          # the folder that holds runs/ and data/
python server.py --runs path/to/kronos_vol/runs     # the same: the root is the folder above runs/
```

| Flag | Environment variable | Default |
|---|---|---|
| `--root` | `LOOKSFINE_ROOT` | the current folder |
| `--runs` | `LOOKSFINE_RUNS` | `ROOT/runs` |
| `--data` | `LOOKSFINE_DATA` | `ROOT/data` |
| `--final` | `LOOKSFINE_FINAL` | `ROOT/final_tests` |
| `--llm` | `LOOKSFINE_LLM` | `ROOT/llm` |
| `--takes` | `LOOKSFINE_TAKES` | `ROOT/rec/takes` |
| `--port` | `LOOKSFINE_PORT` | 8765 |

Only the runs folder and the data panel are required. A page whose folder is missing says it has nothing to show.

## Pages

**Experiments**
- Headline: the fine-tune's calibrated QLIKE on stocks it never saw, against HAR-RV, the control and random weights.
- A release check: 8 criteria for publishing a model, each computed from the runs and green only on the numbers.
- Data efficiency: pretrained against random weights as the number of training stocks grows.
- Recipe search: each training variant (loss, pooling, context length) against the default recipe, same seeds, same rows,
  with a Diebold-Mariano t.
- 1 table per experiment: seeds averaged, t against the control and against HAR-RV, held-out stocks apart.

**Jobs**
- Live jobs with a progress bar and the time left, and the queue read from `queue_*.sh` files.
- Every run, sortable and filterable. Tick runs to compare them.
- A job page with 6 tabs:
  - Overview: loss month by month against HAR-RV and the control, calibration bins, the same TEST rows for every model,
    seen against held-out stocks, each year.
  - Training: the training loss every 25 steps, raw and smoothed, and the validation curves. A running job draws them live.
  - Checkpoints: the 12 validation checks and the one kept.
  - Stocks: loss stock by stock against HAR-RV, and forecast against realised day by day.
  - Configuration and Logs.

**Compare**
- Any set of runs or seed groups, scored on the rows they all share, against a reference you pick.
- All, seen or held-out stocks. Diebold-Mariano t on raw QLIKE, calibrated QLIKE and squared error, each year and month.

**Final tests**
- A volatility-targeting backtest (growth of 1, 21-day volatility, Sharpe, drawdown, risk steadiness, turnover).
- Accuracy against simple rules, with the rank IC left once HAR-RV is removed.
- Where the edge comes from: the fine-tune minus the control in fifths of what the day looked like, and month by month.
- An LLM agent sizing positions with the real forecast, a fake one or none, and a forward ledger of logged forecasts.

**LLM fine-tunes**
- LoRA fine-tunes of small language models that pull numbers out of financial filings: accuracy tuned against out of
  the box, mistake kinds, training loss, validation curve and checkpoint table.

**Takes**
- Screen recordings (.mp4) found in the takes folder, if any.

## One ruler

Every score is computed in `server.py` from the saved forecasts, in 1 way for every model:
- QLIKE on variance, `y/f - log(y/f) - 1`.
- Calibrated QLIKE: each model's own best constant level shift is removed first, because raw QLIKE rewards forecasting high.
- R² on log variance and the cross-sectional rank IC, a daily mean.
- DM t: Diebold-Mariano on the daily cross-sectional mean loss difference, Newey-West errors, 5 lags. Below -2 is a real gain.

## What it reads

Written by `ft_intra.py` in the [Kronos-Vol code](https://github.com/davidalmeida90/yt-videos-guide/tree/main/kronos_vol);
any trainer that writes the same files works.

| File | Content |
|---|---|
| `runs/<run>/card.json` | `args` (mode, seed, loss, pool, tag, panel, holdout, train_stocks, epochs, ...), `history` (VAL checks: step, val_qlike, val_r2, val_cs_ic), `train_loss` ([step, loss] pairs), `best_step`, `minutes` |
| `runs/<run>/test_preds.parquet` | `date`, `symbol`, `pred` (log variance forecast), `holdout` (bool) |
| `runs/<run>/live.json` | while training: `step`, `steps`, `secs`, `train_loss`, `history`, `args` |
| `data/<panel>.parquet` | `date`, `symbol`, the target `lrv1` (next-day log realised variance) and the 9 features HAR fits use |
| `*.log` in the root | optional: a line `RUN: train N  val N  test N windows`, then `step i/n  Ss  val r2 x  qlike x  ic x` lines |

Runs are grouped by the `tag` argument (`_u200` for the main protocol, `_u200n20` and `_u200n60` for the data-efficiency
runs). A `families.json` in the root renames the groups; `make_demo.py` writes one.

## Files

| File | Role |
|---|---|
| `server.py` | HTTP server and scoring, cached by file modification time |
| `static/` | the page: `index.html`, `style.css`, `app.js` (dark theme with a light toggle) |
| `make_demo.py` | the synthetic demo folder |
| `shots.py` | optional: headless full-page screenshots with Playwright, no window opens |

## Credits

Built for the Kronos-Vol fine-tune of [Kronos](https://github.com/shiyu-coder/Kronos) (Shi et al., 2025, MIT).
MIT license, see [LICENSE](LICENSE).
