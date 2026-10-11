#!/usr/bin/env python3
"""Analysis for Assignment 1: tables and figures from the raw measurements.

Inputs : results/raw_measurements.csv  (one row per measured repetition)
         results/run_status.csv        (one row per language x n: status, memory)
Outputs: results/tables/summary_time.csv    median, Q1, Q3, IQR, min, max, reps
         results/tables/summary_memory.csv  peak working set vs array estimate
         results/tables/growth_ratios.csv   time ratio between consecutive n
         results/tables/limits.csv          largest feasible n, timeouts
         results/figures/time_vs_n.png      log-log, with n^3 reference
         results/figures/time_normalised.png  time / n^3 (constant if cubic)
         results/figures/memory_vs_n.png    peak memory vs array-storage estimate

Statistics: median and IQR (Q1 = 25th, Q3 = 75th percentile, linear
interpolation). With 3 repetitions (costly runs) the IQR is weak, so min and
max are reported as well.

Usage (repo root):  python scripts/analyze.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TABLES, FIGS = RES / "tables", RES / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

RAW_HEADER = ["timestamp", "language", "n", "numeric_type", "threads", "timer",
              "rep", "seconds", "checksum", "warmup", "reps_planned", "affinity"]
LANGS = ["C", "Java", "Python"]
STYLE = {"C": ("tab:blue", "o"), "Java": ("tab:orange", "s"), "Python": ("tab:green", "^")}


def load_raw():
    path = RES / "raw_measurements.csv"
    with open(path) as f:
        first = f.readline()
    has_header = first.startswith("timestamp")
    return pd.read_csv(path, header=0 if has_header else None,
                       names=None if has_header else RAW_HEADER)


def summarise(raw):
    g = raw.groupby(["language", "n"])["seconds"]
    s = g.agg(reps="count", median="median", min="min", max="max")
    s["q1"], s["q3"] = g.quantile(0.25), g.quantile(0.75)
    s["iqr"] = s["q3"] - s["q1"]
    s["iqr_over_median_pct"] = 100 * s["iqr"] / s["median"]
    return s.reset_index()


def growth(summary):
    rows = []
    for lang, d in summary.groupby("language"):
        d = d.sort_values("n")
        for (n0, t0), (n1, t1) in zip(d[["n", "median"]].values, d[["n", "median"]].values[1:]):
            rows.append([lang, int(n0), int(n1), t1 / t0, (n1 / n0) ** 3])
    return pd.DataFrame(rows, columns=["language", "n_from", "n_to",
                                       "time_ratio", "cubic_expected_ratio"])


def fig_time(summary):
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        d = summary[summary.language == lang].sort_values("n")
        c, m = STYLE[lang]
        err = [d["median"] - d["q1"], d["q3"] - d["median"]]
        ax.errorbar(d["n"], d["median"], yerr=err, color=c, marker=m, capsize=3, label=lang)
    # n^3 reference anchored at the C median for n = 1000
    ref = summary[(summary.language == "C") & (summary.n == 1000)]["median"].iloc[0]
    xs = np.array([10, 4000])
    ax.plot(xs, ref * (xs / 1000) ** 3, "k--", lw=1, label=r"$\propto n^3$ (reference)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n, float64)")
    ax.set_ylabel("Kernel time per multiplication (s)")
    ax.set_title("Runtime vs matrix size (median, bars = IQR)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "time_vs_n.png", dpi=200); plt.close(fig)


def fig_normalised(summary):
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        d = summary[summary.language == lang].sort_values("n")
        c, m = STYLE[lang]
        ax.plot(d["n"], d["median"] / d["n"] ** 3 * 1e9, color=c, marker=m, label=lang)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Median time / n$^3$ (ns per inner iteration)")
    ax.set_title("Cost per multiply-add (flat line = pure cubic growth)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "time_normalised.png", dpi=200); plt.close(fig)


def memory_table(status):
    ok = status[status.status.isin(["ok", "rep_over_time_limit"])].copy()
    ok["peak_MB"] = ok["peak_memory_bytes"] / 1e6
    ok["array_estimate_MB"] = ok["array_storage_estimate_bytes"] / 1e6
    # C at n <= 100 exited before the 10 ms sampler could read memory: not a measurement
    bad = (ok.language == "C") & (ok.peak_memory_bytes == 0)
    ok.loc[bad, "peak_MB"] = np.nan
    ok["note"] = np.where(bad, "not captured (process too short for 10 ms sampling)", "")
    return ok[["language", "n", "peak_MB", "array_estimate_MB", "note"]]


def fig_memory(mem, status):
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        d = mem[mem.language == lang].dropna(subset=["peak_MB"]).sort_values("n")
        c, m = STYLE[lang]
        ax.plot(d["n"], d["peak_MB"], color=c, marker=m, label=f"{lang} peak working set")
    est = mem[mem.language == "C"].sort_values("n")
    ax.plot(est["n"], est["array_estimate_MB"], "k--", label=r"3 arrays estimate ($24n^2$ B)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Matrix size n"); ax.set_ylabel("Memory (MB)")
    ax.set_title("Peak process memory vs array-storage estimate")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "memory_vs_n.png", dpi=200); plt.close(fig)


def limits(status, summary):
    rows = []
    for lang in LANGS:
        d = status[status.language == lang]
        done = summary[summary.language == lang]["n"]
        fails = d[~d.status.isin(["ok", "rep_over_time_limit"])]
        rows.append([lang, int(done.max()), int(d.n.max()),
                     "; ".join(f"n={r.n}:{r.status}" for r in fails.itertuples())])
    return pd.DataFrame(rows, columns=["language", "largest_completed_n",
                                       "largest_attempted_n", "failures"])


def main():
    raw = load_raw()
    status = pd.read_csv(RES / "run_status.csv")
    summary = summarise(raw)
    mem = memory_table(status)

    summary.to_csv(TABLES / "summary_time.csv", index=False, float_format="%.9g")
    mem.to_csv(TABLES / "summary_memory.csv", index=False, float_format="%.3f")
    growth(summary).to_csv(TABLES / "growth_ratios.csv", index=False, float_format="%.3f")
    limits(status, summary).to_csv(TABLES / "limits.csv", index=False)
    fig_time(summary); fig_normalised(summary); fig_memory(mem, status)

    print(summary[["language", "n", "reps", "median", "iqr", "min", "max"]].to_string(index=False))
    print("\n", limits(status, summary).to_string(index=False))
    print(f"\nTables -> {TABLES}\nFigures -> {FIGS}")


if __name__ == "__main__":
    main()