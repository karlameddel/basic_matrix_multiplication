#!/usr/bin/env python3
"""Benchmark harness for Assignment 1 (C, Java, Python triple-loop matmul).

Protocol (documented in the report):
  * Every (language, n) runs in its own process; single thread.
  * Probe run (1 rep, no warm-up) -> time of one multiplication AND peak
    memory (psutil polling every 10 ms; Windows peak working set, summed
    over the process tree). This is the separate memory run.
  * Timing run: `warmup` excluded repetitions + `reps` measured repetitions
    inside one process. Timer = kernel only (done inside each program).
  * If the probe takes > COSTLY_S seconds, use 3 reps and no warm-up
    (costly run, justified in the report).
  * Limits: 300 s per run, 4 GiB peak memory. Violations are recorded as
    results (timeout / memory_exceeded); larger n for that language is skipped.
  * Processes are pinned to a set of performance-core logical processors.

Outputs (append mode, resumable):
  results/raw_measurements.csv   one row per measured repetition
  results/run_status.csv         one row per (language, n): status + peak memory

Usage (repo root):
  python scripts/run_benchmarks.py
  python scripts/run_benchmarks.py --sizes 10 50 100 --out-dir results/test
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BUILD = ROOT / "build"

TIME_LIMIT_S = 300
MEM_LIMIT_BYTES = 4 * 1024 ** 3
COSTLY_S = 30
DEFAULT_SIZES = [10, 50, 100, 200, 500, 1000, 2000, 4000]
DEFAULT_CPUS = [2, 4, 6, 8]  # first logical processor of 4 different P-cores (verify!)

LANGS = {
    "C": {"cmd": [str(BUILD / "matmul_c.exe")], "warmup": 1},
    "Java": {"cmd": ["java", "-Xmx4g", "-cp", str(BUILD), "MatMul"], "warmup": 3},
    "Python": {"cmd": [sys.executable, str(ROOT / "src" / "python" / "matmul.py")], "warmup": 1},
}

RAW_HEADER = ["timestamp", "language", "n", "numeric_type", "threads", "timer",
              "rep", "seconds", "checksum", "warmup", "reps_planned", "affinity"]
STATUS_HEADER = ["timestamp", "language", "n", "status", "probe_seconds",
                 "peak_memory_bytes", "memory_metric", "array_storage_estimate_bytes",
                 "reps", "warmup", "time_limit_s", "mem_limit_bytes", "affinity"]


def sample_memory(ps):
    """Sum of peak working set (Windows) / RSS over the process tree."""
    total = 0
    try:
        procs = [ps] + ps.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            mi = p.memory_info()
            total += getattr(mi, "peak_wset", mi.rss)
        except psutil.Error:
            pass
    return total


def kill_tree(proc):
    try:
        for c in psutil.Process(proc.pid).children(recursive=True):
            c.kill()
    except psutil.Error:
        pass
    proc.kill()


def run_monitored(cmd, timeout_s, cpus):
    """Return (status, stdout, stderr, peak_bytes)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps = psutil.Process(proc.pid)
    try:
        ps.cpu_affinity(cpus)
    except psutil.Error as e:
        print(f"  WARNING: could not set affinity: {e}")
    start, peak, status = time.perf_counter(), 0, "ok"
    while proc.poll() is None:
        peak = max(peak, sample_memory(ps))
        if peak > MEM_LIMIT_BYTES:
            status = "memory_exceeded"
            break
        if time.perf_counter() - start > timeout_s:
            status = "timeout"
            break
        time.sleep(0.01)
    if status != "ok":
        kill_tree(proc)
    out, err = proc.communicate()
    if status == "ok" and proc.returncode != 0:
        status = "error"
    return status, out, err, peak


def parse(out):
    rows = []
    for line in out.strip().splitlines():
        r, s, c = line.split(",")
        rows.append((int(r), float(s), float(c)))
    return rows


def append(path, header, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def load_status(path):
    done, blocked = set(), set()
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["language"], int(r["n"])))
                if r["status"] != "ok":
                    blocked.add(r["language"])
    return done, blocked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--langs", nargs="+", default=list(LANGS), choices=list(LANGS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cpus", type=int, nargs="+", default=DEFAULT_CPUS)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, status_path = out_dir / "raw_measurements.csv", out_dir / "run_status.csv"
    done, blocked = load_status(status_path)
    aff = "+".join(map(str, args.cpus))
    checks = {}  # n -> {language: checksum}

    for n in args.sizes:
        a, b = DATA / f"A_n{n}.bin", DATA / f"B_n{n}.bin"
        for lang in args.langs:
            if (lang, n) in done:
                print(f"[skip] {lang} n={n} already in {status_path.name}")
                continue
            ts = datetime.now().isoformat(timespec="seconds")
            est = 3 * 8 * n * n
            if lang in blocked:
                append(status_path, STATUS_HEADER, [ts, lang, n, "skipped_after_failure", "", "",
                       "", est, "", "", TIME_LIMIT_S, MEM_LIMIT_BYTES, aff])
                print(f"[skip] {lang} n={n}: a smaller size already failed")
                continue

            base = LANGS[lang]["cmd"] + [str(a), str(b), str(n)]

            # 1) probe run: one multiplication + peak memory
            print(f"[probe] {lang} n={n} ...", flush=True)
            status, out, err, peak = run_monitored(base + ["1", "0"], TIME_LIMIT_S + 60, args.cpus)
            probe_s = parse(out)[0][1] if status == "ok" else ""
            if status == "ok" and probe_s > TIME_LIMIT_S:
                status = "timeout"
            if status == "error":
                sys.exit(f"{lang} n={n} failed: {err.strip()}")
            if status != "ok":
                append(status_path, STATUS_HEADER, [ts, lang, n, status, probe_s, peak,
                       "peak_working_set", est, 0, 0, TIME_LIMIT_S, MEM_LIMIT_BYTES, aff])
                blocked.add(lang)
                print(f"  -> {status} (peak {peak / 1e6:.0f} MB)")
                continue

            # 2) timing run
            if probe_s > COSTLY_S:
                reps, warmup = 3, 0
            else:
                reps, warmup = args.reps, LANGS[lang]["warmup"]
            budget = TIME_LIMIT_S * (reps + warmup) + 60
            print(f"  probe {probe_s:.3f}s, peak {peak / 1e6:.0f} MB -> "
                  f"{reps} reps + {warmup} warm-up", flush=True)
            st2, out, err, _ = run_monitored(base + [str(reps), str(warmup)], budget, args.cpus)
            if st2 != "ok":
                append(status_path, STATUS_HEADER, [ts, lang, n, st2 + "_timing_run", probe_s, peak,
                       "peak_working_set", est, reps, warmup, TIME_LIMIT_S, MEM_LIMIT_BYTES, aff])
                blocked.add(lang)
                print(f"  -> {st2} in timing run")
                continue

            rows = parse(out)
            for r, s, c in rows:
                append(raw_path, RAW_HEADER, [ts, lang, n, "float64", 1, "kernel_only",
                       r, f"{s:.9f}", f"{c:.12e}", warmup, reps, aff])
            sums = {c for _, _, c in rows}
            if len(sums) != 1:
                print(f"  WARNING: checksum changes between repetitions for {lang} n={n}")
            checks.setdefault(n, {})[lang] = rows[0][2]
            final = "ok" if max(s for _, s, _ in rows) <= TIME_LIMIT_S else "rep_over_time_limit"
            append(status_path, STATUS_HEADER, [ts, lang, n, final, probe_s, peak,
                   "peak_working_set", est, reps, warmup, TIME_LIMIT_S, MEM_LIMIT_BYTES, aff])
            times = sorted(s for _, s, _ in rows)
            print(f"  median {times[len(times) // 2]:.6f} s")

        vals = checks.get(n, {})
        if len(vals) > 1:
            ref = next(iter(vals.values()))
            if any(abs(v - ref) > 1e-9 * abs(ref) for v in vals.values()):
                print(f"  WARNING: checksums differ across languages at n={n}: {vals}")


if __name__ == "__main__":
    main()