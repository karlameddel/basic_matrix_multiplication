#!/usr/bin/env python3
"""Correctness validation for Assignment 1 (run BEFORE benchmarking).

Checks the C, Java and Python triple-loop implementations against:
  1. A hand-computed 2x2 case with a known exact answer.
  2. Identity and zero matrices (A*I = A, I*B = B, A*0 = 0).
  3. NumPy (A @ B) as the trusted reference, on the shared random inputs.

Tolerance (documented in the report):
  |x - ref| <= ATOL + RTOL * |ref|   with RTOL = 1e-9, ATOL = 1e-12.
Justification: inputs are in [0, 1), so all products are non-negative and
there is no cancellation. The relative error of a recursive sum of n terms is
bounded by about n * u (u = 2^-53 ~ 1.1e-16): ~2e-14 for n = 200 and
~4.4e-13 even for n = 4000. RTOL = 1e-9 leaves several orders of magnitude of
margin, while still catching real bugs (wrong indexing gives O(1) errors).
ATOL only matters for entries whose exact value is 0 (zero-matrix test).
Different evaluation orders (NumPy uses BLAS) explain tiny differences.

Usage (from the repo root):
  python scripts/verify_correctness.py [--sizes 10 50 100 200]
Writes results/tables/correctness.csv and exits with code 1 on any failure.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_CSV = ROOT / "results" / "tables" / "correctness.csv"

RTOL = 1e-9
ATOL = 1e-12
TIMEOUT_S = 300

IMPLEMENTATIONS = {
    "C": [str(ROOT / "build" / "matmul_c.exe")],
    "Java": ["java", "-cp", str(ROOT / "build"), "MatMul"],
    "Python": [sys.executable, str(ROOT / "src" / "python" / "matmul.py")],
}


def write_bin(path: Path, m: np.ndarray) -> None:
    np.ascontiguousarray(m, dtype="<f8").tofile(path)


def run_impl(cmd, a_path, b_path, n, out_path):
    """Run one implementation (1 rep, no warm-up). Return (C matrix, checksum)."""
    full = cmd + [str(a_path), str(b_path), str(n), "1", "0", str(out_path)]
    res = subprocess.run(full, capture_output=True, text=True, timeout=TIMEOUT_S)
    if res.returncode != 0:
        raise RuntimeError(f"exit code {res.returncode}: {res.stderr.strip()}")
    checksum = float(res.stdout.strip().splitlines()[-1].split(",")[2])
    c = np.fromfile(out_path, dtype="<f8")
    if c.size != n * n:
        raise RuntimeError(f"output has {c.size} values, expected {n * n}")
    return c.reshape(n, n), checksum


def compare(c, ref, checksum):
    """Return (max_abs_err, max_rel_err, passed)."""
    abs_err = np.abs(c - ref)
    nz = ref != 0
    rel_err = abs_err[nz] / np.abs(ref[nz]) if nz.any() else np.array([0.0])
    ok_matrix = bool(np.all(abs_err <= ATOL + RTOL * np.abs(ref)))
    ref_sum = float(ref.sum())
    ok_checksum = abs(checksum - ref_sum) <= ATOL + RTOL * abs(ref_sum)
    return float(abs_err.max()), float(rel_err.max()), ok_matrix and ok_checksum


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 200])
    args = ap.parse_args()

    for name, cmd in IMPLEMENTATIONS.items():
        exe = Path(cmd[0])
        if name == "C" and not exe.exists():
            sys.exit(f"missing {exe}: build it first")
    if not (ROOT / "build" / "MatMul.class").exists():
        sys.exit("missing build/MatMul.class: compile the Java source first")

    # (case name, n, A, B, expected C, file paths or None to write temp files)
    cases = []
    A2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    B2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    cases.append(("known_2x2", 2, A2, B2, np.array([[19.0, 22.0], [43.0, 50.0]])))

    for n in args.sizes:
        A = np.fromfile(DATA / f"A_n{n}.bin", dtype="<f8").reshape(n, n)
        B = np.fromfile(DATA / f"B_n{n}.bin", dtype="<f8").reshape(n, n)
        I = np.eye(n)
        Z = np.zeros((n, n))
        cases.append(("A_times_identity", n, A, I, A))
        cases.append(("identity_times_B", n, I, B, B))
        cases.append(("A_times_zero", n, A, Z, Z))
        cases.append(("random_vs_numpy", n, A, B, A @ B))  # reference: NumPy

    rows, all_ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case, n, A, B, expected in cases:
            a_path, b_path, out_path = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
            write_bin(a_path, A)
            write_bin(b_path, B)
            for lang, cmd in IMPLEMENTATIONS.items():
                try:
                    c, checksum = run_impl(cmd, a_path, b_path, n, out_path)
                    abs_e, rel_e, ok = compare(c, expected, checksum)
                except Exception as e:  # report failures, do not hide them
                    abs_e = rel_e = float("nan")
                    ok = False
                    print(f"  ERROR {lang}/{case}/n={n}: {e}")
                all_ok &= ok
                rows.append([case, n, lang, f"{abs_e:.3e}", f"{rel_e:.3e}",
                             RTOL, ATOL, "PASS" if ok else "FAIL"])
                print(f"[{'PASS' if ok else 'FAIL'}] {case:18s} n={n:<4d} {lang:7s}"
                      f" max_abs={abs_e:.2e} max_rel={rel_e:.2e}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n", "language", "max_abs_error", "max_rel_error",
                    "rtol", "atol", "result"])
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()