# Basic Matrix Multiplication in Python, Java and C

**Big Data · Assignment 1 · Academic year 2026-2027**
Grado en Ciencia e Ingeniería de Datos, Universidad de Las Palmas de Gran Canaria
Author: Karla Mederos Delgado

## Research question

> How does matrix size *n* affect the runtime and memory use of a basic triple-loop
> dense matrix multiplication implemented in Python, Java and C, under the same input
> matrices, numeric type and loop order, on a single machine?

The study compares three implementations of the straightforward O(n³) algorithm. It
does not claim that one language is universally faster: conclusions are limited to the
tested implementations, settings, inputs and hardware. Results, analysis and
limitations are in the PDF report (`report/`), not in this README.

## Repository layout

```
.
├── README.md
├── requirements.txt           # pinned Python dependencies (analysis and harness)
├── docs/
│   └── environment.md         # hardware, versions, flags, budgets, known limitations
├── data/
│   └── manifest.json          # seed, format and SHA-256 of every input file
│                              # (the .bin inputs are NOT versioned; regenerate them)
├── src/
│   ├── c/matmul.c
│   ├── java/MatMul.java
│   └── python/matmul.py
├── scripts/
│   ├── generate_inputs.py     # shared input matrices A and B
│   ├── verify_correctness.py  # correctness tests, run BEFORE benchmarking
│   ├── run_benchmarks.py      # measurement harness
│   └── analyze.py             # tables and figures from the raw data
├── results/
│   ├── raw_measurements.csv   # one row per measured repetition
│   ├── run_status.csv         # one row per (language, n): status and peak memory
│   ├── tables/
│   └── figures/
└── report/                    # PDF report (LaTeX source and PDF)
```

## Experimental design in brief

| Aspect | Setting |
|---|---|
| Algorithm | Triple loop, order i-j-k, `s += A[i][k] * B[k][j]`, no transposition, no blocking |
| Numeric type | `float64` (`double`) |
| Storage | Row-major. C and Java: one contiguous array. Python: list of lists of floats |
| Threads | One compute thread. No matrix-multiplication library in the timed region |
| Inputs | Binary files (little-endian float64, row-major, no header), uniform in [0, 1), generated once with seed 42. All three languages read the same files |
| Sizes | n ∈ {10, 50, 100, 200, 500, 1000, 2000, 4000} |
| Timer | Multiplication kernel only. Inputs are loaded, and C is allocated and zeroed, outside the timer. Monotonic clocks: `QueryPerformanceCounter` (C), `System.nanoTime` (Java), `time.perf_counter` (Python) |
| Result consumption | Every repetition computes a checksum (sum of all entries of C) |
| Repetitions | 5 measured repetitions per configuration, warm-up excluded (C 1, Java 3, Python 1). Costly runs (a single multiplication above 30 s) use 3 repetitions and no warm-up |
| Statistics | Median and interquartile range; minimum and maximum are also reported. All raw measurements are kept |
| Memory metric | Peak working set of the process tree (Windows), sampled every 10 ms with `psutil` in a separate probe run. Reported next to the theoretical array storage (24n² bytes); the two metrics are not equivalent |
| Budgets | 300 s per single run and 4 GB peak memory. Violations are recorded as results (`timeout`, `memory_exceeded`) without invented times. Larger sizes for that language are then skipped |
| CPU affinity | Processes are pinned to logical processors 2, 4, 6, 8 (hybrid CPU) |

Full hardware and software details, compiler and JVM flags, and known limitations
are in [`docs/environment.md`](docs/environment.md).

## Requirements

- Windows 11 (the harness reads Windows memory counters and the C code uses the
  Windows timer API; other systems need small changes)
- Python 3.11 with the packages in `requirements.txt`
- JDK 21 (`javac` and `java` on the `PATH`)
- GCC (developed with MSYS2 UCRT64, gcc 16.2.0)
- A laptop must be plugged in and set to its best performance mode. Close other
  applications and do not use the machine during the benchmark.

## How to reproduce

All commands are run from the repository root in PowerShell. Replace `python` with the
interpreter you want to measure with (the study used CPython 3.11.9; the harness
launches the Python implementation with the interpreter that runs it, `sys.executable`).

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Generate the shared inputs

```powershell
python scripts/generate_inputs.py
```

This writes `A_n<n>.bin` and `B_n<n>.bin` to `data/` (about 0.5 GB in total) and updates
`data/manifest.json`. Existing files are skipped unless `--force` is passed. To check
that you obtained exactly the same matrices, compare the SHA-256 hashes in the manifest.
The inputs depend on the NumPy version recorded in the manifest.

### 3. Build

```powershell
mkdir build
gcc -O2 -o build/matmul_c.exe src/c/matmul.c
javac -d build src/java/MatMul.java
```

Do not build the measured C binary with an IDE default (for example `-g` without
`-O2`): the optimisation level is part of the experiment. No `-march=native` and
no `-ffast-math`, so the compiler does not change the arithmetic.

### 4. Run a single implementation by hand (optional)

All three programs share the same interface: `A.bin B.bin n reps warmup [out.bin]`.
They print one line per measured repetition: `rep,seconds,checksum`.
The optional `out.bin` makes them dump the full result matrix (used for validation).

```powershell
.\build\matmul_c.exe data\A_n100.bin data\B_n100.bin 100 5 1
java -Xmx4g -cp build MatMul data\A_n100.bin data\B_n100.bin 100 5 3
python src/python/matmul.py data\A_n100.bin data\B_n100.bin 100 5 1
```

### 5. Verify correctness (before benchmarking)

```powershell
python scripts/verify_correctness.py
```

Checks, for n ∈ {10, 50, 100, 200}: a hand-computed 2×2 case, A·I = A, I·B = B, A·0 = 0,
and the random inputs against NumPy (`A @ B`, used only as a reference, never in a
measurement). Tolerance: `|x − ref| ≤ 1e-12 + 1e-9·|ref|`. The script writes
`results/tables/correctness.csv` and exits with code 1 if any test fails.

### 6. Run the benchmarks

```powershell
python scripts/run_benchmarks.py
```

For each size and language the harness:

1. Runs a **probe**: one multiplication in its own process, measuring the time of that
   multiplication and the peak memory (this is the separate memory run).
2. Decides the number of repetitions (see the table above).
3. Runs the **timing run**: warm-up repetitions (excluded) plus the measured repetitions,
   inside one process.
4. Appends every measured repetition to `results/raw_measurements.csv` and the status of
   the configuration to `results/run_status.csv`.
5. Compares the checksum between repetitions and between languages and warns on any
   mismatch.

The harness is **resumable**: configurations already in `run_status.csv` are skipped, so
an interrupted run can be restarted with the same command. To measure again, move or
delete the two CSV files.

Useful options:

```powershell
python scripts/run_benchmarks.py --sizes 10 50 100 --out-dir results/test   # quick trial
python scripts/run_benchmarks.py --langs C Java --sizes 500 1000           # subset
python scripts/run_benchmarks.py --reps 10 --cpus 2 4 6 8                  # override defaults
```

`--cpus` takes logical processor numbers. The default (2 4 6 8) assumes these belong to
performance cores on the Intel Core Ultra 9 185H. Check the topology of your own CPU and
adjust it. Use `--out-dir` for trials so the real results are not mixed with test data.

### 7. Analyse

```powershell
python scripts/analyze.py
```

Reads the two CSV files and writes to `results/tables/` and `results/figures/`:
time summaries (median, quartiles, IQR, min, max), growth ratios between consecutive
sizes, memory summary, size limits, and the figures (time against n on log-log axes with an n³
reference, time normalised by n³, and memory against the array-storage estimate).

## Output file formats

`results/raw_measurements.csv`: one row per measured repetition.

| Column | Meaning |
|---|---|
| `timestamp` | When the configuration was run (ISO 8601) |
| `language` | `C`, `Java` or `Python` |
| `n` | Side length of the n × n matrices |
| `numeric_type` | `float64` |
| `threads` | Number of compute threads (1) |
| `timer` | Timing boundary (`kernel_only`) |
| `rep` | Repetition index (0-based, warm-up not included) |
| `seconds` | Kernel time of one multiplication in seconds |
| `checksum` | Sum of all entries of C |
| `warmup` | Number of excluded warm-up repetitions |
| `reps_planned` | Number of measured repetitions requested |
| `affinity` | Logical processors allowed for the process |

`results/run_status.csv`: one row per (language, n), with `status` (`ok`, `timeout`,
`memory_exceeded`, `skipped_after_failure`, ...), probe time, peak memory in bytes, the
memory metric, the array-storage estimate and the limits used.

## Reproducibility notes

- Absolute times depend on the machine, the power mode, the thermal state and the background
  load. Expect the same trends, not the same numbers.
- Do not compare results from different machines as a pure comparison of languages.
- The hardware, versions and flags used for the reported results are in
  `docs/environment.md`.

## Use of generative AI

Generative AI tools were used for explanation, planning, debugging and drafting parts of the
code and documentation, in line with the course policy. 

## License

See [`LICENSE`](LICENSE).