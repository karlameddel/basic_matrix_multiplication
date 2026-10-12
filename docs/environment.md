# Experimental environment

| Component | Value |
|---|---|
| CPU | Intel Core Ultra 9 185H (hybrid architecture) |
| Physical cores / logical processors | 16 / 22 |
| RAM | 31.7 GB |
| OS | Windows 11 Home, version 10.0.26200 (build 26200) |
| Python | 3.11.9 (CPython, Microsoft Store distribution) |
| NumPy / pandas / matplotlib / psutil | 2.2.6 / 3.0.0 / 3.10.6 / 7.0.0 |
| Java | OpenJDK 21.0.6 (Temurin-21.0.6+7), default JIT (C2) |
| C compiler | gcc 16.2.0 (MSYS2) |
| Power / thermal | Laptop running on battery (not plugged in) during all measurements. Battery level and temperature not recorded. |

## Fixed settings
- Numeric type: float64 (`double`)
- Loop order: i-j-k, row-major storage in all three languages
- Single-threaded kernel; no matrix-multiplication library in the timed region
  (NumPy is used only to read input files and as correctness reference)
- Input seed: 42 (see `data/manifest.json`)
- Timer: multiplication kernel only (inputs loaded and C allocated/zeroed outside it)

## Compilation and launch commands
- C: `gcc -O2 -o build/matmul_c.exe src/c/matmul.c` (no `-march=native`, no `-ffast-math`)
- Java: `javac -d build src/java/MatMul.java`; run with `java -Xmx4g -cp build MatMul ...`
  (default JVM flags otherwise; max heap 4 GB matches the memory budget)
- Python: `python3.11.exe src/python/matmul.py ...` (CPython 3.11.9, lists of lists of floats)

## Warm-up and repetitions
- Warm-up (excluded from results): C 1, Java 3, Python 1
- Measured repetitions: 5 per configuration
- Costly runs (probe > 30 s, Python n = 1000): 3 repetitions, no warm-up

## CPU affinity
Every benchmark process is pinned with psutil to logical processors 2, 4, 6, 8
(assumed to belong to four different performance cores; not verified against the
topology). The multiplication is single-threaded; several processors are allowed
so that JVM background threads (JIT, GC) do not compete with the compute thread.

## Budgets
- Time: 300 s per single run (otherwise recorded as timeout)
- Memory: 4 GB peak per run
- Memory metric: peak working set (Windows), summed over the process tree,
  polled every 10 ms by psutil in a separate probe run

## Known limitations
- C processes at n <= 100 finish before the 10 ms sampler reads memory (recorded as 0,
  not a valid measurement).
- Python n = 100 median in the harness (~106 ms) is about twice the value of a manual
  run (~48 ms); cause not investigated.
- Java n = 2000 median is higher than its cold probe run; thermal effects or noise are possible.
- C and Java at n = 10 take ~1 us to ~20 us, close to timer resolution.
- Peak memory of Java and Python includes the runtime and is not comparable with the
  array-storage estimate (24 n^2 bytes).