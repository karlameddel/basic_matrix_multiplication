# Experimental environment

| Component | Value |
|---|---|
| CPU | Intel Core Ultra 9 185H (hybrid architecture) |
| Physical cores / logical processors | 16 / 22 |
| RAM | 31.7 GB |
| OS | Windows 11 Home, version 10.0.26200 (build 26200) |
| Python | 3.11.x (CPython, Microsoft Store distribution) |
| Java | OpenJDK 21.0.6 (Temurin-21.0.6+7) |
| C compiler | gcc 16.2.0 (MSYS2) |
| Power / thermal | Laptop, plugged in, "High performance" power plan (TO CONFIRM) |
| Affinity | Single thread pinned to one performance core (TO CONFIRM in harness) |

## Fixed settings
- Numeric type: float64 (`double`)
- Loop order: i-j-k, row-major storage
- Single thread, no matrix-multiplication libraries in the timed region
- Input seed: 42 (see `data/manifest.json`)

## Budgets
- Time: 5 minutes per single run (otherwise recorded as timeout)
- Memory: 4 GB peak resident memory per run

## Pending (to fill in later)
- gcc optimisation flags, JVM flags and warm-up settings
- NumPy / psutil versions (output of `pip freeze`)