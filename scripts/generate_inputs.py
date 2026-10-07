#!/usr/bin/env python3
"""
Generate the shared input matrices A and B

Format: raw little-endian float64, row-major (C order), no header.
The file for size n holds n*n values. All three languages read these files,
so every implementation multiplies exactly the same logical matrices.

Values: uniform in [0, 1). Seeds: numpy PCG64 seeded with [SEED, n, tag],
tag 0 -> A, tag 1 -> B.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 42
DEFAULT_SIZES = [10, 50, 100, 200, 500, 1000, 2000, 4000]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_matrix(n: int, tag: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng([seed, n, tag])
    return rng.random((n, n), dtype=np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    manifest["_meta"] = {
        "seed": args.seed,
        "dtype": "float64 little-endian",
        "layout": "row-major, no header",
        "distribution": "uniform [0, 1)",
        "generator": "numpy.random.default_rng (PCG64)",
        "numpy_version": np.__version__,
    }

    for n in args.sizes:
        for name, tag in (("A", 0), ("B", 1)):
            path = DATA_DIR / f"{name}_n{n}.bin"
            if path.exists() and not args.force:
                print(f"[skip] {path.name} exists")
            else:
                make_matrix(n, tag, args.seed).astype("<f8").tofile(path)
                print(f"[ok]   {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            manifest.setdefault(str(n), {})[name] = sha256(path)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()