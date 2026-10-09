"""Basic triple-loop (i-j-k) dense matrix multiplication, pure Python.
Usage: python matmul.py A.bin B.bin n reps warmup [out.bin]
Timed region: kernel only. Output: rep,seconds,checksum
Storage: list of lists of floats. NumPy is used only to read the files."""
import sys
import time
import numpy as np


def load(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n).tolist()


def matmul(A, B, C, n):
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Ai[k] * B[k][j]
            Ci[j] = s


def main():
    n = int(sys.argv[3])
    reps, warmup = int(sys.argv[4]), int(sys.argv[5])
    A, B = load(sys.argv[1], n), load(sys.argv[2], n)
    C = [[0.0] * n for _ in range(n)]

    for r in range(-warmup, reps):
        for row in C:
            for j in range(n):
                row[j] = 0.0
        t0 = time.perf_counter()
        matmul(A, B, C, n)
        t1 = time.perf_counter()
        checksum = 0.0
        for row in C:
            for v in row:
                checksum += v
        if r >= 0:
            print(f"{r},{t1 - t0:.9f},{checksum:.12e}")

    if len(sys.argv) >= 7:
        np.array(C, dtype="<f8").tofile(sys.argv[6])


if __name__ == "__main__":
    main()