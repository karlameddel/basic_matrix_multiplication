/* Basic triple-loop (i-j-k) dense matrix multiplication, double precision.
 * Usage: matmul <A.bin> <B.bin> <n> <reps> <warmup> [out.bin]
 * Timed region: only the multiplication kernel (inputs loaded and C
 * allocated/zeroed outside the timer).
 * Output (stdout, CSV-like): rep,seconds,checksum
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double now_seconds(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static double *load(const char *path, size_t n) {
    double *m = malloc(n * n * sizeof(double));
    FILE *fp = fopen(path, "rb");
    if (!m || !fp) { fprintf(stderr, "cannot load %s\n", path); exit(1); }
    if (fread(m, sizeof(double), n * n, fp) != n * n) {
        fprintf(stderr, "short read %s\n", path); exit(1);
    }
    fclose(fp);
    return m;
}

static void matmul(const double *A, const double *B, double *C, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s A.bin B.bin n reps warmup [out.bin]\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atol(argv[3]);
    int reps = atoi(argv[4]), warmup = atoi(argv[5]);
    double *A = load(argv[1], n), *B = load(argv[2], n);
    double *C = calloc(n * n, sizeof(double));
    if (!C) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int r = -warmup; r < reps; r++) {
        memset(C, 0, n * n * sizeof(double));
        double t0 = now_seconds();
        matmul(A, B, C, n);
        double t1 = now_seconds();
        double checksum = 0.0;                /* consumes the result */
        for (size_t i = 0; i < n * n; i++) checksum += C[i];
        if (r >= 0) printf("%d,%.9f,%.12e\n", r, t1 - t0, checksum);
    }

    if (argc >= 7) {
        FILE *fp = fopen(argv[6], "wb");
        fwrite(C, sizeof(double), n * n, fp);
        fclose(fp);
    }
    free(A); free(B); free(C);
    return 0;
}