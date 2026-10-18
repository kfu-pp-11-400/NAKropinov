#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};

void ijk_seq(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

void ijk_omp(const double *A, const double *B, double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

void ikj_seq(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++)
                C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
}

void ikj_omp(const double *A, const double *B, double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++)
                C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
}

void blocked_omp(const double *A, const double *B, double *C, int n, int b) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b)
            for (int kk = 0; kk < n; kk += b)
                for (int i = ii; i < ii + b && i < n; i++)
                    for (int k = kk; k < kk + b && k < n; k++) {
                        double a = A[(size_t)i * n + k];
                        for (int j = jj; j < jj + b && j < n; j++)
                            C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
                    }
}

void multiply(int mode, const double *A, const double *B, double *C, int n, int b) {
    switch (mode) {
        case 0: ijk_seq(A, B, C, n); break;
        case 1: ijk_omp(A, B, C, n); break;
        case 2: ikj_seq(A, B, C, n); break;
        case 3: ikj_omp(A, B, C, n); break;
        case 4: blocked_omp(A, B, C, n, b); break;
    }
}

void fill(double *A, double *B, size_t count) {
    for (size_t i = 0; i < count; i++) {
        A[i] = ((int)(i % 17) - 8) / 10.0;
        B[i] = ((int)((i * 7) % 19) - 9) / 10.0;
    }
}

double benchmark(int mode, const double *A, const double *B, double *C,
                 const double *reference, int n, int b, int threads, int *ok) {
    size_t count = (size_t)n * n;
    size_t bytes = count * sizeof(double);
    double times[3];
    int repeats = n <= 128 ? 200 : n <= 256 ? 30 : n <= 512 ? 3 : 1;
    omp_set_num_threads(threads);

    for (int trial = 0; trial < 3; trial++) {
        double elapsed = 0.0;
        for (int r = 0; r < repeats; r++) {
            memset(C, 0, bytes);
            double start = omp_get_wtime();
            multiply(mode, A, B, C, n, b);
            elapsed += omp_get_wtime() - start;
        }
        times[trial] = elapsed / repeats;
        for (size_t i = 0; i < count; i++)
            if (fabs(C[i] - reference[i]) > 1e-8 * (1.0 + fabs(reference[i]))) {
                *ok = 0;
                return 0.0;
            }
    }

    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++)
            if (times[i] > times[j]) {
                double t = times[i];
                times[i] = times[j];
                times[j] = t;
            }
    *ok = 1;
    return times[1];
}
int main(int argc, char *argv[]) {
    if (argc != 5) return 1;
    int n = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int mode = -1;
    for (int i = 0; i < 5; i++)
        if (strcmp(argv[4], names[i]) == 0) mode = i;
    if (n <= 0 || b <= 0 || threads <= 0 || mode < 0) return 1;

    size_t count = (size_t)n * n;
    size_t bytes = count * sizeof(double);
    double *A = malloc(bytes), *B = malloc(bytes);
    double *C = malloc(bytes), *reference = malloc(bytes);
    if (!A || !B || !C || !reference) return 1;

    fill(A, B, count);
    ijk_seq(A, B, reference, n);
    omp_set_dynamic(0);
    int ok;
    double time = benchmark(mode, A, B, C, reference, n, b, threads, &ok);
    printf("N=%d b=%d threads=%d version=%s time=%.9f s check=%s\n",
           n, b, threads, names[mode], time, ok ? "OK" : "FAIL");

    free(A);
    free(B);
    free(C);
    free(reference);
    return ok ? 0 : 1;
}
