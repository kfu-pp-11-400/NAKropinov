#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void transpose_rows(const float *A, float *B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_cols(const float *A, float *B, int n) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_blocked(const float *A, float *B, int n, int b) {
    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b)
            for (int i = ii; i < ii + b && i < n; i++)
                for (int j = jj; j < jj + b && j < n; j++)
                    B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_rows_omp(const float *A, float *B, int n, int chunk) {
    if (chunk == 1) {
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                B[(size_t)j * n + i] = A[(size_t)i * n + j];
    } else if (chunk == 8) {
        #pragma omp parallel for schedule(static, 8)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                B[(size_t)j * n + i] = A[(size_t)i * n + j];
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                B[(size_t)j * n + i] = A[(size_t)i * n + j];
    }
}

void transpose_blocked_omp(const float *A, float *B, int n, int b) {
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += b)
        for (int jj = 0; jj < n; jj += b)
            for (int i = ii; i < ii + b && i < n; i++)
                for (int j = jj; j < jj + b && j < n; j++)
                    B[(size_t)j * n + i] = A[(size_t)i * n + j];
}

void transpose_inplace(float *A, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            float temp = A[(size_t)i * n + j];
            A[(size_t)i * n + j] = A[(size_t)j * n + i];
            A[(size_t)j * n + i] = temp;
        }
}

void transpose_inplace_omp(float *A, int n) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            float temp = A[(size_t)i * n + j];
            A[(size_t)i * n + j] = A[(size_t)j * n + i];
            A[(size_t)j * n + i] = temp;
        }
}

void run(const char *version, float *A, float *B, int n, int b) {
    if (strcmp(version, "rows") == 0)
        transpose_rows(A, B, n);
    else if (strcmp(version, "cols") == 0)
        transpose_cols(A, B, n);
    else if (strcmp(version, "blocked") == 0)
        transpose_blocked(A, B, n, b);
    else if (strcmp(version, "rows_omp") == 0)
        transpose_rows_omp(A, B, n, 0);
    else if (strcmp(version, "blocked_omp") == 0)
        transpose_blocked_omp(A, B, n, b);
    else if (strcmp(version, "rows_omp_1") == 0)
        transpose_rows_omp(A, B, n, 1);
    else if (strcmp(version, "rows_omp_8") == 0)
        transpose_rows_omp(A, B, n, 8);
    else if (strcmp(version, "inplace") == 0)
        transpose_inplace(A, n);
    else if (strcmp(version, "inplace_omp") == 0)
        transpose_inplace_omp(A, n);
}

int main(int argc, char *argv[]) {
    if (argc != 5) return 1;

    int n = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    const char *version = argv[4];
    int inplace = strcmp(version, "inplace") == 0 || strcmp(version, "inplace_omp") == 0;

    if (n <= 0 || b <= 0 || threads <= 0) return 1;

    size_t count = (size_t)n * n;
    size_t bytes = count * sizeof(float);
    float *A = malloc(bytes);
    float *B = inplace ? NULL : malloc(bytes);
    float *reference = malloc(bytes);
    if (!A || (!inplace && !B) || !reference) return 1;

    for (size_t k = 0; k < count; k++) A[k] = (float)k;
    transpose_rows(A, reference, n);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    int repeats = (int)(20000000LL / (long long)n / n);
    if (repeats < 1) repeats = 1;
    if (inplace && repeats % 2 == 0) repeats++;

    double times[5];
    for (int trial = -1; trial < 5; trial++) {
        if (inplace)
            for (size_t k = 0; k < count; k++) A[k] = (float)k;

        int runs = trial == -1 ? 1 : repeats;
        double start = omp_get_wtime();
        for (int r = 0; r < runs; r++) run(version, A, B, n, b);
        double elapsed = omp_get_wtime() - start;

        float *result = inplace ? A : B;
        long long errors = 0;
        for (size_t k = 0; k < count; k++)
            if (result[k] != reference[k]) errors++;
        if (errors) {
            printf("check=FAIL mismatches=%lld\n", errors);
            return 1;
        }
        if (trial >= 0) times[trial] = elapsed / runs;
    }

    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (times[j] < times[i]) {
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }

    double median = times[2];
    double bw = 2.0 * bytes / median / 1e9;
    printf("N=%d b=%d threads=%d version=%s time=%.9f s BW=%.3f GB/s check=OK\n",
           n, b, threads, version, median, bw);

    free(A);
    free(B);
    free(reference);
    return 0;
}
