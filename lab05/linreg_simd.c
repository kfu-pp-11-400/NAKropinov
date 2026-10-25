#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

enum { SERIAL, OPENMP, SIMD, OPENMP_SIMD };

const char *names[] = {"serial", "openmp", "simd", "openmp_simd"};

typedef struct {
    int n, d;
    double *x, *y, *w, *ref, *partial;
    double ref_b, ref_loss;
} Problem;

void make_data(Problem *p, int n, int d, int threads) {
    p->n = n;
    p->d = d;
    p->x = malloc((size_t)n * d * sizeof(double));
    p->y = malloc((size_t)n * sizeof(double));
    p->w = calloc((size_t)d, sizeof(double));
    p->ref = malloc((size_t)d * sizeof(double));
    p->partial = calloc((size_t)threads * (d + 1), sizeof(double));
    if (!p->x || !p->y || !p->w || !p->ref || !p->partial) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    uint32_t state = 12345u;
    for (int i = 0; i < n; i++) {
        double target = 0.2;
        for (int j = 0; j < d; j++) {
            state = state * 1664525u + 1013904223u;
            double value = (double)(state >> 8) / 8388608.0 - 1.0;
            p->x[(size_t)i * d + j] = value;
            if (j % 8 == 0) target += (j % 16 ? -0.8 : 0.8) * value;
        }
        p->y[i] = target;
    }
}

void free_data(Problem *p) {
    free(p->x);
    free(p->y);
    free(p->w);
    free(p->ref);
    free(p->partial);
}

void train(Problem *p, int mode, int threads, int iterations,
           double eta, double lambda, double *bias) {
    int n = p->n, d = p->d;
    int parallel = mode == OPENMP || mode == OPENMP_SIMD;
    int simd = mode == SIMD || mode == OPENMP_SIMD;
    int workers = parallel ? threads : 1;
    size_t stride = (size_t)d + 1;
    memset(p->w, 0, (size_t)d * sizeof(double));
    *bias = 0.0;

    for (int k = 0; k < iterations; k++) {
        memset(p->partial, 0, (size_t)workers * stride * sizeof(double));
        if (parallel) {
            // Каждый поток накапливает собственный градиент в отдельном буфере, чтобы исключить гонки данных.
            #pragma omp parallel num_threads(threads)
            {
                double *g = p->partial + (size_t)omp_get_thread_num() * stride;
                #pragma omp for schedule(static)
                for (int i = 0; i < n; i++) {
                    const double *row = p->x + (size_t)i * d;
                    double prediction = 0.0;
                    if (simd) {
                        // Директива omp simd векторизует скалярное произведение по признакам и использует reduction для суммы prediction.
                        #pragma omp simd reduction(+:prediction)
                        for (int j = 0; j < d; j++) prediction += row[j] * p->w[j];
                    } else {
                        for (int j = 0; j < d; j++) prediction += row[j] * p->w[j];
                    }
                    double error = prediction + *bias - p->y[i];
                    if (simd) {
                        // Независимые компоненты градиента по признакам удобно векторизовать, так как элементы строки лежат подряд в памяти.
                        #pragma omp simd
                        for (int j = 0; j < d; j++) g[j] += error * row[j];
                    } else {
                        for (int j = 0; j < d; j++) g[j] += error * row[j];
                    }
                    g[d] += error;
                }
            }
        } else {
            double *g = p->partial;
            for (int i = 0; i < n; i++) {
                const double *row = p->x + (size_t)i * d;
                double prediction = 0.0;
                if (simd) {
                    // В последовательной SIMD-версии тот же цикл по признакам векторизуется без распараллеливания по объектам.
                    #pragma omp simd reduction(+:prediction)
                    for (int j = 0; j < d; j++) prediction += row[j] * p->w[j];
                } else {
                    for (int j = 0; j < d; j++) prediction += row[j] * p->w[j];
                }
                double error = prediction + *bias - p->y[i];
                if (simd) {
                    #pragma omp simd
                    for (int j = 0; j < d; j++) g[j] += error * row[j];
                } else {
                    for (int j = 0; j < d; j++) g[j] += error * row[j];
                }
                g[d] += error;
            }
        }
        // После прохода по объектам частичные градиенты всех потоков объединяются перед проксимальным обновлением коэффициентов.
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            for (int t = 0; t < workers; t++) sum += p->partial[(size_t)t * stride + j];
            double z = p->w[j] - eta * sum / n;
            double threshold = eta * lambda;
            p->w[j] = z > threshold ? z - threshold :
                      z < -threshold ? z + threshold : 0.0;
        }
        double gb = 0.0;
        for (int t = 0; t < workers; t++) gb += p->partial[(size_t)t * stride + d];
        *bias -= eta * gb / n;
    }
}

double loss(const Problem *p, double bias, double lambda) {
    double value = 0.0;
    for (int i = 0; i < p->n; i++) {
        const double *row = p->x + (size_t)i * p->d;
        double prediction = bias;
        for (int j = 0; j < p->d; j++) prediction += row[j] * p->w[j];
        double e = prediction - p->y[i];
        value += e * e;
    }
    value /= 2.0 * p->n;
    for (int j = 0; j < p->d; j++) value += lambda * fabs(p->w[j]);
    return value;
}

void reference(Problem *p, int iterations, double eta, double lambda) {
    double bias;
    train(p, SERIAL, 1, iterations, eta, lambda, &bias);
    memcpy(p->ref, p->w, (size_t)p->d * sizeof(double));
    p->ref_b = bias;
    p->ref_loss = loss(p, bias, lambda);
}

double measure(Problem *p, int mode, int threads, int iterations,
               double eta, double lambda, double *result_loss,
               double *difference, int *zero_count, int *ok) {
    double times[3], bias = 0.0;
    omp_set_dynamic(0);
    for (int r = 0; r < 3; r++) {
        double start = omp_get_wtime();
        train(p, mode, threads, iterations, eta, lambda, &bias);
        times[r] = omp_get_wtime() - start;
    }
    if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; }
    if (times[1] > times[2]) { double t = times[1]; times[1] = times[2]; times[2] = t; }
    if (times[0] > times[1]) { double t = times[0]; times[0] = times[1]; times[1] = t; }

    *result_loss = loss(p, bias, lambda);
    *difference = 0.0;
    *zero_count = 0;
    for (int j = 0; j < p->d; j++) {
        double delta = fabs(p->w[j] - p->ref[j]);
        if (delta > *difference) *difference = delta;
        if (fabs(p->w[j]) < 1e-6) (*zero_count)++;
    }
    *ok = *difference < 1e-6 && fabs(bias - p->ref_b) < 1e-6 &&
          fabs(*result_loss - p->ref_loss) < 1e-6;
    return times[1];
}

void print_case(const char *experiment, Problem *p, int mode, int threads,
                int iterations, double eta, double lambda) {
    double final_loss, difference;
    int zeros, ok;
    double seconds = measure(p, mode, threads, iterations, eta, lambda,
                             &final_loss, &difference, &zeros, &ok);
    printf("%s N=%d d=%d K=%d lambda=%.5g eta=%.5g threads=%d mode=%s time=%.9f loss=%.9f dw=%.3e zeros=%d check=%s\n",
           experiment, p->n, p->d, iterations, lambda, eta,
           (mode == SERIAL || mode == SIMD) ? 1 : threads, names[mode],
           seconds, final_loss, difference, zeros, ok ? "OK" : "FAIL");
    fflush(stdout);
}

#ifndef AUTO_RUN
int main(int argc, char *argv[]) {
    if (argc != 8) {
        fprintf(stderr, "Usage: linreg_simd N d lambda eta K threads mode\n");
        return 1;
    }
    int n = atoi(argv[1]), d = atoi(argv[2]);
    double lambda = atof(argv[3]), eta = atof(argv[4]);
    int iterations = atoi(argv[5]), threads = atoi(argv[6]);
    int mode = -1;
    for (int i = 0; i < 4; i++)
        if (strcmp(argv[7], names[i]) == 0) mode = i;
    if (n < 1 || d < 1 || lambda < 0 || eta <= 0 || iterations < 1 ||
        threads < 1 || mode == -1) return 1;
    Problem p;
    make_data(&p, n, d, threads);
    reference(&p, iterations, eta, lambda);
    print_case("RESULT", &p, mode, threads, iterations, eta, lambda);
    free_data(&p);
    return 0;
}
#endif
