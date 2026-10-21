
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x) {
    double c = cos(x), s = sin(x);
    return c * c * c * s * s;
}

double integrate_seq(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;

    for (long long i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }

    return sum * h;
}

int main(int argc, char *argv[]) {
    long long n = atoll(argv[1]);
    int threads = atoi(argv[2]);
    double a = 0.0;
    double b = acos(-1.0) / 3.0;

    if (argc == 5) {
        a = atof(argv[3]);
        b = atof(argv[4]);
    }

    if (n == 0) {
        double sa = sin(a), sb = sin(b);
        double exact = (pow(sb, 3) / 3.0 - pow(sb, 5) / 5.0)
                     - (pow(sa, 3) / 3.0 - pow(sa, 5) / 5.0);

        n = 1;
        while (fabs(integrate_seq(a, b, n) - exact) > 1e-14) {
            n *= 2;
        }
    }

    double start = omp_get_wtime();
    double result_seq = integrate_seq(a, b, n);
    double time_seq = omp_get_wtime() - start;

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    double h = (b - a) / n;
    double sum = 0.0;

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }

    double result_parallel = sum * h;
    double time_parallel = omp_get_wtime() - start;

    printf("n=%lld threads=%d\n", n, threads);
    printf("seq: I=%.15f time=%.9f\n", result_seq, time_seq);
    printf("omp: I=%.15f time=%.9f\n", result_parallel, time_parallel);

    return 0;
}
