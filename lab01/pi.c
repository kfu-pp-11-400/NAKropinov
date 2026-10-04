#include <stdio.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    long long n;
    int threads;

    scanf("%lld", &n);
    scanf("%d", &threads);

    double sum = 0.0;

    double start = omp_get_wtime();

    for (long long i = 0; i < n; i++) {
        double term = 1.0 / (2.0 * i + 1.0);

        if (i % 2 == 0)
            sum += term;
        else
            sum -= term;
    }

    double pi_seq = 4.0 * sum;
    double time_seq = omp_get_wtime() - start;

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    sum = 0.0;

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double term = 1.0 / (2.0 * i + 1.0);

        if (i % 2 == 0)
            sum += term;
        else
            sum -= term;
    }

    double pi_parallel = 4.0 * sum;
    double time_parallel = omp_get_wtime() - start;

    printf("%.15f\n", pi_seq);
    printf("%.15e\n", fabs(pi_seq - M_PI));
    printf("%.6f\n", time_seq);

    printf("%.15f\n", pi_parallel);
    printf("%.15e\n", fabs(pi_parallel - M_PI));
    printf("%.6f\n", time_parallel);
    printf("%d\n", threads);

    return 0;
}