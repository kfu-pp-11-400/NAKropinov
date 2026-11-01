#include <mpi.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WARMUPS 10
#define REPEATS 10000

int main(int argc, char **argv)
{
    int rank, p, n = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (rank == 0) {
        if (argc == 2 || (argc == 3 && strcmp(argv[2], "bench") == 0)) {
            char *end;
            errno = 0;
            long value = strtol(argv[1], &end, 10);
            if (errno == 0 && end != argv[1] && *end == '\0' &&
                value > 0 && value <= INT_MAX) {
                n = (int)value;
            }
        }
        if (n == 0) {
            fprintf(stderr, "Usage: mpi_array_sum N [bench]; N > 0.\n");
        }
    }

    double start = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    int count = n / p + (rank < n % p);
    int *part = malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    int *data = NULL;
    int *counts = NULL;
    int *displs = NULL;
    long long *sums = NULL;

    if (rank == 0) {
        data = malloc((size_t)n * sizeof(int));
        sums = malloc((size_t)p * sizeof(long long));
        if (n % p != 0) {
            counts = malloc((size_t)p * sizeof(int));
            displs = malloc((size_t)p * sizeof(int));
        }
        if (data != NULL) {
            for (int i = 0; i < n; ++i) {
                data[i] = i + 1;
            }
        }
        if (counts != NULL && displs != NULL) {
            int offset = 0;
            for (int r = 0; r < p; ++r) {
                counts[r] = n / p + (r < n % p);
                displs[r] = offset;
                offset += counts[r];
            }
        }
    }

    if (part == NULL ||
        (rank == 0 && (data == NULL || sums == NULL ||
                      (n % p != 0 && (counts == NULL || displs == NULL))))) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (n % p == 0) {
        MPI_Scatter(data, n / p, MPI_INT, part, count, MPI_INT,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(data, counts, displs, MPI_INT, part, count, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    long long local_sum = 0;
    for (int i = 0; i < count; ++i) {
        local_sum += part[i];
    }

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, sums, 1,
               MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    long long gather_sum = 0;
    if (rank == 0) {
        for (int r = 0; r < p; ++r) {
            gather_sum += sums[r];
        }
    }

    long long reduce_sum = 0;
    long long allreduce_sum = 0;
    MPI_Reduce(&local_sum, &reduce_sum, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum, &allreduce_sum, 1, MPI_LONG_LONG_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    long long expected = (long long)n * ((long long)n + 1) / 2;

    if (rank == 0) {
        int correct = gather_sum == expected && reduce_sum == expected &&
                      allreduce_sum == expected;
        printf("RESULT P=%d N=%d EXPECTED=%lld GATHER=%lld REDUCE=%lld ALLREDUCE=%lld OK=%s TIME_S=%.9f DISTRIBUTION=%s\n",
               p, n, expected, gather_sum, reduce_sum, allreduce_sum,
               correct ? "yes" : "no", elapsed,
               n % p == 0 ? "SCATTER" : "SCATTERV");
    }
    printf("RANK=%d COUNT=%d ALLREDUCE=%lld OK=%s\n", rank, count,
           allreduce_sum, allreduce_sum == expected ? "yes" : "no");

    if (argc == 3) {
        double begin = 0.0;
        for (int i = 0; i < WARMUPS + REPEATS; ++i) {
            if (rank == 0 && i == WARMUPS) {
                begin = MPI_Wtime();
            }
            MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, sums, 1,
                       MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                gather_sum = 0;
                for (int r = 0; r < p; ++r) {
                    gather_sum += sums[r];
                }
            }
        }
        if (rank == 0) {
            printf("BENCH P=%d METHOD=GATHER REPEATS=%d AVG_S=%.12f OK=%s\n",
                   p, REPEATS, (MPI_Wtime() - begin) / REPEATS,
                   gather_sum == expected ? "yes" : "no");
        }

        for (int i = 0; i < WARMUPS + REPEATS; ++i) {
            if (rank == 0 && i == WARMUPS) {
                begin = MPI_Wtime();
            }
            MPI_Reduce(&local_sum, &reduce_sum, 1, MPI_LONG_LONG_INT,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            printf("BENCH P=%d METHOD=REDUCE REPEATS=%d AVG_S=%.12f OK=%s\n",
                   p, REPEATS, (MPI_Wtime() - begin) / REPEATS,
                   reduce_sum == expected ? "yes" : "no");
        }
    }

    free(displs);
    free(counts);
    free(sums);
    free(data);
    free(part);
    MPI_Finalize();
    return 0;
}
