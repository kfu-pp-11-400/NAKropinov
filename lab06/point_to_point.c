#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_MESSAGE 1
#define TAG_REPLY 2
#define WARMUPS 10
#define REPEATS 1000

static void exchange_messages(int rank, int process_count)
{
    char message[64];
    char reply[64];

    if (rank == 0) {
        for (int r = 1; r < process_count; r++) {
            snprintf(message, sizeof(message), "Message for rank %d", r);
            MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR,
                     r, TAG_MESSAGE, MPI_COMM_WORLD);
            printf("Rank 0 sent to rank %d: %s\n", r, message);
        }

        for (int r = 1; r < process_count; r++) {
            MPI_Recv(reply, sizeof(reply), MPI_CHAR,
                     r, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Rank 0 received from rank %d: %s\n", r, reply);
        }

        printf("SUMMARY processes=%d sent=%d replies=%d all_received=yes\n",
               process_count, process_count - 1, process_count - 1);
    } else {
        MPI_Recv(message, sizeof(message), MPI_CHAR,
                 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received from rank 0: %s\n", rank, message);

        snprintf(reply, sizeof(reply), "Message received by rank %d", rank);
        MPI_Send(reply, (int)strlen(reply) + 1, MPI_CHAR,
                 0, TAG_REPLY, MPI_COMM_WORLD);
        printf("Rank %d sent reply to rank 0: %s\n", rank, reply);
    }
}

static void measure_exchange(int rank)
{
    const int sizes[] = {1, 1024, 65536, 1048576};
    char *buffer = malloc(1048576);

    if (buffer == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(buffer, 'A', 1048576);

    for (int s = 0; s < 4; s++) {
        int bytes = sizes[s];
        double start = 0.0;

        for (int i = 0; i < WARMUPS + REPEATS; i++) {
            if (rank == 0) {
                if (i == WARMUPS) {
                    start = MPI_Wtime();
                }
                MPI_Send(buffer, bytes, MPI_CHAR,
                         1, TAG_MESSAGE, MPI_COMM_WORLD);
                MPI_Recv(buffer, bytes, MPI_CHAR,
                         1, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buffer, bytes, MPI_CHAR,
                         0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, bytes, MPI_CHAR,
                         0, TAG_REPLY, MPI_COMM_WORLD);
            }
        }

        if (rank == 0) {
            double average = (MPI_Wtime() - start) / REPEATS;
            printf("SIZE_BYTES=%d REPEATS=%d AVG_RTT_S=%.9f\n",
                   bytes, REPEATS, average);
        }
    }

    free(buffer);
}

int main(int argc, char **argv)
{
    int rank, process_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    if (process_count < 2) {
        if (rank == 0) {
            fprintf(stderr, "At least 2 MPI processes are required.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (argc > 1 && strcmp(argv[1], "time") == 0) {
        if (process_count != 2) {
            if (rank == 0) {
                fprintf(stderr, "Time experiment requires 2 processes.\n");
            }
            MPI_Finalize();
            return 1;
        }
        measure_exchange(rank);
    } else {
        exchange_messages(rank, process_count);
    }

    MPI_Finalize();
    return 0;
}
