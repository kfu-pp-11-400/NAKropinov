#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

typedef struct {
    int n, m;
    int *offsets, *edges;
} Graph;

typedef struct {
    int *data;
    int size, cap;
} List;

static void *allocate(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

static void add(List *list, int value) {
    if (list->size == list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 32;
        int *p = realloc(list->data, (size_t)new_cap * sizeof(int));
        if (!p) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        list->data = p;
        list->cap = new_cap;
    }
    list->data[list->size++] = value;
}

static Graph read_graph(const char *filename, int undirected) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n, m;
    if (fscanf(file, "%d %d", &n, &m) != 2 || n <= 0 || m < 0) {
        fprintf(stderr, "Invalid graph header\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (undirected && m > 1073741823) {
        fprintf(stderr, "Too many edges\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *from = allocate((size_t)m * sizeof(int));
    int *to = allocate((size_t)m * sizeof(int));
    int *degree = calloc((size_t)n, sizeof(int));
    if (!degree) MPI_Abort(MPI_COMM_WORLD, 1);

    for (int i = 0; i < m; i++) {
        if (fscanf(file, "%d %d", &from[i], &to[i]) != 2 ||
            from[i] < 0 || from[i] >= n || to[i] < 0 || to[i] >= n) {
            fprintf(stderr, "Invalid edge %d\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        degree[from[i]]++;
        if (undirected) degree[to[i]]++;
    }
    fclose(file);

    int arcs = m * (undirected ? 2 : 1);
    int *offsets = allocate((size_t)(n + 1) * sizeof(int));
    int *edges = allocate((size_t)arcs * sizeof(int));
    int *cursor = allocate((size_t)n * sizeof(int));

    offsets[0] = 0;
    for (int i = 0; i < n; i++) offsets[i + 1] = offsets[i] + degree[i];
    memcpy(cursor, offsets, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        edges[cursor[from[i]]++] = to[i];
        if (undirected) edges[cursor[to[i]]++] = from[i];
    }

    free(from);
    free(to);
    free(degree);
    free(cursor);
    Graph graph = {n, m, offsets, edges};
    return graph;
}

static void sequential_bfs(const Graph *g, int source, int *dist) {
    int *queue = allocate((size_t)g->n * sizeof(int));
    for (int i = 0; i < g->n; i++) dist[i] = -1;
    int head = 0, tail = 0;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        int v = queue[head++];
        for (int j = g->offsets[v]; j < g->offsets[v + 1]; j++) {
            int u = g->edges[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
    free(queue);
}

static void hybrid_bfs(const Graph *g, int source, int rank, int nprocs,
                       int sparse, int stats, int *dist, double *communication,
                       int *levels, int *maximum_frontier, long long *vertices_done,
                       long long *edges_done, int *frontier_sizes) {
    int n = g->n;
    int threads = omp_get_max_threads();
    int *frontier = allocate((size_t)n * sizeof(int));
    int *next_frontier = allocate((size_t)n * sizeof(int));
    int *local = allocate((size_t)n * sizeof(int));
    int *global = allocate((size_t)n * sizeof(int));
    List *lists = calloc((size_t)threads, sizeof(List));
    long long *vertex_counts = calloc((size_t)threads, sizeof(long long));
    long long *edge_counts = calloc((size_t)threads, sizeof(long long));
    if (!lists || !vertex_counts || !edge_counts) MPI_Abort(MPI_COMM_WORLD, 1);

    int *sendbuf = sparse ? allocate((size_t)n * sizeof(int)) : NULL;
    int *counts = sparse ? allocate((size_t)nprocs * sizeof(int)) : NULL;
    int *displs = sparse ? allocate((size_t)nprocs * sizeof(int)) : NULL;
    int *recvbuf = sparse ? allocate(sizeof(int)) : NULL;
    int recv_capacity = 1;

    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[source] = 0;
    frontier[0] = source;
    int size = 1;
    *levels = 0;
    *maximum_frontier = 0;
    *communication = 0.0;
    *vertices_done = 0;
    *edges_done = 0;

    while (size > 0) {
        if (stats) frontier_sizes[*levels] = size;
        (*levels)++;
        if (size > *maximum_frontier) *maximum_frontier = size;
        memset(local, 0, (size_t)n * sizeof(int));
        for (int t = 0; t < threads; t++) lists[t].size = 0;
        int local_size = (size > rank) ? (size - 1 - rank) / nprocs + 1 : 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(runtime)
            for (int k = 0; k < local_size; k++) {
                int v = frontier[rank + k * nprocs];
                vertex_counts[tid]++;
                edge_counts[tid] += g->offsets[v + 1] - g->offsets[v];
                for (int j = g->offsets[v]; j < g->offsets[v + 1]; j++) {
                    int u = g->edges[j];
                    if (dist[u] == -1) add(&lists[tid], u);
                }
            }
        }

        for (int t = 0; t < threads; t++) {
            for (int j = 0; j < lists[t].size; j++) local[lists[t].data[j]] = 1;
        }

        if (!sparse) {
            double start = MPI_Wtime();
            MPI_Allreduce(local, global, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            *communication += MPI_Wtime() - start;
        } else {
            int send_size = 0;
            for (int u = 0; u < n; u++) {
                if (local[u]) sendbuf[send_size++] = u;
            }
            double start = MPI_Wtime();
            MPI_Allgather(&send_size, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
            *communication += MPI_Wtime() - start;

            int received = 0;
            for (int p = 0; p < nprocs; p++) {
                displs[p] = received;
                received += counts[p];
            }
            if (received > recv_capacity) {
                int *p = realloc(recvbuf, (size_t)received * sizeof(int));
                if (!p) MPI_Abort(MPI_COMM_WORLD, 1);
                recvbuf = p;
                recv_capacity = received;
            }
            start = MPI_Wtime();
            MPI_Allgatherv(sendbuf, send_size, MPI_INT, recvbuf, counts,
                           displs, MPI_INT, MPI_COMM_WORLD);
            *communication += MPI_Wtime() - start;
            memset(global, 0, (size_t)n * sizeof(int));
            for (int j = 0; j < received; j++) global[recvbuf[j]] = 1;
        }

        int next_size = 0;
        for (int u = 0; u < n; u++) {
            if (global[u] && dist[u] == -1) {
                dist[u] = *levels;
                next_frontier[next_size++] = u;
            }
        }
        int *tmp = frontier;
        frontier = next_frontier;
        next_frontier = tmp;
        size = next_size;
    }

    for (int t = 0; t < threads; t++) {
        *vertices_done += vertex_counts[t];
        *edges_done += edge_counts[t];
        free(lists[t].data);
    }
    free(lists);
    free(vertex_counts);
    free(edge_counts);
    free(frontier);
    free(next_frontier);
    free(local);
    free(global);
    free(sendbuf);
    free(counts);
    free(displs);
    free(recvbuf);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3 || argc > 7) {
        if (rank == 0) fprintf(stderr, "Usage: bfs graph.txt source [directed|undirected] [dense|sparse] [static|dynamic] [stats]\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    char *end;
    long value = strtol(argv[2], &end, 10);
    if (*end || value < 0 || value > 2147483647L) MPI_Abort(MPI_COMM_WORLD, 1);
    int source = (int)value;
    int undirected = argc > 3 && strcmp(argv[3], "undirected") == 0;
    int sparse = argc > 4 && strcmp(argv[4], "sparse") == 0;
    int dynamic = argc <= 5 || strcmp(argv[5], "dynamic") == 0;
    int stats = argc > 6 && strcmp(argv[6], "stats") == 0;
    if ((argc > 3 && strcmp(argv[3], "directed") && strcmp(argv[3], "undirected")) ||
        (argc > 4 && strcmp(argv[4], "dense") && strcmp(argv[4], "sparse")) ||
        (argc > 5 && strcmp(argv[5], "static") && strcmp(argv[5], "dynamic")) ||
        (argc > 6 && strcmp(argv[6], "stats"))) MPI_Abort(MPI_COMM_WORLD, 1);

    omp_set_dynamic(0);
    omp_set_schedule(dynamic ? omp_sched_dynamic : omp_sched_static, 1);
    Graph graph = read_graph(argv[1], undirected);
    if (source >= graph.n) MPI_Abort(MPI_COMM_WORLD, 1);

    int *dist = allocate((size_t)graph.n * sizeof(int));
    int *seq = rank == 0 ? allocate((size_t)graph.n * sizeof(int)) : NULL;
    int *frontier_sizes = stats ? allocate((size_t)graph.n * sizeof(int)) : NULL;
    double sequential_time = 0.0;
    if (rank == 0) {
        double start = MPI_Wtime();
        sequential_bfs(&graph, source, seq);
        sequential_time = MPI_Wtime() - start;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double communication;
    int levels, max_frontier;
    long long vertices_done, edges_done;
    double start = MPI_Wtime();
    hybrid_bfs(&graph, source, rank, nprocs, sparse, stats, dist, &communication,
               &levels, &max_frontier, &vertices_done, &edges_done, frontier_sizes);
    double elapsed = MPI_Wtime() - start;
    double parallel_time, max_communication;
    MPI_Reduce(&elapsed, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communication, &max_communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long *all_vertices = stats && rank == 0 ? allocate((size_t)nprocs * sizeof(long long)) : NULL;
    long long *all_edges = stats && rank == 0 ? allocate((size_t)nprocs * sizeof(long long)) : NULL;
    if (stats) {
        MPI_Gather(&vertices_done, 1, MPI_LONG_LONG_INT, all_vertices, 1,
                   MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&edges_done, 1, MPI_LONG_LONG_INT, all_edges, 1,
                   MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        int mismatches = 0, unreachable = 0, seq_levels = 0;
        for (int i = 0; i < graph.n; i++) {
            if (seq[i] != dist[i]) mismatches++;
            if (dist[i] == -1) unreachable++;
            if (seq[i] + 1 > seq_levels) seq_levels = seq[i] + 1;
        }
        double speedup = sequential_time / parallel_time;
        double efficiency = speedup / (nprocs * omp_get_max_threads());
        printf("RESULT n=%d m=%d degree=%.3f source=%d directed=%d levels=%d seq_levels=%d unreachable=%d mismatches=%d mpi=%d omp=%d total=%d mode=%s schedule=%s seq=%.9f par=%.9f speedup=%.6f efficiency=%.6f comm=%.9f max_frontier=%d\n",
               graph.n, graph.m, (undirected ? 2.0 : 1.0) * graph.m / graph.n, source, !undirected,
               levels, seq_levels, unreachable, mismatches, nprocs,
               omp_get_max_threads(), nprocs * omp_get_max_threads(),
               sparse ? "sparse" : "dense", dynamic ? "dynamic" : "static",
               sequential_time, parallel_time, speedup, efficiency,
               max_communication, max_frontier);
        if (stats) {
            printf("FRONTIERS sizes=");
            for (int i = 0; i < levels; i++) printf("%s%d", i ? "," : "", frontier_sizes[i]);
            printf("\nLOAD_VERTICES counts=");
            for (int i = 0; i < nprocs; i++) printf("%s%lld", i ? "," : "", all_vertices[i]);
            printf("\nLOAD_EDGES counts=");
            for (int i = 0; i < nprocs; i++) printf("%s%lld", i ? "," : "", all_edges[i]);
            printf("\n");
        }
    }

    free(all_vertices);
    free(all_edges);
    free(frontier_sizes);
    free(seq);
    free(dist);
    free(graph.offsets);
    free(graph.edges);
    MPI_Finalize();
    return 0;
}
