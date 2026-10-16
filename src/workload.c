// 1D heat-diffusion (Jacobi) workload. Strong scaling: N is fixed, split across ranks.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long N = 2000000, iters = 1000, fail_iter = -1;
    int fail_rank = -1;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--n")) N = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atol(argv[++i]);
        else if (!strcmp(argv[i], "--fail-iter")) fail_iter = atol(argv[++i]);
        else if (!strcmp(argv[i], "--fail-rank")) fail_rank = atoi(argv[++i]);
    }

    long base = N / size, rem = N % size;
    long local = base + (rank < rem ? 1 : 0);
    long start = rank * base + (rank < rem ? rank : rem);

    double *u = calloc(local + 2, sizeof(double));
    double *v = calloc(local + 2, sizeof(double));
    for (long i = 0; i < local; i++) u[i + 1] = 1.0 + sin(1e-3 * (double)(start + i));

    int left  = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int right = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (long it = 0; it < iters; it++) {
        // Deterministic failure injection: this rank dies abruptly (real process kill)
        if (it == fail_iter && rank == fail_rank) kill(getpid(), SIGKILL);

        MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0, &u[local + 1], 1, MPI_DOUBLE, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[local], 1, MPI_DOUBLE, right, 1, &u[0], 1, MPI_DOUBLE, left, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (long i = 1; i <= local; i++)
            v[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1]);
        double *tmp = u; u = v; v = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime() - t0;

    double s = 0.0, total = 0.0;
    for (long i = 1; i <= local; i++) s += u[i];
    MPI_Reduce(&s, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("RESULT ranks=%d N=%ld iters=%ld checksum=%.12e time=%.6f\n",
               size, N, iters, total, t);

    free(u); free(v);
    MPI_Finalize();
    return 0;
}
