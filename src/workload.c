// 1D heat-diffusion (Jacobi) workload. Strong scaling: N is fixed, split across ranks.
//
// S01 (Teammate 1): failure injection via --fail-iter / --fail-rank (real SIGKILL).
// S02 (Teammate 2): application-level coordinated checkpoint / restart.
//   --ckpt-mode none|periodic|adaptive   (default none = identical behaviour to S01)
//   --ckpt-interval K                    periodic: every K iterations; adaptive: first interval
//   --ckpt-dir PATH                      where checkpoint files live (default "ckpt")
//   --restart                            resume from the last committed checkpoint, if any
//   --mtbf-iters M                       adaptive only: expected iterations between failures
//   --ckpt-fsync 0|1                     fsync checkpoint files before commit (default 1)
//
// Checkpoint protocol (coordinated, double-buffered):
//   1. every rank writes its local array to slot s (alternates 0/1), fsyncs, closes
//   2. MPI_Allreduce of write success: all ranks must succeed
//   3. rank 0 writes commit.txt via write-to-tmp + rename (atomic), naming the slot
//   A crash at any point leaves the previous committed slot intact.
//
// Output: the S01 RESULT line is unchanged. A second line starting with CKPT is added.
// Rank 0 also appends a timeline to <ckpt-dir>/events.log (ready / commit events with
// wall-clock epoch times) so the runner can measure lost work and recovery time even
// for attempts that are killed.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>

#define CKPT_MAGIC 0x43534b50u  /* "CSKP" */

typedef struct {
    unsigned magic;
    int ranks, rank;
    long N, local, iter;
} ckpt_header;

static double epoch_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static void log_event(const char *dir, const char *line) {
    char path[4096];
    snprintf(path, sizeof path, "%s/events.log", dir);
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%s\n", line);
    fflush(f);
    fsync(fileno(f));
    fclose(f);
}

static int write_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += w; n -= (size_t)w;
    }
    return 0;
}

static int read_all(int fd, void *buf, size_t n) {
    char *p = buf;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

// Write this rank's state to slot file. Returns 0 on success.
static int write_ckpt(const char *dir, int slot, const ckpt_header *h, const double *u,
                      int do_fsync) {
    char path[4096];
    snprintf(path, sizeof path, "%s/ckpt_r%d_s%d.bin", dir, h->rank, slot);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    int bad = write_all(fd, h, sizeof *h) ||
              write_all(fd, u, (size_t)(h->local + 2) * sizeof(double));
    if (!bad && do_fsync) bad = fsync(fd) != 0;
    if (close(fd) != 0) bad = 1;
    return bad ? -1 : 0;
}

static int read_ckpt(const char *dir, int slot, const ckpt_header *expect, double *u,
                     long *iter_out) {
    char path[4096];
    snprintf(path, sizeof path, "%s/ckpt_r%d_s%d.bin", dir, expect->rank, slot);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ckpt_header h;
    int bad = read_all(fd, &h, sizeof h);
    if (!bad && (h.magic != CKPT_MAGIC || h.ranks != expect->ranks || h.rank != expect->rank ||
                 h.N != expect->N || h.local != expect->local)) bad = 1;
    if (!bad) bad = read_all(fd, u, (size_t)(h.local + 2) * sizeof(double));
    close(fd);
    if (bad) return -1;
    *iter_out = h.iter;
    return 0;
}

// Rank 0: atomically publish which slot holds the newest complete checkpoint.
static int write_commit(const char *dir, long iter, int slot, int ranks, long N, long interval,
                        long count) {
    char tmp[4096], fin[4096];
    snprintf(tmp, sizeof tmp, "%s/commit.txt.tmp", dir);
    snprintf(fin, sizeof fin, "%s/commit.txt", dir);
    FILE *f = fopen(tmp, "w");
    if (!f) return -1;
    fprintf(f, "iter=%ld slot=%d ranks=%d N=%ld interval=%ld count=%ld\n",
            iter, slot, ranks, N, interval, count);
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    return rename(tmp, fin);
}

static int read_commit(const char *dir, long *iter, int *slot, int *ranks, long *N,
                       long *interval) {
    char path[4096];
    snprintf(path, sizeof path, "%s/commit.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long count;
    int n = fscanf(f, "iter=%ld slot=%d ranks=%d N=%ld interval=%ld count=%ld",
                   iter, slot, ranks, N, interval, &count);
    fclose(f);
    return n == 6 ? 0 : -1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long N = 2000000, iters = 1000, fail_iter = -1;
    int fail_rank = -1;
    // S02 options
    const char *ckpt_mode_s = "none", *ckpt_dir = "ckpt";
    long ckpt_interval = 100;
    double mtbf_iters = -1.0;
    int do_restart = 0, do_fsync = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--restart")) { do_restart = 1; continue; }
        if (i + 1 >= argc) break;
        if (!strcmp(argv[i], "--n")) N = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atol(argv[++i]);
        else if (!strcmp(argv[i], "--fail-iter")) fail_iter = atol(argv[++i]);
        else if (!strcmp(argv[i], "--fail-rank")) fail_rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt-mode")) ckpt_mode_s = argv[++i];
        else if (!strcmp(argv[i], "--ckpt-interval")) ckpt_interval = atol(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt-dir")) ckpt_dir = argv[++i];
        else if (!strcmp(argv[i], "--mtbf-iters")) mtbf_iters = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt-fsync")) do_fsync = atoi(argv[++i]);
    }

    int mode;  // 0 none, 1 periodic, 2 adaptive
    if (!strcmp(ckpt_mode_s, "none")) mode = 0;
    else if (!strcmp(ckpt_mode_s, "periodic")) mode = 1;
    else if (!strcmp(ckpt_mode_s, "adaptive")) mode = 2;
    else {
        if (rank == 0) fprintf(stderr, "unknown --ckpt-mode %s\n", ckpt_mode_s);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    if (ckpt_interval < 1) ckpt_interval = 1;
    if (mode == 2 && mtbf_iters <= 0) {
        if (rank == 0) fprintf(stderr, "--ckpt-mode adaptive needs --mtbf-iters > 0\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    int uses_dir = (mode != 0) || do_restart;
    if (uses_dir && rank == 0) mkdir(ckpt_dir, 0755);  // ignore EEXIST

    long base = N / size, rem = N % size;
    long local = base + (rank < rem ? 1 : 0);
    long start = rank * base + (rank < rem ? rank : rem);

    double *u = calloc(local + 2, sizeof(double));
    double *v = calloc(local + 2, sizeof(double));
    for (long i = 0; i < local; i++) u[i + 1] = 1.0 + sin(1e-3 * (double)(start + i));

    int left  = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int right = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    ckpt_header hdr = { CKPT_MAGIC, size, rank, N, local, 0 };

    // ---------------- restart ----------------
    long it0 = 0;           // first iteration to execute
    int next_slot = 0;
    long ckpt_count = 0;
    double restore_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (do_restart) {
        double tr = MPI_Wtime();
        long info[5] = { -1, 0, 0, 0, 0 };  // iter, slot, ranks, N, interval
        if (rank == 0) {
            long ci, cN, cint; int cs, cr;
            if (read_commit(ckpt_dir, &ci, &cs, &cr, &cN, &cint) == 0) {
                info[0] = ci; info[1] = cs; info[2] = cr; info[3] = cN; info[4] = cint;
            }
        }
        MPI_Bcast(info, 5, MPI_LONG, 0, MPI_COMM_WORLD);
        if (info[0] >= 0) {
            if (info[2] != size || info[3] != N) {
                if (rank == 0)
                    fprintf(stderr, "checkpoint was taken with ranks=%ld N=%ld; this run has "
                            "ranks=%d N=%ld. Restart needs the same layout.\n",
                            info[2], info[3], size, N);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            long got = -1;
            int ok = read_ckpt(ckpt_dir, (int)info[1], &hdr, u, &got) == 0 && got == info[0];
            int all_ok;
            MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            if (!all_ok) {
                if (rank == 0) fprintf(stderr, "failed to read committed checkpoint\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            it0 = info[0];
            next_slot = 1 - (int)info[1];
            if (mode == 2 && info[4] > 0) ckpt_interval = info[4];  // keep adaptive estimate
        }
        restore_time = MPI_Wtime() - tr;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    if (uses_dir && rank == 0) {
        char line[256];
        snprintf(line, sizeof line, "ready epoch=%.6f resumed_from=%ld restore_s=%.6f",
                 epoch_now(), it0, restore_time);
        log_event(ckpt_dir, line);
    }

    double ckpt_time = 0.0;         // time spent checkpointing in this attempt (this rank)
    double ckpt_cost_sum = 0.0;     // global (max over ranks) per-checkpoint cost, summed
    long last_ckpt = it0;
    long interval = ckpt_interval;

    for (long it = it0; it < iters; it++) {
        // Deterministic failure injection: this rank dies abruptly (real process kill)
        if (it == fail_iter && rank == fail_rank) kill(getpid(), SIGKILL);

        MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0, &u[local + 1], 1, MPI_DOUBLE, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[local], 1, MPI_DOUBLE, right, 1, &u[0], 1, MPI_DOUBLE, left, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (long i = 1; i <= local; i++)
            v[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1]);
        double *tmp = u; u = v; v = tmp;

        long done = it + 1;  // iterations completed so far
        if (mode != 0 && done < iters && done - last_ckpt >= interval) {
            double tc = MPI_Wtime();
            hdr.iter = done;
            int ok = write_ckpt(ckpt_dir, next_slot, &hdr, u, do_fsync) == 0;
            int all_ok;
            MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            if (all_ok) {
                ckpt_count++;
                long new_interval = interval;
                if (mode == 2) {
                    // Young's formula: tau = sqrt(2 * C * MTBF), all in iteration units.
                    // C and per-iteration time are measured online (max over ranks so
                    // every rank computes the same interval).
                    double mine[2] = { MPI_Wtime() - tc, 0.0 };
                    double elapsed = MPI_Wtime() - t0;
                    double compute = elapsed - ckpt_time - mine[0];
                    mine[1] = compute / (double)(done - it0);
                    double g[2];
                    MPI_Allreduce(mine, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                    double c_iters = g[0] / (g[1] > 0 ? g[1] : 1e-9);
                    new_interval = (long)llround(sqrt(2.0 * c_iters * mtbf_iters));
                    if (new_interval < 1) new_interval = 1;
                    if (new_interval > iters) new_interval = iters;
                }
                if (rank == 0) {
                    write_commit(ckpt_dir, done, next_slot, size, N, new_interval, ckpt_count);
                }
                MPI_Barrier(MPI_COMM_WORLD);  // commit is visible before anyone proceeds
                double cost = MPI_Wtime() - tc, gcost;
                MPI_Allreduce(&cost, &gcost, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                ckpt_cost_sum += gcost;
                if (rank == 0) {
                    char line[256];
                    snprintf(line, sizeof line,
                             "commit epoch=%.6f iter=%ld cost_s=%.6f next_interval=%ld",
                             epoch_now(), done, gcost, new_interval);
                    log_event(ckpt_dir, line);
                }
                next_slot = 1 - next_slot;
                last_ckpt = done;
                interval = new_interval;
            } else if (rank == 0) {
                fprintf(stderr, "checkpoint at iter %ld failed on some rank; skipped\n", done);
            }
            ckpt_time += MPI_Wtime() - tc;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime() - t0;

    double s = 0.0, total = 0.0;
    for (long i = 1; i <= local; i++) s += u[i];
    MPI_Reduce(&s, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("RESULT ranks=%d N=%ld iters=%ld checksum=%.12e time=%.6f\n",
               size, N, iters, total, t);
        printf("CKPT mode=%s ckpts=%ld ckpt_time=%.6f restore_time=%.6f resumed_from=%ld "
               "final_interval=%ld\n",
               ckpt_mode_s, ckpt_count, ckpt_cost_sum, restore_time, it0, interval);
        if (uses_dir) {
            char line[128];
            snprintf(line, sizeof line, "finish epoch=%.6f", epoch_now());
            log_event(ckpt_dir, line);
        }
    }

    free(u); free(v);
    MPI_Finalize();
    return 0;
}
