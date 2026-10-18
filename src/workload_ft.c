// S03 (Teammate 3): fault-tolerant version of the S01/S02 workload. src/workload.c stays FROZEN.
//
// Same 1D Jacobi heat diffusion, same checkpoint format and commit protocol as workload.c, so
// the S02 checkpoint files are read and written unchanged. One binary runs every S03 strategy,
// so all of them log the same timestamps:
//
//   A  global restart from scratch   --ft relaunch --ckpt-mode none     (runner relaunches)
//   B  global rollback               --ft relaunch --ckpt-mode periodic (runner relaunches
//                                     with --restart)
//   C  ULFM in-place recovery        --ft ulfm --spares S --ckpt-mode periodic
//      launch p+S processes; p compute positions, S spares. When a process dies the survivors
//      revoke and shrink the communicator, a spare takes over the dead position, everyone
//      reloads the last committed checkpoint and continues. No relaunch, no teardown.
//      If more positions die than there are spares, the job aborts and the runner relaunches
//      it with --restart (falls back to B).
//
// New flags (all workload.c flags still work, with the same meaning):
//   --ft relaunch|ulfm      default relaunch = any MPI error kills the job, exactly like workload.c
//   --spares S              ulfm only: number of spare processes (default 0)
//   --fail-plan F:P,F:P,..  planned failures, consumed in order. Failure k hits position P_k at
//                           iteration resume_k + floor(F_k * (iters - resume_k)), where resume_k
//                           is the iteration the run (re)started from before failure k. F in
//                           [0,1). Same rule for every strategy, so A, B and C see the same
//                           failures (common random numbers, as in S02).
//   --fail-in-ckpt 1        planned failures happen inside the first checkpoint written at or
//                           after the planned iteration: the rank writes its header and half of
//                           its array, then dies (exercises the double buffer).
//   --fail-iter/--fail-rank as in workload.c (one failure at an absolute iteration).
//
// Events (appended to <ckpt-dir>/events.log, wall-clock epoch seconds):
//   fail  epoch iter pos inckpt          written by the dying process just before SIGKILL
//   ready epoch resumed_from restore_s   compute about to start (first start or after recovery)
//         + inplace=1 detect_epoch reconf_s for ULFM in-place recoveries
//   commit epoch iter cost_s next_interval      (as workload.c)
//   finish epoch                                (as workload.c)
// Recovery time for every strategy = ready.epoch - fail.epoch (kill to compute resuming).
//
// Output (position 0): RESULT and CKPT lines as workload.c, then
//   FT ft=<relaunch|ulfm> spares=<S> spares_left=<n> recoveries=<k> reconf_s=<s>
// RESULT time = seconds since this launch's start barrier (includes restores and in-place
// recoveries). CKPT counters only cover the process that ends at position 0; use events.log.
#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif
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

#if defined(MPIX_ERR_PROC_FAILED) && defined(MPIX_ERR_REVOKED)
#define HAVE_ULFM 1
#else
#define HAVE_ULFM 0
#endif

#define CKPT_MAGIC 0x43534b50u  /* "CSKP" */
#define TAG_DONE   77
#define MAX_PLAN   64

typedef struct {
    unsigned magic;
    int ranks, rank;
    long N, local, iter;
} ckpt_header;

typedef struct { double frac; long abs_iter; int pos; } planned_failure;

// ---------------- configuration ----------------
static long N = 2000000, iters = 1000;
static const char *ckpt_mode_s = "none", *ckpt_dir = "ckpt";
static int mode = 0;                 // 0 none, 1 periodic, 2 adaptive
static long ckpt_interval = 100;
static double mtbf_iters = -1.0;
static int do_restart = 0, do_fsync = 1, uses_dir = 0;
static int ulfm = 0, nspares = 0, fail_in_ckpt = 0;
static planned_failure plan[MAX_PLAN];
static int nplan = 0;

// ---------------- process layout ----------------
static MPI_Comm comm = MPI_COMM_NULL;   // every live process (compute + spares)
static MPI_Comm work = MPI_COMM_NULL;   // compute processes only; rank in work == position
static int p;                           // number of compute positions
static int pos;                         // my position, -1 = spare
static int *pos_of = NULL;              // position of every rank of comm
static int comm_n;

// ---------------- state kept across in-place recoveries ----------------
static int fired = 0;                   // planned failures already consumed
static int recoveries = 0, spares_left = 0;
static double reconf_total = 0.0, last_detect = 0.0, last_reconf = 0.0;
static long ckpt_count = 0;
static double ckpt_cost_sum = 0.0, restore_total = 0.0;
static double t_launch = 0.0;

static double epoch_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static void log_event_opt(const char *line, int sync) {
    char path[4096];
    snprintf(path, sizeof path, "%s/events.log", ckpt_dir);
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%s\n", line);
    fflush(f);
    if (sync) fsync(fileno(f));
    fclose(f);
}
static void log_event(const char *line) { log_event_opt(line, 1); }

// A planned failure: record the time, then die abruptly (real SIGKILL). No fsync: the line is
// already in the kernel once fclose returns, and a process kill does not lose it.
static void die(long iter, int inckpt) {
    if (uses_dir) {
        char line[256];
        snprintf(line, sizeof line, "fail epoch=%.6f iter=%ld pos=%d inckpt=%d",
                 epoch_now(), iter, pos, inckpt);
        log_event_opt(line, 0);
    }
    kill(getpid(), SIGKILL);
}

static int write_all(int fd, const void *buf, size_t n) {
    const char *q = buf;
    while (n > 0) {
        ssize_t w = write(fd, q, n);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        q += w; n -= (size_t)w;
    }
    return 0;
}

static int read_all(int fd, void *buf, size_t n) {
    char *q = buf;
    while (n > 0) {
        ssize_t r = read(fd, q, n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        q += r; n -= (size_t)r;
    }
    return 0;
}

// Same file format as workload.c. die_iter >= 0: write the header and half of the array,
// then die (failure during a checkpoint).
static int write_ckpt(int slot, const ckpt_header *h, const double *u, long die_iter) {
    char path[4096];
    snprintf(path, sizeof path, "%s/ckpt_r%d_s%d.bin", ckpt_dir, h->rank, slot);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    size_t n = (size_t)(h->local + 2);
    if (die_iter >= 0) {
        write_all(fd, h, sizeof *h);
        write_all(fd, u, (n / 2) * sizeof(double));
        die(die_iter, 1);
    }
    int bad = write_all(fd, h, sizeof *h) || write_all(fd, u, n * sizeof(double));
    if (!bad && do_fsync) bad = fsync(fd) != 0;
    if (close(fd) != 0) bad = 1;
    return bad ? -1 : 0;
}

static int read_ckpt(int slot, const ckpt_header *expect, double *u, long *iter_out) {
    char path[4096];
    snprintf(path, sizeof path, "%s/ckpt_r%d_s%d.bin", ckpt_dir, expect->rank, slot);
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

static int write_commit(long iter, int slot, long interval, long count) {
    char tmp[4096], fin[4096];
    snprintf(tmp, sizeof tmp, "%s/commit.txt.tmp", ckpt_dir);
    snprintf(fin, sizeof fin, "%s/commit.txt", ckpt_dir);
    FILE *f = fopen(tmp, "w");
    if (!f) return -1;
    fprintf(f, "iter=%ld slot=%d ranks=%d N=%ld interval=%ld count=%ld\n",
            iter, slot, p, N, interval, count);
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    return rename(tmp, fin);
}

static int read_commit(long *iter, int *slot, int *ranks, long *n, long *interval) {
    char path[4096];
    snprintf(path, sizeof path, "%s/commit.txt", ckpt_dir);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long count;
    int k = fscanf(f, "iter=%ld slot=%d ranks=%d N=%ld interval=%ld count=%ld",
                   iter, slot, ranks, n, interval, &count);
    fclose(f);
    return k == 6 ? 0 : -1;
}

// Iteration of the next planned failure for a run that (re)started at `resume`, or -1.
static long plan_fi(long resume) {
    if (fired >= nplan) return -1;
    const planned_failure *f = &plan[fired];
    long fi = f->abs_iter >= 0 ? f->abs_iter
                               : resume + (long)(f->frac * (double)(iters - resume));
    return (fi < resume || fi >= iters) ? -1 : fi;
}

static void parse_plan(const char *s, int world_rank) {
    char *copy = strdup(s), *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        double fr; int q;
        if (sscanf(tok, "%lf:%d", &fr, &q) != 2 || fr < 0.0 || fr >= 1.0 || nplan >= MAX_PLAN) {
            if (world_rank == 0) fprintf(stderr, "bad --fail-plan entry '%s' (want F:P, 0<=F<1)\n", tok);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        plan[nplan].frac = fr; plan[nplan].abs_iter = -1; plan[nplan].pos = q; nplan++;
    }
    free(copy);
}

// ---------------- compute ----------------
// Runs this position from a fresh start or the last committed checkpoint to the end.
// Returns MPI_SUCCESS when the whole job finished, otherwise the MPI error code of the
// first call that failed (only possible with --ft ulfm).
static int compute_phase(int want_restore, int after_failure) {
    long base = N / p, rem = N % p;
    long local = base + (pos < rem ? 1 : 0);
    long start = pos * base + (pos < rem ? pos : rem);
    double *u = calloc(local + 2, sizeof(double));
    double *v = calloc(local + 2, sizeof(double));
    for (long i = 0; i < local; i++) u[i + 1] = 1.0 + sin(1e-3 * (double)(start + i));

    int left  = pos > 0 ? pos - 1 : MPI_PROC_NULL;
    int right = pos < p - 1 ? pos + 1 : MPI_PROC_NULL;
    ckpt_header hdr = { CKPT_MAGIC, p, pos, N, local, 0 };
    long it0 = 0, interval = ckpt_interval;
    int next_slot = 0, rc = MPI_SUCCESS;
    double restore_time = 0.0;
#define CHK(call) do { rc = (call); if (rc != MPI_SUCCESS) goto failed; } while (0)

    if (want_restore) {
        double tr = MPI_Wtime();
        long info[6] = { -1, 0, 0, 0, 0, 0 };  // iter, slot, ranks, N, interval, count
        if (pos == 0) {
            long ci, cN, cint; int cs, cr;
            if (read_commit(&ci, &cs, &cr, &cN, &cint) == 0) {
                info[0] = ci; info[1] = cs; info[2] = cr; info[3] = cN; info[4] = cint;
            }
        }
        CHK(MPI_Bcast(info, 6, MPI_LONG, 0, work));
        if (info[0] >= 0) {
            if (info[2] != p || info[3] != N) {
                if (pos == 0)
                    fprintf(stderr, "checkpoint was taken with ranks=%ld N=%ld; this run has "
                            "ranks=%d N=%ld. Restart needs the same layout.\n",
                            info[2], info[3], p, N);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            long got = -1;
            int ok = read_ckpt((int)info[1], &hdr, u, &got) == 0 && got == info[0], all_ok;
            CHK(MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, work));
            if (!all_ok) {
                if (pos == 0) fprintf(stderr, "failed to read committed checkpoint\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            it0 = info[0];
            next_slot = 1 - (int)info[1];
            if (mode == 2 && info[4] > 0) interval = info[4];  // keep adaptive estimate
        }
        restore_time = MPI_Wtime() - tr;
    }

    CHK(MPI_Barrier(work));
    double t0 = MPI_Wtime();
    restore_total += restore_time;
    if (uses_dir && pos == 0) {
        char line[256];
        if (after_failure)
            snprintf(line, sizeof line, "ready epoch=%.6f resumed_from=%ld restore_s=%.6f "
                     "inplace=1 detect_epoch=%.6f reconf_s=%.6f",
                     epoch_now(), it0, restore_time, last_detect, last_reconf);
        else
            snprintf(line, sizeof line, "ready epoch=%.6f resumed_from=%ld restore_s=%.6f",
                     epoch_now(), it0, restore_time);
        log_event(line);
    }

    long fi = plan_fi(it0);
    int i_fail = fi >= 0 && plan[fired].pos == pos;
    double ckpt_time = 0.0;
    long last_ckpt = it0;

    for (long it = it0; it < iters; it++) {
        if (i_fail && !fail_in_ckpt && it == fi) die(it, 0);

        CHK(MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0, &u[local + 1], 1, MPI_DOUBLE, right, 0,
                         work, MPI_STATUS_IGNORE));
        CHK(MPI_Sendrecv(&u[local], 1, MPI_DOUBLE, right, 1, &u[0], 1, MPI_DOUBLE, left, 1,
                         work, MPI_STATUS_IGNORE));
        for (long i = 1; i <= local; i++)
            v[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1]);
        double *tmp = u; u = v; v = tmp;

        long done = it + 1;
        if (mode != 0 && done < iters && done - last_ckpt >= interval) {
            double tc = MPI_Wtime();
            hdr.iter = done;
            long die_iter = (i_fail && fail_in_ckpt && done >= fi) ? done : -1;
            int ok = write_ckpt(next_slot, &hdr, u, die_iter) == 0, all_ok;
            CHK(MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, work));
            if (all_ok) {
                ckpt_count++;
                long new_interval = interval;
                if (mode == 2) {
                    // Young's formula, as in workload.c (max over ranks so all agree)
                    double mine[2] = { MPI_Wtime() - tc, 0.0 };
                    double elapsed = MPI_Wtime() - t0;
                    double compute = elapsed - ckpt_time - mine[0];
                    mine[1] = compute / (double)(done - it0);
                    double g[2];
                    CHK(MPI_Allreduce(mine, g, 2, MPI_DOUBLE, MPI_MAX, work));
                    double c_iters = g[0] / (g[1] > 0 ? g[1] : 1e-9);
                    new_interval = (long)llround(sqrt(2.0 * c_iters * mtbf_iters));
                    if (new_interval < 1) new_interval = 1;
                    if (new_interval > iters) new_interval = iters;
                }
                if (pos == 0) write_commit(done, next_slot, new_interval, ckpt_count);
                CHK(MPI_Barrier(work));  // commit is visible before anyone proceeds
                double cost = MPI_Wtime() - tc, gcost;
                CHK(MPI_Allreduce(&cost, &gcost, 1, MPI_DOUBLE, MPI_MAX, work));
                ckpt_cost_sum += gcost;
                if (pos == 0) {
                    char line[256];
                    snprintf(line, sizeof line,
                             "commit epoch=%.6f iter=%ld cost_s=%.6f next_interval=%ld",
                             epoch_now(), done, gcost, new_interval);
                    log_event(line);
                }
                next_slot = 1 - next_slot;
                last_ckpt = done;
                interval = new_interval;
            } else if (pos == 0) {
                fprintf(stderr, "checkpoint at iter %ld failed on some rank; skipped\n", done);
            }
            ckpt_time += MPI_Wtime() - tc;
        }
    }

    {
        double s = 0.0, total = 0.0;
        for (long i = 1; i <= local; i++) s += u[i];
        int frc = MPI_Barrier(work);
        if (frc == MPI_SUCCESS) frc = MPI_Reduce(&s, &total, 1, MPI_DOUBLE, MPI_SUM, 0, work);
#if HAVE_ULFM
        if (ulfm) {
            // Errors are not reported uniformly: make sure every compute rank finished.
            int flag = frc == MPI_SUCCESS;
            int arc = MPIX_Comm_agree(work, &flag);
            if (arc != MPI_SUCCESS) frc = arc;
            else if (!flag && frc == MPI_SUCCESS) frc = MPI_ERR_OTHER;
        }
#endif
        CHK(frc);

        if (pos == 0) {
            double t = MPI_Wtime() - t_launch;
            printf("RESULT ranks=%d N=%ld iters=%ld checksum=%.12e time=%.6f\n",
                   p, N, iters, total, t);
            printf("CKPT mode=%s ckpts=%ld ckpt_time=%.6f restore_time=%.6f resumed_from=%ld "
                   "final_interval=%ld\n",
                   ckpt_mode_s, ckpt_count, ckpt_cost_sum, restore_total, it0, interval);
            printf("FT ft=%s spares=%d spares_left=%d recoveries=%d reconf_s=%.6f\n",
                   ulfm ? "ulfm" : "relaunch", nspares, spares_left, recoveries, reconf_total);
            fflush(stdout);
            if (uses_dir) {
                char line[128];
                snprintf(line, sizeof line, "finish epoch=%.6f", epoch_now());
                log_event(line);
            }
            int one = 1;
            for (int r = 0; r < comm_n; r++)  // release the spares
                if (pos_of[r] < 0) MPI_Send(&one, 1, MPI_INT, r, TAG_DONE, comm);
        }
    }
    free(u); free(v);
    return MPI_SUCCESS;

failed:
    free(u); free(v);
    return rc;
#undef CHK
}

// A spare waits for either the end of the job or a failure (the communicator is revoked).
static int spare_wait(void) {
    int x;
    return MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE, TAG_DONE, comm, MPI_STATUS_IGNORE);
}

// ---------------- ULFM recovery ----------------
// Revoke, shrink, find the dead positions, give them to spares, rebuild the compute
// communicator. Every step that could see a new failure is checked with MPIX_Comm_agree,
// and the whole round is retried if anything failed. State changes only after a round that
// every survivor agreed on. Returns 0, or -1 if there are not enough spares.
static int reconfigure(double my_detect) {
#if HAVE_ULFM
    double tstart = MPI_Wtime();
    for (int round = 0; round < 100; round++) {
        if (work != MPI_COMM_NULL) MPIX_Comm_revoke(work);
        MPIX_Comm_revoke(comm);
        MPI_Comm nc;
        if (MPIX_Comm_shrink(comm, &nc) != MPI_SUCCESS) continue;
        if (work != MPI_COMM_NULL) MPI_Comm_free(&work);
        MPI_Comm_free(&comm);
        comm = nc;
        MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

        int n, me;
        MPI_Comm_size(comm, &n);
        MPI_Comm_rank(comm, &me);
        double mine[2] = { (double)pos, my_detect };
        double *all = malloc(2 * (size_t)n * sizeof(double));
        int rc = MPI_Allgather(mine, 2, MPI_DOUBLE, all, 2, MPI_DOUBLE, comm);
        int flag = rc == MPI_SUCCESS;
        if (MPIX_Comm_agree(comm, &flag) != MPI_SUCCESS || !flag) { free(all); continue; }

        char *present = calloc((size_t)p, 1);
        int *missing = malloc((size_t)p * sizeof(int)), *spare = malloc((size_t)n * sizeof(int));
        int *new_pos_of = malloc((size_t)n * sizeof(int));
        int nmiss = 0, nsp = 0;
        double gdet = 1e300;
        for (int r = 0; r < n; r++) {
            int q = (int)all[2 * r];
            new_pos_of[r] = q;
            if (q >= 0) present[q] = 1; else spare[nsp++] = r;
            if (all[2 * r + 1] < gdet) gdet = all[2 * r + 1];
        }
        for (int q = 0; q < p; q++) if (!present[q]) missing[nmiss++] = q;
        free(all); free(present);
        if (nmiss > nsp) {
            if (me == 0)
                fprintf(stderr, "ULFM: %d position(s) lost, only %d spare(s) left; aborting\n",
                        nmiss, nsp);
            free(missing); free(spare); free(new_pos_of);
            return -1;
        }
        for (int i = 0; i < nmiss; i++) new_pos_of[spare[i]] = missing[i];
        int new_pos = new_pos_of[me];
        free(missing); free(spare);

        MPI_Comm nw = MPI_COMM_NULL;
        rc = MPI_Comm_split(comm, new_pos >= 0 ? 0 : MPI_UNDEFINED, new_pos, &nw);
        flag = rc == MPI_SUCCESS;
        if (MPIX_Comm_agree(comm, &flag) != MPI_SUCCESS || !flag) {
            if (nw != MPI_COMM_NULL) MPI_Comm_free(&nw);
            free(new_pos_of);
            continue;
        }
        // commit the new layout
        pos = new_pos;
        free(pos_of); pos_of = new_pos_of; comm_n = n;
        fired += nmiss;
        spares_left = nsp - nmiss;
        work = nw;
        if (work != MPI_COMM_NULL) MPI_Comm_set_errhandler(work, MPI_ERRORS_RETURN);
        last_detect = gdet;
        last_reconf = MPI_Wtime() - tstart;
        reconf_total += last_reconf;
        return 0;
    }
#else
    (void)my_detect;
#endif
    return -1;
}

int main(int argc, char **argv) {
    // ULFM: skip the all-process fence at the start of MPI_Finalize. That fence includes the
    // dead processes and occasionally never completes (seen ~1 in 7 runs when rank 0 had died,
    // Open MPI 5.0.11). Safe here: survivors already agree (MPIX_Comm_agree) that the job
    // finished before anyone reaches MPI_Finalize. Must be set before MPI_Init.
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--ft") && !strcmp(argv[i + 1], "ulfm"))
            setenv("OMPI_MCA_async_mpi_finalize", "1", 0);
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long fail_iter = -1;
    int fail_rank = -1;
    const char *ft_s = "relaunch";
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
        else if (!strcmp(argv[i], "--ft")) ft_s = argv[++i];
        else if (!strcmp(argv[i], "--spares")) nspares = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fail-plan")) parse_plan(argv[++i], world_rank);
        else if (!strcmp(argv[i], "--fail-in-ckpt")) fail_in_ckpt = atoi(argv[++i]);
    }
    if (fail_iter >= 0 && fail_rank >= 0 && nplan < MAX_PLAN) {  // workload.c-style failure
        plan[nplan].frac = 0; plan[nplan].abs_iter = fail_iter; plan[nplan].pos = fail_rank;
        nplan++;
    }

#define DIE_USAGE(...) do { if (world_rank == 0) fprintf(stderr, __VA_ARGS__); \
                            MPI_Abort(MPI_COMM_WORLD, 2); return 2; } while (0)
    if (!strcmp(ckpt_mode_s, "none")) mode = 0;
    else if (!strcmp(ckpt_mode_s, "periodic")) mode = 1;
    else if (!strcmp(ckpt_mode_s, "adaptive")) mode = 2;
    else DIE_USAGE("unknown --ckpt-mode %s\n", ckpt_mode_s);
    if (!strcmp(ft_s, "relaunch")) ulfm = 0;
    else if (!strcmp(ft_s, "ulfm")) ulfm = 1;
    else DIE_USAGE("unknown --ft %s (relaunch or ulfm)\n", ft_s);
    if (ulfm && !HAVE_ULFM)
        DIE_USAGE("--ft ulfm needs an MPI with ULFM (Open MPI 5 built with --with-ft=ulfm)\n");
    if (!ulfm && nspares > 0) DIE_USAGE("--spares needs --ft ulfm\n");
    if (nspares < 0 || nspares >= world_size) DIE_USAGE("--spares must be 0..ranks-1\n");
    if (ckpt_interval < 1) ckpt_interval = 1;
    if (mode == 2 && mtbf_iters <= 0)
        DIE_USAGE("--ckpt-mode adaptive needs --mtbf-iters > 0\n");
    p = world_size - nspares;
    for (int k = 0; k < nplan; k++)
        if (plan[k].pos < 0 || plan[k].pos >= p)
            DIE_USAGE("failure position %d is outside 0..%d\n", plan[k].pos, p - 1);

    uses_dir = mode != 0 || do_restart || ulfm || nplan > 0;
    if (uses_dir && world_rank == 0) mkdir(ckpt_dir, 0755);  // ignore EEXIST

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    if (ulfm) MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    comm_n = world_size;
    pos = world_rank < p ? world_rank : -1;
    pos_of = malloc((size_t)world_size * sizeof(int));
    for (int r = 0; r < world_size; r++) pos_of[r] = r < p ? r : -1;
    spares_left = nspares;
    MPI_Comm_split(comm, pos >= 0 ? 0 : MPI_UNDEFINED, pos, &work);
    if (ulfm && work != MPI_COMM_NULL) MPI_Comm_set_errhandler(work, MPI_ERRORS_RETURN);

    MPI_Barrier(comm);
    t_launch = MPI_Wtime();

    int want_restore = do_restart, after_failure = 0;
    for (;;) {
        int rc = pos >= 0 ? compute_phase(want_restore, after_failure) : spare_wait();
        if (rc == MPI_SUCCESS) break;
        double my_detect = epoch_now();   // only reached with --ft ulfm
        if (reconfigure(my_detect) != 0) MPI_Abort(comm, 4);  // runner relaunches (strategy B)
        recoveries++;
        want_restore = 1;
        after_failure = 1;
    }

    if (work != MPI_COMM_NULL) MPI_Comm_free(&work);
    MPI_Comm_free(&comm);
    free(pos_of);
    MPI_Finalize();
    return 0;
}
