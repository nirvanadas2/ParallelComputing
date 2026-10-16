# CS 14 Shared Spec (v1, written by Teammate 1)

Status: S01 code is done. Please do not change anything marked FROZEN without telling all three of us.

## 1. Environment
- Ubuntu on WSL2, Open MPI 4.1.6, Python 3.12, gcc via build-essential
- Machine: one laptop, 16 cores. All MPI ranks share this machine (no cluster).
- Run with: `mpirun --oversubscribe -np <ranks> ./workload ...`
- Compile with: `mpicc -O2 -o workload src/workload.c -lm`

## 2. Workload (src/workload.c) FROZEN
1D Jacobi heat diffusion. Total grid size N is fixed and split across ranks (strong scaling).
Each iteration: halo exchange with neighbours (MPI_Sendrecv), then a 3-point stencil update.

Flags:
| Flag | Default | Meaning |
|---|---|---|
| --n | 2000000 | total grid points |
| --iters | 1000 | number of iterations |
| --fail-iter | -1 | iteration at which a rank kills itself (-1 = never) |
| --fail-rank | -1 | which rank dies |

Failure injection: the chosen rank calls `kill(getpid(), SIGKILL)` at the start of iteration `fail-iter`.
This is a real process kill, not a simulated error. With plain Open MPI 4.1.6 (no ULFM) the whole job aborts.

Output (rank 0, one line at the end):
`RESULT ranks=<p> N=<n> iters=<i> checksum=<sum> time=<seconds>`

Correctness check: the checksum of a failed-then-recovered run must equal the failure-free
reference (1-rank run), relative tolerance 1e-9. Different rank counts add in different order,
so do not compare checksums bit-for-bit.

Planned additions by Teammate 2 (checkpointing). Proposed flags, to be agreed before coding:
`--ckpt-mode none|periodic|adaptive`, `--ckpt-interval <iters>`, `--ckpt-dir <path>`, `--restart`.
Existing flags and the RESULT line must keep working unchanged.

## 3. Metric definitions (what we actually measure)
- **Makespan**: wall-clock seconds from the first launch of the job until the final successful
  attempt ends, including all crashed attempts and relaunches. Failed runs with no recovery
  report time until the crash and are marked completed=0.
- **Attempt**: one mpirun invocation.
- **Lost work (seconds)**: wall-clock time spent in attempts that crashed. Includes MPI startup
  and teardown, not only compute iterations. Reported in seconds.
- **Lost work (iterations)**: by design equals the fail iteration for restart-from-scratch
  (the iterations done before the crash must all be redone). A definition, not a measurement.
- **Failure rate**: expected failures per failure-free run, = 1 / mtbf_mult, where each attempt
  samples its failure iteration from an exponential distribution with mean mtbf_mult * iters.
- **Slowdown**: makespan divided by mean failure-free makespan at the same rank count and size.
- **Recovery time (restart-from-scratch)**: not separately measured yet. relaunch_gap is only the gap
  between processes in our script and is near zero. Checkpoint recovery time is for Teammate 2/3 to define.
- **Availability proxy** (Teammate 3 to finalise): final successful attempt time / makespan.

Time is never counted twice: makespan = lost work (crashed attempts) + final successful attempt
(+ small script gaps).

## 4. CSV schemas FROZEN
results/runs.csv (rank-count and failure-position sweep):
run_id, mode, ranks, N, iters, fail_iter, fail_rank, failure_occurred, time_to_failure,
iters_lost, relaunch_gap, completed, makespan, checksum_ok
- mode is one of: baseline, norecovery, restart

results/rate_runs.csv (random failure-rate experiment):
run_id, ranks, iters, mtbf_mult, expected_failures_per_run, seed, attempts, failures,
completed, makespan, lost_work_s, final_attempt_s, checksum_ok
- mtbf_mult is "inf" for failure-free baselines

New modes added by teammates should reuse these columns and add new columns at the END only.

## 5. Scripts
- scripts/run_one.py : one experiment (baseline / norecovery / restart), appends to results/runs.csv
- scripts/sweep.py : rank-count x failure-position sweep (84 runs)
- scripts/run_rate.py : random-failure-rate experiment (44 rows)
- scripts/analyze.py, scripts/analyze_rate.py : summary tables and plots into results/

## 6. Experiment conventions
- Sweep size: N = 2,000,000, iters = 5000 (about 13 to 28 s failure-free depending on ranks).
- Rate experiment size: 4 ranks, iters = 2000 (about 6 s failure-free).
- Rank counts 1, 2, 4, 8 (never above 16 = core count).
- Seeds: rate experiment uses seed * 1000 + int(mtbf_mult * 100).
- Close other heavy programs, keep the laptop plugged in and awake during runs.
- Failed or capped runs are kept in the CSV (completed=0), never deleted.

## 7. Known limitations (S01)
- One machine, ranks share cores; not a cluster result.
- Baselines use only 3 repeats; the 1-rank baseline varies about 20%.
- Only restart-from-scratch recovery so far; no checkpointing yet.
- Failures are injected at iteration boundaries, not at arbitrary times.
- Scaling saturates at about 4 ranks on this laptop; cause not tested.

## 8. Git rules
- Repo: https://github.com/nirvanadas2/ParallelComputing
- Pull before you start, commit small, never commit compiled binaries (see .gitignore).
- Tell the others before changing src/workload.c or a FROZEN item.
