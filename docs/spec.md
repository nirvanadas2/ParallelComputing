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

### 2a. Checkpointing (S02, Teammate 2) — implemented
The four proposed flags are kept exactly as proposed, plus two extras. With no new flags the
program behaves exactly as in S01 (same RESULT line, same checksum, nothing written to disk).

| Flag | Default | Meaning |
|---|---|---|
| --ckpt-mode | none | none, periodic or adaptive |
| --ckpt-interval | 100 | periodic: checkpoint every K iterations. adaptive: first interval only |
| --ckpt-dir | ckpt | directory for checkpoint files and the event log |
| --restart | off | resume from the last committed checkpoint; if none exists, start at iteration 0 |
| --mtbf-iters | (none) | adaptive only (required): expected iterations between failures |
| --ckpt-fsync | 1 | 1 = fsync each checkpoint file before committing (realistic cost), 0 = skip |

Protocol: coordinated, application-level, double-buffered. After iteration k every rank writes its
local array to slot 0 or 1 (`ckpt_r<rank>_s<slot>.bin`, with a header holding ranks, N and k),
fsyncs it, then all ranks agree via MPI_Allreduce that every write succeeded. Rank 0 then writes
`commit.txt` atomically (write tmp file, rename). A crash at any point leaves the previous committed
slot intact. No checkpoint is taken after the last iteration.

Adaptive mode uses Young's formula K* = sqrt(2 * C * MTBF), in iteration units. C (cost of one
checkpoint) and the time per iteration are measured online at every checkpoint (max over ranks, so
all ranks pick the same next interval). The latest K* is stored in commit.txt, so a restarted
attempt keeps it.

Restart needs the same rank count and N as the checkpoint; otherwise the program stops with an error.
A restarted run gives a bit-identical checksum to a failure-free run with the same rank count.

Extra output (rank 0), after the unchanged RESULT line:
`CKPT mode=<m> ckpts=<n> ckpt_time=<s> restore_time=<s> resumed_from=<iter> final_interval=<K>`
Rank 0 also appends to `<ckpt-dir>/events.log` (wall-clock epoch times, also written by attempts
that later crash): `ready epoch=.. resumed_from=.. restore_s=..`,
`commit epoch=.. iter=.. cost_s=.. next_interval=..`, `finish epoch=..`.

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
- **Recovery time (S02 proposal, Teammate 3 please confirm for S03)**: per failure, wall-clock from
  the moment the crashed mpirun exits to the moment the relaunched job has restored its state and
  passed the start barrier (the `ready` event). Includes relaunch, MPI startup and checkpoint read.
  Measured the same way for "none" (restart from scratch), so strategies are comparable.
- **Checkpoint overhead (S02)**: two views. (a) measured: sum of per-checkpoint cost (write + fsync +
  agreement + commit, max over ranks) over all attempts, column ckpt_overhead_s. (b) end-to-end:
  failure-free makespan with checkpointing minus failure-free makespan without, as % of the latter.
- **Lost compute (S02)**: iterations that had to be redone (fail iteration minus last committed
  checkpoint iteration) times the measured time per iteration. Column lost_compute_s.
- **Failure detection + teardown (S02)**: per crashed attempt, time from the last safe point to the
  crashed mpirun exiting, minus lost compute. This is Open MPI noticing the killed rank and aborting
  the others (about 0.5 s per failure on the Mac). Column detect_teardown_s.
- **lost_work_s keeps its S01 meaning** (all wall time in crashed attempts), so S01 and S02 rows can
  be compared. With checkpointing, the useful part of a crashed attempt is not lost, so for S02 use
  lost_compute_s as "lost work" and report lost_work_s only for comparison with S01.
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

results/ckpt_runs.csv (S02). The first 13 columns are exactly the rate_runs.csv columns, same
meaning; S02 columns follow at the end:
run_id, ranks, iters, mtbf_mult, expected_failures_per_run, seed, attempts, failures, completed,
makespan, lost_work_s, final_attempt_s, checksum_ok, experiment, strategy, ckpt_interval, ckpts,
ckpt_overhead_s, restore_s, lost_compute_s, lost_iters, recovery_time_s_total,
recovery_time_s_mean, final_interval, N, fsync, detect_teardown_s, t_iter_s
- experiment is overhead (failure-free) or failures
- strategy is none, periodic or adaptive; ckpt_interval is set for periodic only
- final_interval is the interval in use at the end (the adaptive K*)

New modes added by teammates should reuse these columns and add new columns at the END only.

## 5. Scripts
- scripts/run_one.py : one experiment (baseline / norecovery / restart), appends to results/runs.csv
- scripts/sweep.py : rank-count x failure-position sweep (84 runs)
- scripts/run_rate.py : random-failure-rate experiment (44 rows)
- scripts/analyze.py, scripts/analyze_rate.py : summary tables and plots into results/
- scripts/run_ckpt.py : S02 runner, overhead and failure experiments, appends to results/ckpt_runs.csv
  (`--quick` for a 2-minute smoke test; MPIRUN_EXTRA env var for extra mpirun flags)
- scripts/analyze_ckpt.py : S02 tables (results/ckpt_summary.csv) and plots (results/plots/ckpt_*.png)

## 6. Experiment conventions
- Sweep size: N = 2,000,000, iters = 5000 (about 13 to 28 s failure-free depending on ranks).
- Rate experiment size: 4 ranks, iters = 2000 (about 6 s failure-free).
- Rank counts 1, 2, 4, 8 (never above 16 = core count).
- Seeds: rate experiment uses seed * 1000 + int(mtbf_mult * 100).
- Close other heavy programs, keep the laptop plugged in and awake during runs.
- Failed or capped runs are kept in the CSV (completed=0), never deleted.

- S02 size: 4 ranks, N = 2,000,000, iters = 2000 (same as the rate experiment). Intervals
  10, 25, 50, 100, 200, 500, 1000. Failure rates 1 and 2 expected failures per run
  (mtbf_mult 1.0, 0.5). 3 overhead repeats, 5 seeds per failure configuration.
- S02 failure draws: each attempt draws iterations-until-failure from the same exponential as S01,
  counted from the iteration it resumes at. Same seed rule as S01, and every strategy uses the same
  random sequence for a given (mtbf_mult, seed).

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

## 9. Known limitations (S02)
- Checkpoints go to the local SSD of one machine, and the OS page cache absorbs part of the cost;
  on a cluster with a shared filesystem a checkpoint would cost much more.
- Failure exposure is counted in computed iterations, so time spent checkpointing cannot itself
  be hit by a failure. Real failures happen in wall-clock time.
- Adaptive mode is told the true MTBF; it does not estimate MTBF from observed failures.
- Failures happen at iteration boundaries, never in the middle of writing a checkpoint, so the
  double-buffer protection is implemented but not exercised by the experiments.
- About 0.5 s of Open MPI failure detection/teardown per failure (Mac, Open MPI 5.0.11) is the
  same for every strategy. Once the interval is tuned it is the largest remaining cost of a failure,
  larger than checkpoint writing plus lost compute, and checkpointing cannot reduce it.
- Per-checkpoint cost is not constant: about 7 ms at K >= 50 but 16-20 ms at K = 10-25, because
  back-to-back fsyncs queue up. Young's formula assumes a constant C; we use the median.
- Only 5 seeds per failure configuration. The restart-from-scratch ("none") results vary a lot
  (sd 3-4 s), so its comparison with checkpointing is directional; the interval trend is clearer.
