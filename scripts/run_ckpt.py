"""S02 experiment runner: checkpoint overhead and checkpoint/restart under random failures.

Two experiments, both appending to results/ckpt_runs.csv:

  overhead  failure-free runs: no checkpoint vs periodic (several intervals) vs adaptive.
            Gives checkpoint overhead as a function of interval.
  failures  random failures (same model as S01's run_rate.py) with automatic relaunch.
            Strategies: none (restart from scratch, = S01), periodic(K), adaptive.

Failure model: each attempt draws iterations-until-failure from an exponential distribution
with mean mtbf_mult * iters, counted from the iteration the attempt resumes at. For a given
(mtbf_mult, seed) every strategy sees the same sequence of random draws (common random
numbers), so differences between strategies are not caused by luckier failure draws.

Usage (from repo root, after `mpicc -O2 -o workload src/workload.c -lm`):
  python3 scripts/run_ckpt.py                    # both experiments, full size
  python3 scripts/run_ckpt.py --exp overhead
  python3 scripts/run_ckpt.py --quick            # small smoke test (~2 min)
Extra mpirun flags (e.g. --allow-run-as-root) can be passed via env MPIRUN_EXTRA.
"""
import argparse, csv, os, random, re, shutil, subprocess, time, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--exp", choices=["overhead", "failures", "both"], default="both")
ap.add_argument("--ranks", type=int, default=4)
ap.add_argument("--n", type=int, default=2000000)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--repeats", type=int, default=3, help="overhead experiment repeats")
ap.add_argument("--seeds", type=int, default=5, help="failures experiment seeds")
ap.add_argument("--intervals", default="10,25,50,100,200,500,1000")
ap.add_argument("--mults", default="1.0,0.5", help="MTBF as multiple of run length")
ap.add_argument("--max-attempts", type=int, default=25)
ap.add_argument("--fsync", type=int, default=1)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results/ckpt_runs.csv")
a = ap.parse_args()

if a.quick:
    a.repeats, a.seeds, a.iters = 1, 2, 1000
    a.intervals, a.mults = "25,100,500", "0.5"

INTERVALS = [int(x) for x in a.intervals.split(",")]
MULTS = [float(x) for x in a.mults.split(",")]
EXTRA = os.environ.get("MPIRUN_EXTRA", "").split()
CKPT_ROOT = "ckpt_tmp"


def run(ranks, extra_args, timeout=900):
    cmd = (["mpirun", "--oversubscribe"] + EXTRA + ["-np", str(ranks), "./workload",
           "--n", str(a.n), "--iters", str(a.iters)] + [str(x) for x in extra_args])
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    t1 = time.time()
    m = re.search(r"RESULT .*checksum=(\S+)", p.stdout)
    ck = dict(re.findall(r"(\w+)=(\S+)", re.search(r"CKPT .*", p.stdout).group(0))) \
        if "CKPT " in p.stdout else {}
    return p.returncode, t0, t1, (float(m.group(1)) if m else None), ck


def read_events(d):
    path = os.path.join(d, "events.log")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        parts = line.split()
        if parts:
            ev = {"kind": parts[0]}
            ev.update({k: float(v) for k, v in (x.split("=") for x in parts[1:])})
            out.append(ev)
    return out


def strategy_args(strategy, interval, adapt_mult, d):
    args = ["--ckpt-dir", d, "--ckpt-fsync", a.fsync]
    if strategy == "periodic":
        args += ["--ckpt-mode", "periodic", "--ckpt-interval", interval]
    elif strategy == "adaptive":
        # first interval is a guess; the program replaces it using Young's formula
        args += ["--ckpt-mode", "adaptive", "--ckpt-interval", 50,
                 "--mtbf-iters", adapt_mult * a.iters]
    else:
        args += ["--ckpt-mode", "none"]
    return args


def one_job(strategy, interval, mult, seed, experiment, adapt_mult=None):
    """Run one job to completion (relaunching after crashes). Returns a CSV row.
    mult = failure MTBF multiple (inf = no failures). adapt_mult = MTBF the adaptive
    strategy is told (defaults to the true one)."""
    if adapt_mult is None:
        adapt_mult = mult
    d = os.path.join(CKPT_ROOT, uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    rng = random.Random(seed * 1000 + int(mult * 100)) if mult != float("inf") else None
    start = time.time()
    attempts = failures = 0
    lost_wall = lost_compute = lost_iters = detect = 0.0
    recov = []
    done, final_len, cs, ck = False, 0.0, None, {}
    prev_end = None
    resume = 0
    while attempts < a.max_attempts and not done:
        attempts += 1
        fi, fr = -1, -1
        if rng is not None:
            gap = int(rng.expovariate(1.0 / (mult * a.iters)))
            fr = rng.randrange(a.ranks)
            fi = resume + gap if resume + gap < a.iters else -1
        n_before = len(read_events(d))
        # --restart on every attempt: with no committed checkpoint the program starts at 0,
        # so strategy "none" restarts from scratch exactly like S01.
        rc, s, e, cs, ck = run(a.ranks, strategy_args(strategy, interval, adapt_mult, d) +
                               ["--fail-iter", fi, "--fail-rank", fr, "--restart"])
        evs = read_events(d)[n_before:]
        ready = next((x for x in evs if x["kind"] == "ready"), None)
        if prev_end is not None and ready is not None:
            recov.append(ready["epoch"] - prev_end)
        if rc == 0:
            done, final_len = True, e - s
        else:
            failures += 1
            lost_wall += e - s
            commits = [x for x in evs if x["kind"] == "commit"]
            # work is safe up to the later of: this attempt being ready, or its last commit
            safe_epoch = max([s] + ([ready["epoch"]] if ready else []) +
                             [c["epoch"] for c in commits])
            safe_iter = commits[-1]["iter"] if commits else (ready["resumed_from"] if ready else 0)
            li = (fi - safe_iter) if fi >= 0 else 0
            lost_iters += li
            # lost compute = iterations redone x measured time per iteration.
            # The rest of (crash end - last safe point) is Open MPI noticing the dead rank
            # and tearing the job down: failure detection/teardown, reported separately.
            lc = li * T_ITER
            lost_compute += lc
            detect += max(0.0, (e - safe_epoch) - lc)
            resume = int(commits[-1]["iter"]) if commits else (
                int(ready["resumed_from"]) if ready else resume)
            prev_end = e
    end = time.time()
    all_evs = read_events(d)
    ckpt_cost = sum(x["cost_s"] for x in all_evs if x["kind"] == "commit")
    n_ckpts = sum(1 for x in all_evs if x["kind"] == "commit")
    restore = sum(x["restore_s"] for x in all_evs if x["kind"] == "ready")
    shutil.rmtree(d, ignore_errors=True)
    ok = done and cs is not None and abs(cs - REF) <= 1e-9 * abs(REF)
    return dict(
        run_id=uuid.uuid4().hex[:8], ranks=a.ranks, iters=a.iters,
        mtbf_mult="inf" if mult == float("inf") else mult,
        expected_failures_per_run=0 if mult == float("inf") else round(1 / mult, 3),
        seed=seed, attempts=attempts, failures=failures, completed=int(done),
        makespan=round(end - start, 4), lost_work_s=round(lost_wall, 4),
        final_attempt_s=round(final_len, 4), checksum_ok=int(ok),
        # ---- S02 columns (appended after the S01 rate_runs.csv columns) ----
        experiment=experiment, strategy=strategy,
        ckpt_interval=interval if strategy == "periodic" else "",
        ckpts=n_ckpts, ckpt_overhead_s=round(ckpt_cost, 4), restore_s=round(restore, 4),
        lost_compute_s=round(lost_compute, 4), lost_iters=int(lost_iters),
        detect_teardown_s=round(detect, 4), t_iter_s=round(T_ITER, 7),
        recovery_time_s_total=round(sum(recov), 4),
        recovery_time_s_mean=round(sum(recov) / len(recov), 4) if recov else "",
        final_interval=ck.get("final_interval", ""), N=a.n, fsync=a.fsync)


COLS = ["run_id", "ranks", "iters", "mtbf_mult", "expected_failures_per_run", "seed",
        "attempts", "failures", "completed", "makespan", "lost_work_s", "final_attempt_s",
        "checksum_ok", "experiment", "strategy", "ckpt_interval", "ckpts", "ckpt_overhead_s",
        "restore_s", "lost_compute_s", "lost_iters", "recovery_time_s_total",
        "recovery_time_s_mean", "final_interval", "N", "fsync", "detect_teardown_s",
        "t_iter_s"]

os.makedirs("results", exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)
# 1-rank failure-free reference checksum, same tolerance rule as S01
ref_path = f"results/ref_{a.n}_{a.iters}.txt"
if os.path.exists(ref_path):
    REF = float(open(ref_path).read())
else:
    _, _, _, REF, _ = run(1, [])
    open(ref_path, "w").write(repr(REF))
# time per iteration at this rank count, from one failure-free run without checkpoints
_, _, _, _, _ = run(a.ranks, [])  # warm-up
_p = subprocess.run(["mpirun", "--oversubscribe"] + EXTRA + ["-np", str(a.ranks), "./workload",
                    "--n", str(a.n), "--iters", str(a.iters)], capture_output=True, text=True)
T_ITER = float(re.search(r"time=(\S+)", _p.stdout).group(1)) / a.iters
print(f"time per iteration at {a.ranks} ranks: {T_ITER*1e3:.3f} ms", flush=True)

new = not os.path.exists(a.out)
f = open(a.out, "a", newline="")
w = csv.DictWriter(f, fieldnames=COLS)
if new:
    w.writeheader()


def log(row):
    w.writerow(row); f.flush()
    print({k: row[k] for k in ("experiment", "strategy", "ckpt_interval", "mtbf_mult", "seed",
                               "failures", "makespan", "ckpt_overhead_s", "lost_compute_s",
                               "checksum_ok")}, flush=True)


configs = [("none", "")] + [("periodic", k) for k in INTERVALS] + [("adaptive", "")]
if a.exp in ("overhead", "both"):
    # failure-free. Adaptive is told MTBF = 1 run length so it picks a realistic interval.
    for rep in range(a.repeats):
        for strat, k in configs:
            log(one_job(strat, k, float("inf"), rep, "overhead", adapt_mult=1.0))

if a.exp in ("failures", "both"):
    for mult in MULTS:
        for seed in range(a.seeds):
            for strat, k in configs:
                log(one_job(strat, k, mult, seed, "failures"))
f.close()
shutil.rmtree(CKPT_ROOT, ignore_errors=True)
print(f"\nDone. Rows appended to {a.out}")
