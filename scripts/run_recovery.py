"""S03 experiment runner: recovery strategies under several failure scenarios.

Strategies (all run the same binary, ./workload_ft, so all log the same timestamps):
  A  restart     global restart from scratch: relaunch mpirun, start at iteration 0
  B  rollback    global rollback: relaunch mpirun with --restart, resume from last checkpoint
  C  ulfm        ULFM in-place recovery: p compute + S spare processes; a spare replaces the
                 dead process, everyone reloads the last checkpoint, no relaunch. If spares
                 run out the job aborts and is relaunched with --restart (falls back to B).

Scenarios (one failure plan per seed, identical for every strategy = common random numbers):
  single  one failure, random non-zero position
  multi   three failures, random positions (C gets 3 spares)
  rank0   one failure of position 0, which writes commit.txt
  ckpt    one failure while a checkpoint is being written (A has no checkpoints: skipped)
  none    failure-free baseline (overhead of being resilient)

Failure k hits at iteration resume_k + F_k * (iters - resume_k), F_k ~ U(0.1, 0.9)
(U(0.1, 0.7) for ckpt), where resume_k is where the run restarted before failure k.
So every strategy sees exactly the planned number of failures.

Usage (repo root, Open MPI 5 with ULFM on PATH):
  mpicc -O2 -o workload_ft src/workload_ft.c -lm
  python3 scripts/run_recovery.py            # full: 4 scenarios x 3 strategies x 5 seeds + baselines
  python3 scripts/run_recovery.py --quick    # smoke test, a few minutes
Extra mpirun flags (e.g. --allow-run-as-root) via env MPIRUN_EXTRA.
"""
import argparse, csv, os, random, re, shutil, subprocess, time, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--ranks", type=int, default=4, help="compute positions p")
ap.add_argument("--n", type=int, default=2000000)
ap.add_argument("--iters", type=int, default=10000, help="S02 used 10000")
ap.add_argument("--interval", type=int, default=300, help="checkpoint interval K for B and C")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--repeats", type=int, default=3, help="failure-free baseline repeats")
ap.add_argument("--scenarios", default="none,single,multi,rank0,ckpt")
ap.add_argument("--strategies", default="restart,rollback,ulfm")
ap.add_argument("--multi-failures", type=int, default=3)
ap.add_argument("--max-attempts", type=int, default=10)
ap.add_argument("--timeout", type=int, default=600, help="seconds per mpirun before it is killed")
ap.add_argument("--fsync", type=int, default=1)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results/recovery_runs.csv")
a = ap.parse_args()
if a.quick:
    a.iters, a.seeds, a.repeats, a.interval = 2000, 1, 1, 200

EXTRA = os.environ.get("MPIRUN_EXTRA", "").split()
CKPT_ROOT = "ckpt_tmp_s03"
SCEN_CODE = {"none": 0, "single": 1, "multi": 2, "rank0": 3, "ckpt": 4}


def mpirun(nprocs, args, ulfm=False):
    cmd = (["mpirun", "--oversubscribe"] + (["--with-ft", "ulfm"] if ulfm else []) + EXTRA +
           ["-np", str(nprocs), "./workload_ft", "--n", str(a.n), "--iters", str(a.iters)] +
           [str(x) for x in args])
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        out, rc = p.stdout, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")), -999
        subprocess.run(["pkill", "-9", "-f", "workload_ft"], capture_output=True)
    t1 = time.time()
    m = re.search(r"RESULT .*checksum=(\S+)", out)
    return rc, t0, t1, (float(m.group(1)) if m else None), out


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


def make_plan(scenario, seed):
    """List of (fraction, position) failures for this scenario and seed."""
    rng = random.Random(seed * 1000 + SCEN_CODE[scenario])
    if scenario == "single":
        return [(rng.uniform(0.1, 0.9), rng.randrange(1, a.ranks))]
    if scenario == "multi":
        return [(rng.uniform(0.1, 0.9), rng.randrange(a.ranks)) for _ in range(a.multi_failures)]
    if scenario == "rank0":
        return [(rng.uniform(0.1, 0.9), 0)]
    if scenario == "ckpt":
        return [(rng.uniform(0.1, 0.7), rng.randrange(a.ranks))]
    return []


def one_job(strategy, scenario, seed):
    plan = make_plan(scenario, seed)
    spares = max(1, len(plan)) if strategy == "ulfm" else 0
    d = os.path.join(CKPT_ROOT, uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    base = ["--ckpt-dir", d, "--ckpt-fsync", a.fsync, "--restart"]
    if strategy == "restart":
        base += ["--ckpt-mode", "none", "--ft", "relaunch"]
    elif strategy == "rollback":
        base += ["--ckpt-mode", "periodic", "--ckpt-interval", a.interval, "--ft", "relaunch"]
    else:
        base += ["--ckpt-mode", "periodic", "--ckpt-interval", a.interval,
                 "--ft", "ulfm", "--spares", spares]
    if scenario == "ckpt":
        base += ["--fail-in-ckpt", 1]

    remaining = list(plan)
    start = time.time()
    attempts, lost_wall, final_len, cs, done = 0, 0.0, 0.0, None, False
    exits = []          # (exit epoch, attempt crashed?) per attempt
    while attempts < a.max_attempts and not done:
        attempts += 1
        args = list(base)
        if remaining:
            args += ["--fail-plan", ",".join(f"{f:.6f}:{q}" for f, q in remaining)]
        n_before = len(read_events(d))
        rc, s, e, ck, out = mpirun(a.ranks + spares, args, ulfm=(strategy == "ulfm"))
        evs = read_events(d)[n_before:]
        n_fail = sum(1 for x in evs if x["kind"] == "fail")
        remaining = remaining[n_fail:]
        if ck is not None:          # RESULT printed = job finished
            done, final_len, cs = True, e - s, ck
            exits.append((e, False))
        else:
            lost_wall += e - s
            exits.append((e, True))
            if n_fail == 0 and rc != -999:
                print(f"  attempt crashed without a planned failure (rc={rc}):\n{out[-500:]}")
    end = time.time()

    evs = read_events(d)
    shutil.rmtree(d, ignore_errors=True)
    fails = [x for x in evs if x["kind"] == "fail"]
    readies = [x for x in evs if x["kind"] == "ready"]
    rec, det, relaunch, reconf, restore, lost_it = [], [], [], [], [], 0
    for f in fails:
        nxt = next((r for r in readies if r["epoch"] > f["epoch"]), None)
        if nxt is None:
            continue
        rec.append(nxt["epoch"] - f["epoch"])
        restore.append(nxt["restore_s"])
        lost_it += max(0, int(f["iter"] - nxt["resumed_from"]))
        if nxt.get("inplace"):
            det.append(nxt["detect_epoch"] - f["epoch"])
            reconf.append(nxt["reconf_s"])
            relaunch.append(0.0)
        else:
            ex = next((t for t, crashed in exits if crashed and t >= f["epoch"]), None)
            if ex is not None:
                det.append(ex - f["epoch"])          # detection + teardown of the dead job
                relaunch.append(nxt["epoch"] - ex)   # relaunch + MPI startup + restore
            reconf.append(0.0)
    mean = lambda v: round(sum(v) / len(v), 4) if v else ""
    ok = done and cs is not None and abs(cs - REF) <= 1e-9 * abs(REF)
    commits = [x for x in evs if x["kind"] == "commit"]
    return dict(
        run_id=uuid.uuid4().hex[:8], ranks=a.ranks, iters=a.iters,
        mtbf_mult="inf" if not plan else "",
        expected_failures_per_run=len(plan), seed=seed, attempts=attempts,
        failures=len(fails), completed=int(done), makespan=round(end - start, 4),
        lost_work_s=round(lost_wall, 4), final_attempt_s=round(final_len, 4),
        checksum_ok=int(ok),
        # ---- S03 columns ----
        scenario=scenario, strategy=strategy,
        ckpt_interval=a.interval if strategy != "restart" else "",
        spares=spares, procs=a.ranks + spares,
        spare_fraction=round(spares / (a.ranks + spares), 4),
        ckpts=len(commits), ckpt_overhead_s=round(sum(x["cost_s"] for x in commits), 4),
        lost_iters=lost_it, lost_compute_s=round(lost_it * T_ITER, 4),
        recovery_time_s_total=round(sum(rec), 4), recovery_time_s_mean=mean(rec),
        detect_s_mean=mean(det), relaunch_s_mean=mean(relaunch),
        reconf_s_mean=mean(reconf), restore_s_mean=mean(restore),
        relaunches=attempts - 1, N=a.n, fsync=a.fsync, t_iter_s=round(T_ITER, 7),
        fail_plan=" ".join(f"{f:.3f}:{q}" for f, q in plan))


COLS = ["run_id", "ranks", "iters", "mtbf_mult", "expected_failures_per_run", "seed",
        "attempts", "failures", "completed", "makespan", "lost_work_s", "final_attempt_s",
        "checksum_ok",
        "scenario", "strategy", "ckpt_interval", "spares", "procs", "spare_fraction", "ckpts",
        "ckpt_overhead_s", "lost_iters", "lost_compute_s", "recovery_time_s_total",
        "recovery_time_s_mean", "detect_s_mean", "relaunch_s_mean", "reconf_s_mean",
        "restore_s_mean", "relaunches", "N", "fsync", "t_iter_s", "fail_plan"]

if not os.path.exists("./workload_ft"):
    raise SystemExit("build first: mpicc -O2 -o workload_ft src/workload_ft.c -lm")
os.makedirs("results", exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)
ref_path = f"results/ref_{a.n}_{a.iters}.txt"
if os.path.exists(ref_path):
    REF = float(open(ref_path).read())
else:
    _, _, _, REF, _ = mpirun(1, [])
    open(ref_path, "w").write(repr(REF))
mpirun(a.ranks, [])  # warm-up
_, _, _, _, _out = mpirun(a.ranks, [])
T_ITER = float(re.search(r"time=(\S+)", _out).group(1)) / a.iters
print(f"time per iteration at {a.ranks} ranks: {T_ITER*1e3:.3f} ms", flush=True)

new = not os.path.exists(a.out)
f = open(a.out, "a", newline="")
w = csv.DictWriter(f, fieldnames=COLS)
if new:
    w.writeheader()


def log(row):
    w.writerow(row); f.flush()
    print({k: row[k] for k in ("scenario", "strategy", "seed", "failures", "attempts", "makespan",
                               "recovery_time_s_mean", "lost_compute_s", "checksum_ok")},
          flush=True)


strategies = a.strategies.split(",")
for scen in a.scenarios.split(","):
    n_runs = a.repeats if scen == "none" else a.seeds
    for seed in range(n_runs):
        for strat in strategies:
            if scen == "ckpt" and strat == "restart":
                continue  # no checkpoints, nothing to fail inside
            log(one_job(strat, scen, seed))
f.close()
shutil.rmtree(CKPT_ROOT, ignore_errors=True)
print(f"\nDone. Rows appended to {a.out}")
