import subprocess, time, csv, os, re, random, uuid

RANKS, N, ITERS = 4, 2000000, 2000
MULTS = [4.0, 2.0, 1.0, 0.5]   # MTBF as a multiple of run length (iterations)
REPEATS = 10
MAX_ATTEMPTS = 25

def run(fi=-1, fr=-1, ranks=RANKS):
    cmd = ["mpirun", "--oversubscribe", "-np", str(ranks), "./workload",
           "--n", str(N), "--iters", str(ITERS),
           "--fail-iter", str(fi), "--fail-rank", str(fr)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    t1 = time.time()
    m = re.search(r"RESULT .*checksum=(\S+)", p.stdout)
    return p.returncode, t0, t1, (float(m.group(1)) if m else None)

os.makedirs("results", exist_ok=True)
_, _, _, ref = run(ranks=1)   # 1-rank reference checksum for correctness

path = "results/rate_runs.csv"
cols = ["run_id", "ranks", "iters", "mtbf_mult", "expected_failures_per_run", "seed",
        "attempts", "failures", "completed", "makespan", "lost_work_s",
        "final_attempt_s", "checksum_ok"]
new = not os.path.exists(path)
f = open(path, "a", newline="")
w = csv.DictWriter(f, fieldnames=cols)
if new: w.writeheader()

def log(**kw):
    w.writerow(kw); f.flush(); print(kw, flush=True)

# failure-free baselines at this size
for i in range(3):
    rc, s, e, cs = run()
    log(run_id=uuid.uuid4().hex[:8], ranks=RANKS, iters=ITERS, mtbf_mult="inf",
        expected_failures_per_run=0, seed=i, attempts=1, failures=0, completed=int(rc == 0),
        makespan=round(e - s, 4), lost_work_s=0, final_attempt_s=round(e - s, 4),
        checksum_ok=int(cs is not None and abs(cs - ref) <= 1e-9 * abs(ref)))

for mult in MULTS:
    for seed in range(REPEATS):
        rng = random.Random(seed * 1000 + int(mult * 100))
        start = time.time()
        attempts = failures = 0
        lost = 0.0
        done, last_len, cs = False, 0.0, None
        while attempts < MAX_ATTEMPTS and not done:
            attempts += 1
            fi = int(rng.expovariate(1.0 / (mult * ITERS)))
            if fi >= ITERS: fi = -1
            fr = rng.randrange(RANKS)
            rc, s, e, cs = run(fi, fr)
            if rc == 0:
                done, last_len = True, e - s
            else:
                failures += 1
                lost += e - s
        end = time.time()
        ok = done and cs is not None and abs(cs - ref) <= 1e-9 * abs(ref)
        log(run_id=uuid.uuid4().hex[:8], ranks=RANKS, iters=ITERS, mtbf_mult=mult,
            expected_failures_per_run=round(1 / mult, 3), seed=seed, attempts=attempts,
            failures=failures, completed=int(done), makespan=round(end - start, 4),
            lost_work_s=round(lost, 4), final_attempt_s=round(last_len, 4), checksum_ok=int(ok))
f.close()
