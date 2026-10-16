import subprocess, time, csv, os, re, argparse, uuid

def run(ranks, N, iters, fail_iter=-1, fail_rank=-1):
    cmd = ["mpirun", "--oversubscribe", "-np", str(ranks), "./workload",
           "--n", str(N), "--iters", str(iters),
           "--fail-iter", str(fail_iter), "--fail-rank", str(fail_rank)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    t1 = time.time()
    m = re.search(r"RESULT .*checksum=(\S+)", p.stdout)
    return p.returncode, t0, t1, (float(m.group(1)) if m else None)

def reference(N, iters):
    path = f"results/ref_{N}_{iters}.txt"
    if os.path.exists(path):
        return float(open(path).read())
    rc, _, _, cs = run(1, N, iters)
    open(path, "w").write(repr(cs))
    return cs

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["baseline", "norecovery", "restart"], required=True)
ap.add_argument("--ranks", type=int, default=4)
ap.add_argument("--n", type=int, default=2000000)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--fail-iter", type=int, default=-1)
ap.add_argument("--fail-rank", type=int, default=0)
a = ap.parse_args()

os.makedirs("results", exist_ok=True)
ref = reference(a.n, a.iters)
fi = a.fail_iter if a.mode != "baseline" else -1

rc1, s1, e1, cs1 = run(a.ranks, a.n, a.iters, fi, a.fail_rank)
failed = rc1 != 0
t_end, cs_final, completed = e1, cs1, (not failed)
t_to_fail = (e1 - s1) if failed else ""
iters_lost = fi if failed else 0
if failed and a.mode == "restart":
    rc2, s2, e2, cs2 = run(a.ranks, a.n, a.iters)   # restart from the beginning, no failure
    t_end, cs_final, completed = e2, cs2, (rc2 == 0)
recovery_gap = (s2 - e1) if (failed and a.mode == "restart") else ""

ok = (cs_final is not None) and abs(cs_final - ref) <= 1e-9 * abs(ref)
row = dict(run_id=uuid.uuid4().hex[:8], mode=a.mode, ranks=a.ranks, N=a.n, iters=a.iters,
           fail_iter=fi, fail_rank=a.fail_rank, failure_occurred=int(failed),
           time_to_failure=t_to_fail, iters_lost=iters_lost,
           relaunch_gap=recovery_gap, completed=int(completed),
           makespan=round(t_end - s1, 4), checksum_ok=int(ok))
path = "results/runs.csv"
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if new: w.writeheader()
    w.writerow(row)
print(row)
