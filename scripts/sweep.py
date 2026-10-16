import subprocess
ITERS = 5000
RANKS = [1, 2, 4, 8]
FAIL_FRACS = [0.25, 0.5, 0.75]
REPEATS = 3

def go(*args):
    subprocess.run(["python3", "scripts/run_one.py", *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL)

for r in RANKS:
    for rep in range(REPEATS):
        go("--mode", "baseline", "--ranks", r, "--iters", ITERS)
        for f in FAIL_FRACS:
            fi = int(ITERS * f)
            fr = r - 1 if r > 1 else 0
            for mode in ["norecovery", "restart"]:
                go("--mode", mode, "--ranks", r, "--iters", ITERS,
                   "--fail-iter", fi, "--fail-rank", fr)
        print(f"done ranks={r} rep={rep}", flush=True)
