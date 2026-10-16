import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/runs.csv")
os.makedirs("results/plots", exist_ok=True)
df["frac"] = df.fail_iter / df.iters

# Sanity checks
done = df[df.completed == 1]
print("Total runs:", len(df))
print("Completed runs with WRONG checksum:", int((done.checksum_ok == 0).sum()))
print("Runs per (mode, ranks):")
print(df.groupby(["mode", "ranks"]).size().unstack(), "\n")

base = df[df["mode"] == "baseline"].groupby("ranks").makespan.agg(["mean", "std", "count"])
print("Baseline makespan (s):\n", base, "\n")

rst = df[df["mode"] == "restart"].copy()
rst["slowdown"] = rst.apply(lambda r: r.makespan / base.loc[r.ranks, "mean"], axis=1)
rst["overhead_s"] = rst.apply(lambda r: r.makespan - base.loc[r.ranks, "mean"], axis=1)

summ = rst.groupby(["ranks", "frac"]).agg(
    makespan_mean=("makespan", "mean"), makespan_std=("makespan", "std"),
    slowdown_mean=("slowdown", "mean"), slowdown_std=("slowdown", "std"),
    wasted_s_mean=("time_to_failure", "mean"), wasted_s_std=("time_to_failure", "std"),
    iters_lost=("iters_lost", "mean"), n=("makespan", "count")).reset_index()
summ.to_csv("results/summary.csv", index=False)
print(summ.round(3).to_string(index=False), "\n")

# Plot 1: slowdown vs failure position (restart from scratch)
plt.figure(figsize=(7, 4.5))
for r, g in summ.groupby("ranks"):
    plt.errorbar(g.frac * 100, g.slowdown_mean, yerr=g.slowdown_std, marker="o", capsize=3, label=f"{r} ranks")
plt.axhline(1.0, color="gray", ls="--", lw=1, label="no failure")
plt.xlabel("Failure position (% of run completed)")
plt.ylabel("Makespan / failure-free makespan")
plt.title("Restart-from-beginning: slowdown vs failure position")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/plots/slowdown_vs_failure_position.png", dpi=150); plt.close()

# Plot 2: lost work (wasted seconds) vs failure position
plt.figure(figsize=(7, 4.5))
for r, g in summ.groupby("ranks"):
    plt.errorbar(g.frac * 100, g.wasted_s_mean, yerr=g.wasted_s_std, marker="o", capsize=3, label=f"{r} ranks")
plt.xlabel("Failure position (% of run completed)")
plt.ylabel("Lost work: time spent before crash (s)")
plt.title("Lost work vs failure position")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/plots/lost_work_vs_failure_position.png", dpi=150); plt.close()

# Plot 3: scaling (failure-free)
plt.figure(figsize=(7, 4.5))
rk = base.index.values
plt.errorbar(rk, base["mean"], yerr=base["std"], marker="o", capsize=3, label="measured")
plt.plot(rk, base["mean"].iloc[0] / rk, "k--", lw=1, label="ideal (T1/p)")
plt.xscale("log", base=2); plt.yscale("log")
plt.xlabel("MPI ranks"); plt.ylabel("Makespan (s)")
plt.title("Strong scaling, failure-free baseline")
plt.legend(); plt.grid(alpha=.3, which="both"); plt.tight_layout()
plt.savefig("results/plots/strong_scaling.png", dpi=150); plt.close()

print("Saved plots in results/plots/ and results/summary.csv")
