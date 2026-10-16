import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)
df = pd.read_csv("results/rate_runs.csv")
df["mtbf_mult"] = df["mtbf_mult"].astype(str)
base = df[df.mtbf_mult == "inf"].makespan.mean()
d = df[df.mtbf_mult != "inf"].copy()
d["mtbf_mult"] = d.mtbf_mult.astype(float)
d["rate"] = 1 / d.mtbf_mult

print("Failure-free makespan (mean s):", round(base, 3))
print("Completed runs with wrong checksum:", int(((d.completed == 1) & (d.checksum_ok == 0)).sum()))
print("Runs that did NOT complete (hit attempt cap):", int((d.completed == 0).sum()), "\n")

g = d.groupby("rate").agg(runs=("seed", "count"), completed=("completed", "mean"),
    attempts=("attempts", "mean"), failures=("failures", "mean"),
    makespan=("makespan", "mean"), makespan_sd=("makespan", "std"),
    lost=("lost_work_s", "mean"), lost_sd=("lost_work_s", "std")).reset_index()
g["slowdown"] = g.makespan / base
g["lost_share"] = g.lost / g.makespan
print(g.round(3).to_string(index=False))
g.to_csv("results/rate_summary.csv", index=False)

for col, sd, yl, name in [("makespan", "makespan_sd", "Total makespan (s)", "rate_vs_makespan"),
                          ("lost", "lost_sd", "Lost work: time in crashed attempts (s)", "rate_vs_lost_work")]:
    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(g.rate, g[col], yerr=g[sd], marker="o", capsize=3)
    if col == "makespan":
        plt.axhline(base, color="gray", ls="--", label="failure-free")
        plt.legend()
    plt.xlabel("Failure rate (expected failures per failure-free run)")
    plt.ylabel(yl); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"results/plots/{name}.png", dpi=150); plt.close()
print("\nSaved plots to results/plots/ and results/rate_summary.csv")
