"""S02 analysis: summary tables and plots from results/ckpt_runs.csv.

Outputs:
  results/ckpt_summary.csv
  results/plots/ckpt_interval_vs_overhead.png
  results/plots/ckpt_interval_vs_lost_work.png
  results/plots/ckpt_total_time.png
  results/plots/ckpt_failure_cost_breakdown.png
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def k_axis(ax, ks):
    """log-scaled interval axis with plain tick labels at the tested intervals"""
    ks = sorted(int(k) for k in ks)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in ks]))
    ax.xaxis.set_minor_locator(NullLocator())

path = sys.argv[1] if len(sys.argv) > 1 else "results/ckpt_runs.csv"
os.makedirs("results/plots", exist_ok=True)

# palette: categorical slots 1-3 in fixed order, neutral ink for text and references
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, REFC = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": REFC, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 7, "legend.frameon": False, "font.size": 10})

df = pd.read_csv(path)
df["mtbf_mult"] = df["mtbf_mult"].astype(str)
print("Rows:", len(df))
bad = df[(df.completed == 1) & (df.checksum_ok == 0)]
print("Completed runs with WRONG checksum:", len(bad))
print("Runs that did NOT complete (hit attempt cap):", int((df.completed == 0).sum()), "\n")


def label(r):
    return f"periodic K={int(r.ckpt_interval)}" if r.strategy == "periodic" else r.strategy


df["config"] = df.apply(label, axis=1)

# ---------------- overhead experiment (failure-free) ----------------
ov = df[df.experiment == "overhead"].copy()
t0 = ov[ov.strategy == "none"].makespan.mean()
t0_compute = ov[ov.strategy == "none"].final_attempt_s.mean()
ov["overhead_pct"] = 100 * (ov.makespan - t0) / t0
ov["ckpt_write_pct"] = 100 * ov.ckpt_overhead_s / t0
per = ov[ov.strategy == "periodic"]
# Cost of one checkpoint, per interval. Back-to-back checkpoints (small K) cost more each,
# because fsyncs queue up, so the mean over all K overstates C at realistic intervals.
# Young's formula uses the median, which is robust to those small-K outliers.
per = per.assign(c_ms=1e3 * per.ckpt_overhead_s / per.ckpts.replace(0, np.nan))
c_by_k = per.groupby("ckpt_interval").c_ms.mean()
C_per_ckpt = per.c_ms.median() / 1e3  # seconds
t_iter = df.t_iter_s.dropna().mean() if "t_iter_s" in df else t0_compute / df.iters.iloc[0]
print(f"Failure-free makespan, no checkpoint: {t0:.3f} s")
print("Cost of one checkpoint (ms) by interval K:")
print(c_by_k.round(2).to_string())
print(f"Checkpoint cost C used for Young's formula (median): {C_per_ckpt*1e3:.1f} ms "
      f"(= {C_per_ckpt / t_iter:.1f} iterations at {t_iter*1e3:.3f} ms/iter)\n")

og = ov.groupby(["strategy", "ckpt_interval"], dropna=False).agg(
    runs=("seed", "count"), makespan=("makespan", "mean"), makespan_sd=("makespan", "std"),
    ckpts=("ckpts", "mean"), ckpt_write_s=("ckpt_overhead_s", "mean"),
    overhead_pct=("overhead_pct", "mean"), overhead_pct_sd=("overhead_pct", "std"),
    ckpt_write_pct=("ckpt_write_pct", "mean")).reset_index()
print("Overhead experiment (failure-free):")
print(og.round(3).to_string(index=False), "\n")

pg = og[og.strategy == "periodic"].sort_values("ckpt_interval")
ad = og[og.strategy == "adaptive"]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.errorbar(pg.ckpt_interval, pg.overhead_pct, yerr=pg.overhead_pct_sd, color=C1, marker="o",
            capsize=3, label="periodic: total slowdown vs no checkpoint")
ax.plot(pg.ckpt_interval, pg.ckpt_write_pct, color=C2, marker="s",
        label="periodic: measured checkpoint write time")
if len(ad):
    ax.axhline(ad.overhead_pct.iloc[0], color=C3, ls="--",
               label=f"adaptive: total slowdown ({ad.ckpts.iloc[0]:.0f} checkpoints)")
ax.axhline(0, color=REFC, lw=1)
k_axis(ax, pg.ckpt_interval)
ax.set_xlabel("Checkpoint interval K (iterations, log scale)")
ax.set_ylabel("Overhead (% of failure-free time)")
ax.set_title("Checkpoint overhead vs interval (no failures)", loc="left")
ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig("results/plots/ckpt_interval_vs_overhead.png", dpi=150)
plt.close(fig)

# ---------------- failures experiment ----------------
fl = df[df.experiment == "failures"].copy()
fl["mult"] = fl.mtbf_mult.astype(float)
fl["rate"] = 1 / fl.mult
fl["detect_teardown_s"] = fl.get("detect_teardown_s", 0)
fl["restart_s"] = fl.recovery_time_s_total  # relaunch + MPI init + restore, all failures
fg = fl.groupby(["rate", "strategy", "ckpt_interval"], dropna=False).agg(
    runs=("seed", "count"), completed=("completed", "mean"), failures=("failures", "mean"),
    makespan=("makespan", "mean"), makespan_sd=("makespan", "std"),
    ckpt_overhead_s=("ckpt_overhead_s", "mean"), lost_compute_s=("lost_compute_s", "mean"),
    lost_compute_sd=("lost_compute_s", "std"), lost_iters=("lost_iters", "mean"),
    detect_teardown_s=("detect_teardown_s", "mean"), restart_s=("restart_s", "mean"),
    recovery_time_s=("recovery_time_s_mean", "mean"),
    final_interval=("final_interval", "mean")).reset_index()
fg["slowdown"] = fg.makespan / t0
fg.to_csv("results/ckpt_summary.csv", index=False)
print("Failures experiment (means over seeds):")
print(fg.round(3).to_string(index=False), "\n")

rates = sorted(fl.rate.unique())
colors = {r: c for r, c in zip(rates, [C1, C2, C3])}
young = {}
for r in rates:
    mtbf_iters = fl[fl.rate == r].mult.iloc[0] * fl.iters.iloc[0]
    young[r] = np.sqrt(2 * (C_per_ckpt / t_iter) * mtbf_iters)
    print(f"Young's optimum interval at {r:g} failures/run: K* = {young[r]:.0f} iterations")

for col, sdcol, ylab, title, fname in [
        ("lost_compute_s", "lost_compute_sd", "Lost compute per run (s)",
         "Lost work vs checkpoint interval", "ckpt_interval_vs_lost_work"),
        ("makespan", "makespan_sd", "Total time to finish (s)",
         "Total time vs checkpoint interval", "ckpt_total_time")]:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for r in rates:
        g = fg[(fg.rate == r) & (fg.strategy == "periodic")].sort_values("ckpt_interval")
        c = colors[r]
        ax.errorbar(g.ckpt_interval, g[col], yerr=g[sdcol], color=c, marker="o", capsize=3,
                    label=f"periodic, {r:g} failures/run")
        none = fg[(fg.rate == r) & (fg.strategy == "none")][col]
        if len(none):
            ax.axhline(none.iloc[0], color=c, ls="--", lw=1.4,
                       label=f"no checkpoint, {r:g} failures/run")
        if col == "makespan":
            adp = fg[(fg.rate == r) & (fg.strategy == "adaptive")]
            if len(adp):
                ax.plot([adp.final_interval.iloc[0]], [adp[col].iloc[0]], marker="D", ms=8,
                        color=c, mfc="#fcfcfb", mew=2, ls="none",
                        label=f"adaptive, {r:g} failures/run")
            ax.axvline(young[r], color=c, ls=":", lw=1.2)
    if col == "makespan":
        ax.axhline(t0, color=REFC, lw=1.2, label="failure-free, no checkpoint")
    k_axis(ax, fl[fl.strategy == "periodic"].ckpt_interval.dropna().unique())
    ax.set_xlabel("Checkpoint interval K (iterations, log scale)")
    ax.set_ylabel(ylab)
    ax.set_title(title + (" (dotted: Young's K*)" if col == "makespan" else ""), loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"results/plots/{fname}.png", dpi=150); plt.close(fig)

# breakdown of where failure cost goes, highest failure rate, per strategy
r = rates[-1]
sub = fg[fg.rate == r].copy()
sub["config"] = sub.apply(lambda x: f"K={int(x.ckpt_interval)}" if x.strategy == "periodic"
                          else x.strategy, axis=1)
order = (["none"] + [f"K={k}" for k in sorted(sub[sub.strategy == "periodic"]
                                                .ckpt_interval.astype(int))] + ["adaptive"])
sub = sub.set_index("config").loc[[o for o in order if o in set(sub.config)]]
parts = [("ckpt_overhead_s", "checkpoint writing", C1), ("lost_compute_s", "lost compute", C2),
         ("detect_teardown_s", "failure detection + teardown", C3),
         ("restart_s", "relaunch + restore", "#8a8984")]
fig, ax = plt.subplots(figsize=(7.8, 4.6))
bottom = np.zeros(len(sub))
for col, lab, c in parts:
    ax.bar(sub.index, sub[col], bottom=bottom, color=c, label=lab, edgecolor="#fcfcfb",
           linewidth=1.5, width=0.7)
    bottom += sub[col].values
ax.set_ylabel("Seconds added to the run (mean)")
ax.set_title(f"Where the extra time goes, {r:g} expected failures/run", loc="left")
ax.legend(fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("results/plots/ckpt_failure_cost_breakdown.png", dpi=150)
plt.close(fig)
print("\nSaved plots to results/plots/ and results/ckpt_summary.csv")
