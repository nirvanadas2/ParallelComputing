"""S03 analysis: summary table and plots from results/recovery_runs.csv.

Outputs:
  results/recovery_summary.csv
  results/plots/recovery_scenario_vs_recovery_time.png   (expected graph 1)
  results/plots/recovery_time_breakdown.png              (where a recovery's time goes)
  results/plots/recovery_overhead_vs_resilience.png      (expected graph 2)
  results/plots/recovery_lost_work.png                   (expected graph 3)

Metrics (spec section 3, S03 definitions):
  recovery time   kill of the failing process -> compute resumes (fail -> next ready event)
  availability    failure-free makespan of A (no checkpoints, no spares) / actual makespan
  lost compute    iterations redone x time per iteration
  overhead        failure-free makespan of the strategy vs A, in %; C also idles S spares
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/recovery_runs.csv"
os.makedirs("results/plots", exist_ok=True)

# palette: categorical slots 1-3 (fixed order) = strategies A, B, C; slots 4-7 = breakdown parts
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
C4, C5, C6, C7 = "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, INK2, GRID, REFC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": REFC, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 8, "legend.frameon": False, "font.size": 10})
plt.rcParams["axes.axisbelow"] = True

STRATS = ["restart", "rollback", "ulfm"]
NAMES = {"restart": "A  restart from scratch", "rollback": "B  global rollback",
         "ulfm": "C  ULFM + spare"}
SHORT = {"restart": "A", "rollback": "B", "ulfm": "C"}
COL = {"restart": C1, "rollback": C2, "ulfm": C3}
SCEN = ["single", "multi", "rank0", "ckpt"]
SCEN_LABEL = {"single": "single failure", "multi": "3 failures", "rank0": "rank 0 fails",
              "ckpt": "fails during\ncheckpoint"}

df = pd.read_csv(path)
print("Rows:", len(df))
bad = df[(df.completed == 1) & (df.checksum_ok == 0)]
print("Completed runs with WRONG checksum:", len(bad))
print("Runs that did NOT complete:", int((df.completed == 0).sum()))
short = df[(df.scenario != "none") & (df.failures < df.expected_failures_per_run)]
print("Runs with fewer failures than planned:", len(short), "\n")

ok = df[df.completed == 1]
base = ok[ok.scenario == "none"]
if not len(base[base.strategy == "restart"]):
    raise SystemExit("need failure-free 'restart' baselines (scenario none) for availability")
t_ff = base[base.strategy == "restart"].makespan.mean()
ok = ok.assign(availability=t_ff / ok.makespan)

# ---------------- summary table ----------------
agg = dict(runs=("run_id", "count"), failures=("failures", "mean"),
           makespan=("makespan", "mean"), makespan_sd=("makespan", "std"),
           availability=("availability", "mean"), recovery_s=("recovery_time_s_mean", "mean"),
           recovery_sd=("recovery_time_s_mean", "std"), detect_s=("detect_s_mean", "mean"),
           relaunch_s=("relaunch_s_mean", "mean"), reconf_s=("reconf_s_mean", "mean"),
           restore_s=("restore_s_mean", "mean"), lost_compute_s=("lost_compute_s", "mean"),
           lost_iters=("lost_iters", "mean"), ckpt_overhead_s=("ckpt_overhead_s", "mean"),
           relaunches=("relaunches", "mean"), spare_fraction=("spare_fraction", "first"))
summ = ok.groupby(["scenario", "strategy"]).agg(**agg).reset_index()
ff = base.groupby("strategy").makespan.mean()
summ["failure_free_overhead_pct"] = summ.strategy.map(100 * (ff - t_ff) / t_ff)
summ.to_csv("results/recovery_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 30):
    print(summ.round(3).to_string(index=False))


def get(scen, strat, col):
    r = summ[(summ.scenario == scen) & (summ.strategy == strat)]
    return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan


def grouped_bars(ax, col, fmt):
    scen = [s for s in SCEN if s in set(summ.scenario)]
    x = np.arange(len(scen))
    w = 0.26
    for i, st in enumerate(STRATS):
        vals = [get(s, st, col) for s in scen]
        xs = x + (i - 1) * w
        ax.bar(xs, np.nan_to_num(vals), w - 0.03, color=COL[st], label=NAMES[st], zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, (0 if np.isnan(v) else v), " n/a" if np.isnan(v) else fmt(v),
                    ha="center", va="bottom", fontsize=8, color=INK2, rotation=90)
    ax.set_xticks(x, [SCEN_LABEL[s] for s in scen])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=3, fontsize=9)


# ---------------- 1. failure scenario vs recovery time ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
grouped_bars(ax, "recovery_s", lambda v: f"{v:.2f} s")
ax.set_ylabel("recovery time per failure (s)\nkill -> compute resumes")
ax.set_ylim(0, np.nanmax(summ.recovery_s) * 1.35)
fig.tight_layout()
fig.savefig("results/plots/recovery_scenario_vs_recovery_time.png", dpi=150)
plt.close(fig)

# ---------------- 1b. where a recovery's time goes (single failure) ----------------
sc = "single" if "single" in set(summ.scenario) else summ.scenario.iloc[0]
parts = [("detect_s", "detect (+ teardown for A/B)", C4), ("relaunch_extra", "relaunch + MPI startup", C5),
         ("reconf_s", "revoke + shrink + reassign", C6), ("restore_s", "read checkpoint", C7),
         ("other", "other (resync barrier, logging)", "#c3c2b7")]
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, st in enumerate(STRATS):
    vals = {k: get(sc, st, k) for k in ("detect_s", "relaunch_s", "reconf_s", "restore_s")}
    vals = {k: (0.0 if np.isnan(v) else v) for k, v in vals.items()}
    vals["relaunch_extra"] = max(0.0, vals["relaunch_s"] - vals["restore_s"])
    total = get(sc, st, "recovery_s")
    known = vals["detect_s"] + vals["relaunch_extra"] + vals["reconf_s"] + vals["restore_s"]
    vals["other"] = max(0.0, (0.0 if np.isnan(total) else total) - known)
    left = 0.0
    for key, lab, c in parts:
        v = vals[key]
        ax.barh(i, v, left=left, color=c, height=0.55, edgecolor=SURF, linewidth=2,
                label=lab if i == 0 else None, zorder=2)
        left += v
    ax.text(left, i, f"  {get(sc, st, 'recovery_s'):.3f} s", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(STRATS)), [NAMES[s] for s in STRATS])
ax.invert_yaxis()
ax.set_xlabel(f"seconds per failure ({SCEN_LABEL[sc]} scenario)")
ax.set_xlim(0, np.nanmax(summ[summ.scenario == sc].recovery_s) * 1.25)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig("results/plots/recovery_time_breakdown.png", dpi=150)
plt.close(fig)

# ---------------- 2. overhead vs resilience ----------------
fail_rows = ok[ok.scenario != "none"]
avail = fail_rows.groupby("strategy").availability.mean()
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for st in STRATS:
    if st not in avail.index or st not in ff.index:
        continue
    xo, ya = 100 * (ff[st] - t_ff) / t_ff, avail[st]
    ax.scatter([xo], [ya], s=90, color=COL[st], edgecolor=SURF, linewidth=2, zorder=3)
    note = ""
    if st == "ulfm":
        fr = summ[summ.strategy == "ulfm"].spare_fraction
        note = (f"\n(+ idle spares: {100*fr.min():.0f}% of processes"
                + (f",\n {100*fr.max():.0f}% in the 3-failure runs)" if fr.max() > fr.min() else ")"))
    ax.annotate(NAMES[st] + note, (xo, ya), textcoords="offset points", xytext=(10, -4),
                fontsize=9, color=INK)
ax.set_xlabel("failure-free overhead vs A (% of makespan)")
ax.set_ylabel("availability under failures\n(failure-free makespan / actual)")
ax.set_ylim(0, 1.05)
xl = ax.get_xlim()
ax.set_xlim(min(xl[0], -1), max(xl[1], 5) * 1.6)
fig.tight_layout()
fig.savefig("results/plots/recovery_overhead_vs_resilience.png", dpi=150)
plt.close(fig)

# ---------------- 3. lost work ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
grouped_bars(ax, "lost_compute_s", lambda v: f"{v:.2f} s")
ax.set_ylabel("lost compute per run (s)\niterations redone x time per iteration")
ax.set_ylim(0, np.nanmax(summ.lost_compute_s) * 1.35 + 1e-9)
fig.tight_layout()
fig.savefig("results/plots/recovery_lost_work.png", dpi=150)
plt.close(fig)
print("\nPlots written to results/plots/recovery_*.png")
