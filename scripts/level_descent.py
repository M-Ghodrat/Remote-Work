import csv
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(os.environ.get("REMOTE_WORK_OUTPUT_DIR", Path(__file__).parent))
rows = list(csv.DictReader(open(HERE / "consolidated_outcomes.csv")))
N = len(rows)

LEVELS = {
  "Individual": ["emp_job_satisfaction","emp_engagement","emp_wellbeing","emp_burnout","emp_stress",
                 "emp_work_life","emp_isolation","emp_autonomy","emp_flexibility","emp_belonging",
                 "emp_technostress","emp_turnover_intention","org_job_performance","org_productivity"],
  "Team":       ["org_collaboration","org_team_effectiveness","org_innovation","org_project_success","org_leadership"],
  "Firm":       ["org_firm_performance","org_financial","org_cost","org_real_estate",
                 "org_retention_rate","org_absenteeism","org_culture","org_commitment"],
}
def short(c): return c.replace("emp_","").replace("org_","").replace("_"," ")
prev = {c: sum(1 for r in rows if r.get(c)=="1") for lvl in LEVELS.values() for c in lvl}

print("LEVEL ASSIGNMENTS:")
for lvl, cols in LEVELS.items():
    print(f"\n{lvl}:")
    for c in sorted(cols, key=lambda x:-prev[x]):
        print(f"   {short(c):22s} {100*prev[c]/N:4.0f}%")

# per-level mean coverage and spread (min-max), but we show mean as the bar
colors = {"Individual":"#185FA5","Team":"#854F0B","Firm":"#A32D2D"}
fig, ax = plt.subplots(figsize=(9,6))
xs = list(LEVELS.keys())
means = [np.mean([100*prev[c]/N for c in LEVELS[l]]) for l in xs]
bars = ax.bar(xs, means, color=[colors[l] for l in xs], width=0.6, zorder=3)
# overlay individual outcome dots so the reader sees the spread without a variance chart
for i,l in enumerate(xs):
    ys = [100*prev[c]/N for c in LEVELS[l]]
    ax.scatter([i]*len(ys), ys, color="white", edgecolors=colors[l], s=40, zorder=4, linewidths=1.2)
ax.set_ylabel("% of 387 papers engaging the outcome")
ax.set_title("Coverage descends across organizational levels\n(bars = mean coverage; dots = individual outcomes)")
for b,m in zip(bars,means):
    ax.text(b.get_x()+b.get_width()/2, m+1.5, f"{m:.0f}%", ha="center", fontsize=11, zorder=5)
ax.set_ylim(0,75); ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
out = HERE/"outcome_figures"; out.mkdir(exist_ok=True)
fig.savefig(out/"level_descent.png", dpi=150, bbox_inches="tight")
fig.savefig(out/"level_descent.svg", bbox_inches="tight")
print(f"\nMean coverage by level: " + ", ".join(f"{l}={m:.0f}%" for l,m in zip(xs,means)))
print(f"saved {out/'level_descent.png'}")
