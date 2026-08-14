import csv
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(os.environ.get("REMOTE_WORK_OUTPUT_DIR", Path(__file__).parent))
rows = list(csv.DictReader(open(HERE / "consolidated_outcomes.csv")))
N = len(rows)

# all real outcomes (exclude summary flags)
ALL = [c for c in rows[0] if (c.startswith("emp_") or c.startswith("org_"))
       and c not in ("emp_engaged","org_engaged")]
prev = {c: 100*sum(1 for r in rows if r.get(c)=="1")/N for c in ALL}
corpus_avg = np.mean(list(prev.values()))

# the hard firm bottom-line outcomes
FIRM = ["org_firm_performance","org_financial","org_cost","org_real_estate",
        "org_retention_rate","org_absenteeism"]
def short(c): return c.replace("org_","").replace("emp_","").replace("_"," ")

order = sorted(FIRM, key=lambda c: prev[c])
vals = [prev[c] for c in order]
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(range(len(order)), vals, color="#A32D2D", zorder=3, height=0.6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([short(c) for c in order])
for i,v in enumerate(vals): ax.text(v+0.4, i, f"{v:.0f}%", va="center", fontsize=10)
ax.axvline(corpus_avg, color="#444441", linestyle="--", linewidth=1.5, zorder=4)
ax.text(corpus_avg+0.4, len(order)-0.4, f"corpus average\nacross all outcomes ({corpus_avg:.0f}%)",
        fontsize=9, color="#444441", va="top")
ax.set_xlabel("% of 387 papers engaging the outcome")
ax.set_title("Firm bottom-line outcomes sit far below typical coverage")
ax.set_xlim(0, max(corpus_avg, max(vals))+10); ax.grid(axis="x", alpha=0.3, zorder=0)
fig.tight_layout()
out = HERE/"outcome_figures"; out.mkdir(exist_ok=True)
fig.savefig(out/"firm_gap.png", dpi=150, bbox_inches="tight")
fig.savefig(out/"firm_gap.svg", bbox_inches="tight")
print(f"corpus average across {len(ALL)} outcomes: {corpus_avg:.1f}%")
print("firm outcomes:")
for c in order: print(f"  {short(c):20s} {prev[c]:.0f}%")
print(f"saved {out/'firm_gap.png'}")
