import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "consolidated_outcomes.csv")))
N = len(rows)
out = HERE / "outcome_figures"; out.mkdir(exist_ok=True)

EMP = [c for c in rows[0] if c.startswith("emp_") and c not in ("emp_engaged",)]
ORG = [c for c in rows[0] if c.startswith("org_") and c not in ("org_engaged",)]
labels = {0:"Performance",1:"Work-Life",2:"Belonging",3:"Technostress",4:"Psych Safety",5:"Surveillance"}
comms = sorted(labels)
def short(c): return c.replace("emp_","").replace("org_","").replace("_"," ")
BLUE, RED = "#185FA5", "#A32D2D"

# ---------- 1. COVERAGE BAR ----------
allcols = EMP + ORG
prev = {c: sum(1 for r in rows if r[c]=="1") for c in allcols}
order = sorted(allcols, key=lambda c: -prev[c])
fig, ax = plt.subplots(figsize=(10, 9))
yvals = range(len(order))
ax.barh(list(yvals), [100*prev[c]/N for c in order],
        color=[BLUE if c.startswith("emp_") else RED for c in order])
ax.set_yticks(list(yvals)); ax.set_yticklabels([short(c) for c in order], fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("% of 387 papers")
ax.set_title("Outcome coverage: employee (blue) vs organizational (red)")
for i,c in enumerate(order): ax.text(100*prev[c]/N+0.5, i, f"{round(100*prev[c]/N)}%", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(out/"coverage_bar.png", dpi=150, bbox_inches="tight")
fig.savefig(out/"coverage_bar.svg", bbox_inches="tight"); plt.close(fig)
print("saved coverage_bar")

# ---------- 2. HEATMAPS (pct + count), emp & org panels ----------
def matrix(cols, mode):
    M = np.zeros((len(comms), len(cols)))
    for ci, c in enumerate(comms):
        papers = [r for r in rows if int(r["community"])==c]
        n = len(papers) if papers else 1
        for oi, col in enumerate(cols):
            k = sum(1 for r in papers if r[col]=="1")
            M[ci,oi] = (100*k/n) if mode=="pct" else k
    return M

cmap_b = LinearSegmentedColormap.from_list("b",["#FFFFFF","#85B7EB","#185FA5","#042C53"])
cmap_r = LinearSegmentedColormap.from_list("r",["#FFFFFF","#F09595","#A32D2D","#501313"])

for mode in ["pct","count"]:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6),
                             gridspec_kw={"width_ratios":[len(EMP),len(ORG)]})
    for ax, cols, cm, name in [(axes[0],EMP,cmap_b,"Employee"),(axes[1],ORG,cmap_r,"Organizational")]:
        M = matrix(cols, mode)
        im = ax.imshow(M, cmap=cm, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([short(c) for c in cols], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(comms))); ax.set_yticklabels([labels[c] for c in comms], fontsize=9)
        ax.set_title(f"{name} outcomes ({'%' if mode=='pct' else 'count'})")
        for i in range(len(comms)):
            for j in range(len(cols)):
                v = M[i,j]; txt = f"{int(round(v))}" + ("%" if mode=="pct" else "")
                ax.text(j,i,txt,ha="center",va="center",fontsize=7,
                        color="white" if v > (M.max()*0.55) else "#333")
    fig.suptitle(f"Community × outcome coverage ({'percentage within community' if mode=='pct' else 'raw counts'}) — 387 papers")
    fig.tight_layout(); fig.savefig(out/f"heatmap_{mode}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out/f"heatmap_{mode}.svg", bbox_inches="tight"); plt.close(fig)
    print(f"saved heatmap_{mode}")

# ---------- 3. VALENCE BY PERSPECTIVE (valence-carrying subset only) ----------
# benefit / risk valence assigned ONLY to outcomes that carry clear valence; neutrals excluded
VAL = {
  "emp_job_satisfaction":"benefit","emp_engagement":"benefit","emp_wellbeing":"benefit",
  "emp_autonomy":"benefit","emp_flexibility":"benefit","emp_belonging":"benefit",
  "emp_burnout":"risk","emp_technostress":"risk","emp_stress":"risk",
  "emp_isolation":"risk","emp_turnover_intention":"risk","emp_work_life":"neutral",
  "org_innovation":"benefit","org_collaboration":"benefit","org_team_effectiveness":"benefit",
  "org_culture":"neutral","org_absenteeism":"risk",
  # neutral/excluded: productivity, performance, financial, cost, real_estate, retention, commitment, leadership, project_success, job_performance
}
emp_val = {k:v for k,v in VAL.items() if k.startswith("emp_") and v!="neutral"}
org_val = {k:v for k,v in VAL.items() if k.startswith("org_") and v!="neutral"}
def vshare(valmap):
    ben = sum(prev[k] for k,v in valmap.items() if v=="benefit")
    rsk = sum(prev[k] for k,v in valmap.items() if v=="risk")
    tot = ben+rsk
    return (100*ben/tot, 100*rsk/tot) if tot else (0,0)
eb,er = vshare(emp_val); ob,orr = vshare(org_val)
fig, ax = plt.subplots(figsize=(7,5))
x = [0,1]
ax.bar(x,[eb,ob],color="#1D9E75",label="benefit-valence outcomes")
ax.bar(x,[er,orr],bottom=[eb,ob],color="#D85A30",label="risk-valence outcomes")
ax.set_xticks(x); ax.set_xticklabels(["Employee\noutcomes","Organizational\noutcomes"])
ax.set_ylabel("% of valence-carrying outcome mentions"); ax.legend()
ax.set_title("Benefit vs risk valence, by perspective\n(valence-carrying outcomes only; neutral outcomes excluded)")
for xi,(b,r) in zip(x,[(eb,er),(ob,orr)]):
    ax.text(xi,b/2,f"{round(b)}%",ha="center",color="white",fontsize=10)
    ax.text(xi,b+r/2,f"{round(r)}%",ha="center",color="white",fontsize=10)
fig.tight_layout(); fig.savefig(out/"valence_by_perspective.png", dpi=150, bbox_inches="tight")
fig.savefig(out/"valence_by_perspective.svg", bbox_inches="tight"); plt.close(fig)
print("saved valence_by_perspective")
print(f"\nAll figures in {out}")
