import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
E = np.load(HERE / "embeddings_specter.npy")
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

comm = {}
for r in csv.DictReader(open(HERE / "rq1_analysis" / "communities.csv")):
    comm[int(r["row"])] = int(r["community"])
labels = {0:"Performance",1:"Work-Life",2:"Belonging",3:"Technostress",4:"Psych Safety",5:"Surveillance"}

order = sorted(range(len(E)), key=lambda i: (comm.get(i,99), i))
S = E[order] @ E[order].T

bounds = []
prev = None
for pos, i in enumerate(order):
    c = comm.get(i,99)
    if c != prev:
        bounds.append((pos, c)); prev = c
bounds.append((len(order), None))

fig, ax = plt.subplots(figsize=(10,9))
cmap = LinearSegmentedColormap.from_list("sim", ["#FFFFFF","#B5D4F4","#185FA5","#042C53"])
im = ax.imshow(S, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

for pos, c in bounds[:-1]:
    ax.axhline(pos-0.5, color="#A32D2D", lw=0.8)
    ax.axvline(pos-0.5, color="#A32D2D", lw=0.8)

ticks, ticklabels = [], []
for (start,c),(end,_) in zip(bounds[:-1], bounds[1:]):
    ticks.append((start+end)/2); ticklabels.append(f"{labels.get(c,c)}\n(n={end-start})")
ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=8, rotation=45, ha="right")
ax.set_yticks(ticks); ax.set_yticklabels(ticklabels, fontsize=8)

ax.set_title("Paper-paper cosine similarity (SPECTER), block-ordered by community\n"
             "387-paper corpus; red lines = community boundaries", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("cosine similarity")
fig.tight_layout()
fig.savefig(HERE / "rq1_analysis" / "similarity_heatmap.png", dpi=150, bbox_inches="tight")
fig.savefig(HERE / "rq1_analysis" / "similarity_heatmap.svg", bbox_inches="tight")
print("Saved rq1_analysis/similarity_heatmap.png and .svg")
print(f"Matrix: {S.shape}, communities ordered: {[labels.get(c,c) for _,c in bounds[:-1]]}")
