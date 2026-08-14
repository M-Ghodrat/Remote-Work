import csv, math, os
from pathlib import Path
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

HERE = Path(os.environ.get("REMOTE_WORK_OUTPUT_DIR", Path(__file__).parent))
rows = list(csv.DictReader(open(HERE / "consolidated_outcomes.csv")))
N = len(rows)
EMP = [c for c in rows[0] if c.startswith("emp_")]
ORG = [c for c in rows[0] if c.startswith("org_")]

def presence(cols):
    return {c: set(i for i, r in enumerate(rows) if r[c] == "1") for c in cols}

def build_graph(cols, min_co=8, keep_top_frac=0.4):
    pres = presence(cols)
    counts = {c: len(pres[c]) for c in cols}
    cand = []
    for a, b in combinations([c for c in cols if counts[c] > 0], 2):
        co = len(pres[a] & pres[b])
        if co < min_co:
            continue
        pa, pb, pab = counts[a]/N, counts[b]/N, co/N
        ppmi = math.log(pab / (pa*pb))
        if ppmi > 0:
            cand.append((a, b, ppmi, co))
    cand.sort(key=lambda x: -x[2])
    keep = cand[:max(1, int(len(cand)*keep_top_frac))]
    G = nx.Graph()
    for c in cols:
        if counts[c] > 0: G.add_node(c, n=counts[c])
    for a, b, ppmi, co in keep:
        G.add_edge(a, b, weight=ppmi, co=co)
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])
    return G

def short(name): return name.replace("emp_","").replace("org_","").replace("_"," ")

def draw(G, title, colorfn, path):
    if G.number_of_nodes() == 0: print(f"  {title}: empty"); return
    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(G, k=2.0, seed=0, weight="weight", iterations=300)
    sizes = [G.nodes[n]["n"] * 14 for n in G.nodes()]
    colors = [colorfn(n) for n in G.nodes()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    wmax = max(weights) if weights else 1
    nx.draw_networkx_edges(G, pos, width=[3.0*w/wmax for w in weights], edge_color="#94a3b8", alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, edgecolors="white", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: short(n) for n in G.nodes()}, font_size=9, ax=ax)
    ax.set_title(title + f"  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", fontsize=12); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).replace(".png",".svg"), bbox_inches="tight")
    print(f"  saved {path.name}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

out = HERE / "outcome_networks"; out.mkdir(exist_ok=True)
BLUE, RED = "#185FA5", "#A32D2D"
both_color = lambda n: BLUE if n.startswith("emp_") else RED
print("Building networks (stricter thresholds)...")
draw(build_graph(EMP+ORG, min_co=10, keep_top_frac=0.35), "Employee (blue) + Org (red) outcome co-occurrence", both_color, out/"combined.png")
draw(build_graph(EMP, min_co=8, keep_top_frac=0.45), "Employee outcome co-occurrence", lambda n: BLUE, out/"employee.png")
draw(build_graph(ORG, min_co=8, keep_top_frac=0.45), "Organizational outcome co-occurrence", lambda n: RED, out/"org.png")
print(f"\nDone. Outputs in {out}")
