#!/usr/bin/env python3
"""
visualize_rq3.py  --  RQ3 concept network figure

Reads the RQ3 outputs and renders the concept co-occurrence network:
    - node size   = degree centrality (from concept_centrality.csv)
    - node color  = target construct (highlighted) vs context concept
    - node ring   = benefit (blue) / risk (red) for target constructs
    - edge width  = PPMI weight (from concept_edges.csv)
    - the six target constructs are labelled prominently

Static figure via matplotlib + networkx. Runs locally; builds the graph on your machine.

Inputs (in <clean>/rq3_constructs/):
    concept_edges.csv         (concept_a, concept_b, cooccur, ppmi)
    concept_centrality.csv    (concept, is_target, degree, ...)

Output:
    <clean>/rq3_constructs/concept_network.png   (300 dpi)
    <clean>/rq3_constructs/concept_network.svg   (vector, for publication)

Dependencies:
    pip install networkx matplotlib --break-system-packages

Usage:
    python3 visualize_rq3.py ~/Desktop/MyResearch/RemoteWork/Outputs/clean
"""
import sys, csv
from pathlib import Path

try:
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install networkx matplotlib --break-system-packages")

TARGETS = ["trust", "psychological safety", "surveillance", "autonomy", "belonging", "technostress"]
# benefit/risk coloring for the target constructs (from your RQ1 orientation logic)
RISK = {"surveillance", "technostress"}          # risk-pole constructs
BENEFIT = {"trust", "psychological safety", "autonomy", "belonging"}

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python3 visualize_rq3.py <clean_folder>")
        return
    base = Path(args[0]) / "rq3_constructs"
    edges_f = base / "concept_edges.csv"
    cent_f = base / "concept_centrality.csv"
    if not edges_f.exists() or not cent_f.exists():
        sys.exit(f"Missing {edges_f.name} or {cent_f.name}. Run rq3_constructs.py first.")

    # centrality (node size + target flag)
    deg = {}
    is_target = {}
    for r in csv.DictReader(open(cent_f)):
        c = r["concept"]
        deg[c] = float(r["degree"])
        is_target[c] = (r["is_target"].strip().lower() == "true")

    # build graph from edges
    G = nx.Graph()
    G.add_nodes_from(deg)
    max_ppmi = 1e-9
    for r in csv.DictReader(open(edges_f)):
        w = float(r["ppmi"])
        G.add_edge(r["concept_a"], r["concept_b"], weight=w)
        max_ppmi = max(max_ppmi, w)

    # layout: weight-aware spring with stronger repulsion; seed for reproducibility
    n = G.number_of_nodes()
    pos = nx.spring_layout(G, weight="weight", k=2.2/ (n**0.5) + 0.6,
                           iterations=600, seed=7)
    # spread positions to fill the frame
    import numpy as np
    xs = np.array([pos[c][0] for c in G]); ys = np.array([pos[c][1] for c in G])
    if np.ptp(xs) > 0: xs = (xs - xs.min())/np.ptp(xs)*2 - 1
    if np.ptp(ys) > 0: ys = (ys - ys.min())/np.ptp(ys)*2 - 1
    pos = {c: (xs[i], ys[i]) for i, c in enumerate(G)}

    fig, ax = plt.subplots(figsize=(14, 11))

    # edges (width + alpha by ppmi)
    for u, v, d in G.edges(data=True):
        w = d["weight"] / max_ppmi
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#9aa5b1", linewidth=0.4 + 3.2 * w, alpha=0.18 + 0.42 * w, zorder=1)

    # nodes
    def node_size(c):
        return 300 + 6500 * deg.get(c, 0)        # scale degree -> marker area

    # context (non-target) concepts
    ctx = [c for c in G if not is_target.get(c)]
    ax.scatter([pos[c][0] for c in ctx], [pos[c][1] for c in ctx],
               s=[node_size(c) for c in ctx], c="#d9e2ec",
               edgecolors="#bcccdc", linewidths=1.0, zorder=2)

    # target constructs, colored by benefit/risk
    for c in TARGETS:
        if c not in pos:
            continue
        color = "#d64545" if c in RISK else "#2b6cb0"
        ax.scatter([pos[c][0]], [pos[c][1]], s=node_size(c) * 1.15,
                   c=color, edgecolors="black", linewidths=1.6, zorder=3)

    # labels: targets bold below node, context light further below
    for c in G:
        size = node_size(c)
        yoff = 0.055 + (size ** 0.5) / 1400.0
        if is_target.get(c):
            ax.text(pos[c][0], pos[c][1] - yoff, c, fontsize=11, fontweight="bold",
                    ha="center", va="top", color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        else:
            ax.text(pos[c][0], pos[c][1] - yoff*0.8, c, fontsize=7,
                    ha="center", va="top", color="#627d98", zorder=4)

    ax.set_title("RQ3: Psychosocial construct co-occurrence network\n"
                 "(node size = degree centrality; blue = benefit construct, red = risk construct; "
                 "edge width = PPMI)", fontsize=13)
    ax.axis("off")

    legend = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#2b6cb0",
               markeredgecolor="black", markersize=14, label="Target construct (benefit pole)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d64545",
               markeredgecolor="black", markersize=14, label="Target construct (risk pole)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d9e2ec",
               markeredgecolor="#bcccdc", markersize=12, label="Context concept"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=10, frameon=True)

    plt.tight_layout()
    png = base / "concept_network.png"
    svg = base / "concept_network.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print("Saved:")
    print(" ", png)
    print(" ", svg)
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

if __name__ == "__main__":
    main()
