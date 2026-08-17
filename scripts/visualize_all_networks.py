"""
visualize_all_networks.py

Renders the full set of networks for the remote/hybrid-work paper:

  SECTION diagnostic : RQ1 community graph + construct(PMI) + construct(Jaccard)
                       side by side. The two construct panels exist to SHOW the
                       weighting instability, not to assert a centrality finding.

  SECTION theory     : the paper-level graph colored three ways, one per theory
                       lens (boundary role, control archetype, JD-R class),
                       using only stable community-level mappings.

  SECTION codings    : the RQ1 community graph in three colorings
                       (by community, by boundary role, by control archetype).

INPUTS
  Paper graph is rebuilt from SPECTER embeddings (kNN, k=15) so it is
  reproducible and matches the RQ1 pipeline. Requires:
    clean/embeddings_specter.npy   (N x 768, normalized)
    clean/corpus_index.csv         (row -> filename, must align to embeddings)
    communities.csv                (row, filename, community)
  Construct panels reuse:
    jdr/construct_presence.csv     (from build_jdr_construct_network.py)

CAVEATS (read before using any output)
  1. Construct-network centrality is weighting-dependent (PMI vs Jaccard give
     opposite orderings). The construct panels are diagnostic, not evidence.
  2. Theory-to-cluster mappings are interpretive claims applied post hoc.
  3. Corpus N must match the manuscript. If communities.csv is 393/389 and the
     manuscript is 387, reconcile before any figure is cited.

USAGE (zsh, conda base):
  python visualize_all_networks.py --list
  python visualize_all_networks.py --section diagnostic
  python visualize_all_networks.py --section theory
  python visualize_all_networks.py --section codings
  python visualize_all_networks.py --section all
"""

import os
import sys
import csv
import argparse
from itertools import combinations
from collections import defaultdict

import numpy as np

SEED = 0

CLEAN = os.path.expanduser("~/Desktop/MyResearch/RemoteWork/Outputs/clean/")
COMM  = os.path.expanduser("~/Desktop/MyResearch/RemoteWork/Outputs/communities.csv")
JDRDIR = os.path.expanduser("~/Desktop/MyResearch/RemoteWork/jdr/")
OUT   = os.path.expanduser("~/Desktop/MyResearch/RemoteWork/jdr/figures/")

KNN = 15

COMMUNITY_LABEL = {
    0: "Performance / productivity",
    1: "Work-life boundary",
    2: "Belonging / identity",
    3: "Psych safety / trust / leadership",
    4: "Technostress / fatigue",
    5: "Surveillance / monitoring",
}

COMMUNITY_COLOR = {
    0: "#888780", 1: "#1D9E75", 2: "#7F77DD",
    3: "#378ADD", 4: "#D85A30", 5: "#A32D2D",
}

BOUNDARY_ROLE = {
    0: "outcome", 1: "boundary-core", 2: "boundary-adjacent",
    3: "boundary-adjacent", 4: "strain", 5: "strain",
}
BOUNDARY_COLOR = {
    "boundary-core": "#0F6E56", "boundary-adjacent": "#5DCAA5",
    "strain": "#D85A30", "outcome": "#888780",
}

CONTROL_ARCH = {
    0: "output", 1: "output", 2: "clan",
    3: "clan", 4: "unmarked", 5: "behaviour",
}
CONTROL_COLOR = {
    "clan": "#378ADD", "behaviour": "#A32D2D",
    "output": "#BA7517", "unmarked": "#B4B2A9",
}

JDR_CLASS = {
    0: "neutral", 1: "resource", 2: "resource",
    3: "resource", 4: "demand", 5: "demand",
}
JDR_COLOR = {
    "resource": "#1D9E75", "demand": "#D85A30", "neutral": "#888780",
}

CONSTRUCTS = [
    "trust", "psychological safety", "surveillance", "monitoring", "autonomy",
    "belonging", "technostress", "burnout", "fatigue", "isolation", "engagement",
    "wellbeing", "communication", "collaboration", "leadership", "productivity",
    "performance", "identity", "inclusion", "flexibility", "work-life balance",
    "privacy", "overload", "satisfaction",
]
CONSTRUCT_JDR = {
    "trust": "resource", "psychological safety": "resource", "autonomy": "resource",
    "leadership": "resource", "inclusion": "resource", "flexibility": "resource",
    "collaboration": "resource", "belonging": "resource", "engagement": "resource",
    "wellbeing": "resource", "satisfaction": "resource", "work-life balance": "resource",
    "surveillance": "demand", "monitoring": "demand", "technostress": "demand",
    "burnout": "demand", "fatigue": "demand", "isolation": "demand",
    "overload": "demand", "privacy": "demand",
    "performance": "neutral", "productivity": "neutral",
    "communication": "neutral", "identity": "neutral",
}
CFILL = {"resource": "#E1F5EE", "demand": "#FAECE7", "neutral": "#F1EFE8"}


def load_paper_graph():
    import igraph as ig
    emb_path = os.path.join(CLEAN, "embeddings_specter.npy")
    idx_path = os.path.join(CLEAN, "corpus_index.csv")
    if not (os.path.exists(emb_path) and os.path.exists(idx_path)):
        sys.stderr.write("missing embeddings or corpus_index in %s\n" % CLEAN)
        sys.exit(1)

    X = np.load(emb_path)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    filenames = []
    with open(idx_path) as f:
        r = csv.DictReader(f)
        fld = "filename" if "filename" in r.fieldnames else r.fieldnames[1]
        for row in r:
            filenames.append(row[fld])

    if len(filenames) != X.shape[0]:
        sys.stderr.write("index/embedding length mismatch: %d vs %d\n"
                         % (len(filenames), X.shape[0]))
        sys.exit(1)

    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    n = X.shape[0]
    edges, weights = set(), {}
    for i in range(n):
        nn = np.argsort(-S[i])[:KNN]
        for j in nn:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a == b:
                continue
            edges.add((a, b))
            weights[(a, b)] = float(S[a, b])

    elist = sorted(edges)
    g = ig.Graph(n=n, edges=elist)
    g.es["weight"] = [max(weights[e], 1e-6) for e in elist]
    g.vs["filename"] = filenames

    comm = {}
    with open(COMM) as f:
        for row in csv.DictReader(f):
            comm[row["filename"]] = int(row["community"])
    g.vs["community"] = [comm.get(fn, -1) for fn in filenames]

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    layout = g.layout_fruchterman_reingold(weights="weight", niter=500)
    return g, np.array(layout.coords)


def draw_paper(ax, g, coords, node_color_fn, title):
    for e in g.es:
        a, b = e.tuple
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                color="#D3D1C7", linewidth=0.2, alpha=0.3, zorder=1)
    colors = [node_color_fn(g.vs[i]["community"]) for i in range(g.vcount())]
    ax.scatter(coords[:, 0], coords[:, 1], s=18, c=colors,
               edgecolors="none", zorder=2)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def legend_from(ax, mapping_color, title):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", label=k,
                      markerfacecolor=v, markersize=9)
               for k, v in mapping_color.items()]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=8, title=title, title_fontsize=8)


def build_construct_graph(assoc, top_pct=0.30):
    import igraph as ig
    ppath = os.path.join(JDRDIR, "construct_presence.csv")
    if not os.path.exists(ppath):
        sys.stderr.write("run build_jdr_construct_network.py --section presence first\n")
        sys.exit(1)
    presence = []
    with open(ppath) as f:
        for row in csv.DictReader(f):
            presence.append([int(row[c]) for c in CONSTRUCTS])
    M = np.array(presence)
    n = len(CONSTRUCTS)
    N = float(len(M))
    counts = M.sum(axis=0).astype(float)
    W = np.zeros((n, n))
    for paper in M:
        idx = [j for j in range(n) if paper[j] == 1]
        for a, b in combinations(idx, 2):
            W[a, b] += 1
            W[b, a] += 1

    raw = []
    for a in range(n):
        for b in range(a + 1, n):
            co = W[a, b]
            if co == 0:
                continue
            if assoc == "jaccard":
                union = counts[a] + counts[b] - co
                val = co / union if union > 0 else 0.0
            else:
                p_ab, p_a, p_b = co / N, counts[a] / N, counts[b] / N
                d = p_a * p_b
                val = max(np.log(p_ab / d), 0.0) if (d > 0 and p_ab > 0) else 0.0
            raw.append((a, b, val))

    raw.sort(key=lambda t: t[2], reverse=True)
    raw = raw[:max(1, int(len(raw) * top_pct))]
    edges = [(a, b) for a, b, _ in raw]
    weights = [w for _, _, w in raw]

    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights
    g.vs["name"] = CONSTRUCTS
    dist = [max(weights) - w + min(weights) for w in weights]
    btw = g.betweenness(weights=dist)
    bmax = max(btw) if max(btw) > 0 else 1.0
    g.vs["btw"] = [b / bmax for b in btw]

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    layout = g.layout_fruchterman_reingold(weights="weight", niter=800)
    return g, np.array(layout.coords)


def draw_construct(ax, g, coords, title):
    wmax = max(g.es["weight"])
    for e in g.es:
        a, b = e.tuple
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                color="#B4B2A9", linewidth=0.3 + 1.6 * (e["weight"] / wmax),
                alpha=0.35, zorder=1)
    for i, nm in enumerate(g.vs["name"]):
        cls = CONSTRUCT_JDR.get(nm, "neutral")
        size = 60 + 900 * g.vs[i]["btw"]
        ax.scatter(coords[i, 0], coords[i, 1], s=size, c=CFILL[cls],
                   edgecolors=JDR_COLOR[cls], linewidths=1.1, zorder=2)
        ax.annotate(nm, (coords[i, 0], coords[i, 1]), ha="center",
                    va="center", fontsize=6, color="#2C2C2A", zorder=3)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def section_diagnostic(g, coords):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_paper(axes[0], g, coords,
               lambda c: COMMUNITY_COLOR.get(c, "#000000"),
               "Paper graph, six communities (stable)")
    legend_from(axes[0],
                {COMMUNITY_LABEL[k]: COMMUNITY_COLOR[k] for k in range(6)},
                "community")
    gp, cp = build_construct_graph("pmi")
    draw_construct(axes[1], gp, cp,
                   "Construct graph, PMI weighting\n(demand brokers > resource -- NOT stable)")
    gj, cj = build_construct_graph("jaccard")
    draw_construct(axes[2], gj, cj,
                   "Construct graph, Jaccard weighting\n(resource > demand -- ordering FLIPS)")
    fig.suptitle("Diagnostic: construct centrality is weighting-dependent; "
                 "only the paper community structure is stable", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "networks_diagnostic.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)


def section_theory(g, coords):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    draw_paper(axes[0], g, coords,
               lambda c: BOUNDARY_COLOR.get(BOUNDARY_ROLE.get(c, "outcome"), "#000"),
               "Boundary theory lens")
    legend_from(axes[0], BOUNDARY_COLOR, "boundary role")

    draw_paper(axes[1], g, coords,
               lambda c: CONTROL_COLOR.get(CONTROL_ARCH.get(c, "unmarked"), "#000"),
               "Control theory lens")
    legend_from(axes[1], CONTROL_COLOR, "control archetype")

    draw_paper(axes[2], g, coords,
               lambda c: JDR_COLOR.get(JDR_CLASS.get(c, "neutral"), "#000"),
               "JD-R lens")
    legend_from(axes[2], JDR_COLOR, "JD-R class")

    fig.suptitle("Three theory lenses over the same paper graph "
                 "(cluster-level interpretive mappings, applied post hoc)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "networks_theory_lenses.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)


def section_codings(g, coords):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_paper(axes[0], g, coords,
               lambda c: COMMUNITY_COLOR.get(c, "#000"), "By community")
    legend_from(axes[0],
                {COMMUNITY_LABEL[k]: COMMUNITY_COLOR[k] for k in range(6)},
                "community")
    draw_paper(axes[1], g, coords,
               lambda c: BOUNDARY_COLOR.get(BOUNDARY_ROLE.get(c, "outcome"), "#000"),
               "By boundary role")
    legend_from(axes[1], BOUNDARY_COLOR, "boundary role")
    draw_paper(axes[2], g, coords,
               lambda c: CONTROL_COLOR.get(CONTROL_ARCH.get(c, "unmarked"), "#000"),
               "By control archetype")
    legend_from(axes[2], CONTROL_COLOR, "control archetype")
    fig.suptitle("Same community network, three colorings", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "networks_three_codings.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)


SECTIONS = {
    "diagnostic": "RQ1 community + construct PMI + construct Jaccard side by side",
    "theory":     "paper graph colored by boundary / control / JD-R lens",
    "codings":    "community network in three colorings",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in SECTIONS.items():
            print("  %-11s %s" % (k, v))
        return
    os.makedirs(OUT, exist_ok=True)
    g, coords = load_paper_graph()
    s = a.section
    if s in (None, "all"):
        section_diagnostic(g, coords)
        section_theory(g, coords)
        section_codings(g, coords)
    elif s == "diagnostic":
        section_diagnostic(g, coords)
    elif s == "theory":
        section_theory(g, coords)
    elif s == "codings":
        section_codings(g, coords)
    else:
        sys.stderr.write("unknown section: %s\n" % s)
        sys.exit(1)


if __name__ == "__main__":
    main()
