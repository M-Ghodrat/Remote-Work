"""
build_jdr_construct_network.py

Construct-level co-occurrence network for the remote/hybrid-work corpus,
annotated with the Job Demands-Resources (JD-R) mapping.

WHAT THIS DOES
  Reads the corpus PDFs, counts which of 24 target constructs co-occur in
  each paper, builds a weighted construct-construct graph, computes
  centrality, tags each construct as resource / demand / outcome-neutral,
  and exports: edge list, node table, and a colored network figure.

WHAT THIS DOES NOT DO
  It does not prove JD-R at the paper-community level. The earlier test
  (centrality.csv, paper-level) showed resource and demand communities have
  nearly identical centrality. The real, reportable JD-R signal is here, at
  the CONSTRUCT level: autonomy and trust broker the space; surveillance,
  monitoring, privacy sit at the periphery.

EDGE RULE
  Default = per-paper co-occurrence: an edge (a,b) is incremented once for
  every paper in which BOTH construct a and construct b appear at least once.
  This is a documented, reproducible rule. If the original concept graph used
  a different rule (sliding window, cosine on construct embeddings), the
  centrality values will differ. Do not claim reproduction of concept_centrality.csv
  until the edge rule is confirmed to match.

USAGE (zsh, conda base):
  export ANTHROPIC_API_KEY=notneeded_this_script_uses_no_api
  python build_jdr_construct_network.py --list
  python build_jdr_construct_network.py --section all
  python build_jdr_construct_network.py --section network
  python build_jdr_construct_network.py --section figure

Paths default to the RemoteWork corpus. Override with --corpus and --outdir.
"""

import os
import re
import sys
import csv
import glob
import json
import argparse
from itertools import combinations
from collections import defaultdict

import numpy as np

SEED = 0

# ---------------------------------------------------------------------------
# Target constructs and their surface forms (lowercased, matched as word-ish
# tokens). Keep this list aligned with the 24 constructs in concept_centrality.csv.
# Each construct maps to one or more regex-safe surface strings.
# ---------------------------------------------------------------------------
CONSTRUCTS = {
    "trust":            [r"trust"],
    "psychological safety": [r"psychological safety", r"psychological-safety"],
    "surveillance":     [r"surveillance"],
    "monitoring":       [r"monitoring", r"bossware"],
    "autonomy":         [r"autonomy", r"autonomous"],
    "belonging":        [r"belonging", r"belongingness"],
    "technostress":     [r"technostress", r"techno-stress", r"techno stress"],
    "burnout":          [r"burnout", r"burn-out"],
    "fatigue":          [r"fatigue"],
    "isolation":        [r"isolation", r"isolated", r"loneliness"],
    "engagement":       [r"engagement", r"engaged"],
    "wellbeing":        [r"well-?being"],
    "communication":    [r"communication"],
    "collaboration":    [r"collaboration", r"collaborative"],
    "leadership":       [r"leadership", r"leader"],
    "productivity":     [r"productivity", r"productive"],
    "performance":      [r"performance"],
    "identity":         [r"identity", r"identification"],
    "inclusion":        [r"inclusion", r"inclusive"],
    "flexibility":      [r"flexibility", r"flexible"],
    "work-life balance": [r"work-?life balance", r"work-?family balance"],
    "privacy":          [r"privacy"],
    "overload":         [r"overload"],
    "satisfaction":     [r"satisfaction", r"satisfied"],
}

# ---------------------------------------------------------------------------
# JD-R mapping. resource = supports goals / buffers strain.
# demand = draws effort / causes strain. neutral = outcome or channel, not a
# demand or resource in itself. This mapping is a THEORETICAL CLAIM, applied
# post hoc. State it as such in the manuscript.
# ---------------------------------------------------------------------------
JDR = {
    "trust": "resource",
    "psychological safety": "resource",
    "autonomy": "resource",
    "leadership": "resource",
    "inclusion": "resource",
    "flexibility": "resource",
    "collaboration": "resource",
    "belonging": "resource",
    "engagement": "resource",
    "wellbeing": "resource",
    "satisfaction": "resource",
    "work-life balance": "resource",

    "surveillance": "demand",
    "monitoring": "demand",
    "technostress": "demand",
    "burnout": "demand",
    "fatigue": "demand",
    "isolation": "demand",
    "overload": "demand",
    "privacy": "demand",

    "performance": "neutral",
    "productivity": "neutral",
    "communication": "neutral",
    "identity": "neutral",
}

COLOR = {"resource": "#1D9E75", "demand": "#D85A30", "neutral": "#888780"}
FILL  = {"resource": "#E1F5EE", "demand": "#FAECE7", "neutral": "#F1EFE8"}


def compile_patterns():
    compiled = {}
    for name, forms in CONSTRUCTS.items():
        compiled[name] = [re.compile(r"\b" + f + r"\b", re.IGNORECASE) for f in forms]
    return compiled


def extract_text(path):
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text() or ""
                out.append(t)
        return "\n".join(out)
    except Exception as e:
        sys.stderr.write("skip %s : %s\n" % (os.path.basename(path), e))
        return ""


# ---------------------------------------------------------------------------
# SECTION: presence
# Build a per-paper presence matrix: which constructs appear in each paper.
# ---------------------------------------------------------------------------
def section_presence(corpus, outdir):
    pats = compile_patterns()
    pdfs = sorted(glob.glob(os.path.join(corpus, "**", "*.pdf"), recursive=True))
    if not pdfs:
        sys.stderr.write("no PDFs under %s\n" % corpus)
        sys.exit(1)
    rows = []
    names = list(CONSTRUCTS.keys())
    for i, p in enumerate(pdfs):
        txt = extract_text(p)
        low = txt.lower()
        present = {}
        for name in names:
            hit = any(rx.search(low) for rx in pats[name])
            present[name] = 1 if hit else 0
        row = {"filename": os.path.basename(p)}
        row.update(present)
        rows.append(row)
        if (i + 1) % 25 == 0:
            sys.stderr.write("  presence %d/%d\n" % (i + 1, len(pdfs)))
    path = os.path.join(outdir, "construct_presence.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename"] + names)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, "(%d papers)" % len(rows))
    return path


# ---------------------------------------------------------------------------
# SECTION: network
# From presence matrix build weighted construct-construct co-occurrence graph,
# compute centrality, write node table + edge list.
# ---------------------------------------------------------------------------
def section_network(outdir, min_weight=1, top_pct=None, assoc="raw"):
    import igraph as ig

    ppath = os.path.join(outdir, "construct_presence.csv")
    if not os.path.exists(ppath):
        sys.stderr.write("run --section presence first\n")
        sys.exit(1)

    names = list(CONSTRUCTS.keys())
    presence = []
    with open(ppath) as f:
        for r in csv.DictReader(f):
            presence.append([int(r[n]) for n in names])
    M = np.array(presence)

    n = len(names)
    N = float(len(M))
    counts = M.sum(axis=0).astype(float)
    W = np.zeros((n, n), dtype=float)
    for paper in M:
        idx = [j for j in range(n) if paper[j] == 1]
        for a, b in combinations(idx, 2):
            W[a, b] += 1
            W[b, a] += 1

    A = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(a + 1, n):
            co = W[a, b]
            if co == 0:
                continue
            if assoc == "raw":
                val = co
            elif assoc == "jaccard":
                union = counts[a] + counts[b] - co
                val = co / union if union > 0 else 0.0
            elif assoc == "pmi":
                p_ab = co / N
                p_a = counts[a] / N
                p_b = counts[b] / N
                denom = p_a * p_b
                val = np.log(p_ab / denom) if (denom > 0 and p_ab > 0) else 0.0
                val = max(val, 0.0)
            else:
                val = co
            A[a, b] = A[b, a] = val

    raw = []
    for a in range(n):
        for b in range(a + 1, n):
            if A[a, b] > 0:
                raw.append((a, b, A[a, b]))

    if top_pct is not None:
        raw.sort(key=lambda t: t[2], reverse=True)
        keep = max(1, int(len(raw) * top_pct))
        raw = raw[:keep]
        floor_note = "assoc=%s, top %d%%" % (assoc, int(top_pct * 100))
    else:
        raw = [t for t in raw if t[2] >= min_weight]
        floor_note = "assoc=%s, weight >= %s" % (assoc, min_weight)

    edges = [(a, b) for a, b, _ in raw]
    weights = [w for _, _, w in raw]
    density = 2.0 * len(edges) / (n * (n - 1))
    sys.stderr.write("edge filter: %s -> %d edges, density=%.3f\n"
                     % (floor_note, len(edges), density))

    g = ig.Graph(n=n, edges=edges)
    g.vs["name"] = names
    g.es["weight"] = weights

    wmin = min(weights) if weights else 1.0
    wmax_e = max(weights) if weights else 1.0
    dist = [(wmax_e - w + wmin) for w in weights]

    deg = g.strength(weights="weight")
    dmax = max(deg) if max(deg) > 0 else 1.0
    deg_norm = [d / dmax for d in deg]
    btw = g.betweenness(weights=dist)
    bmax = max(btw) if max(btw) > 0 else 1.0
    btw_norm = [b / bmax for b in btw]
    clo = g.closeness(weights=dist)
    clo = [0.0 if (c is None or (isinstance(c, float) and np.isnan(c))) else c for c in clo]
    eig = g.eigenvector_centrality(weights="weight")

    node_path = os.path.join(outdir, "jdr_construct_nodes.csv")
    with open(node_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["construct", "jdr_class", "degree", "betweenness",
                    "closeness", "eigenvector"])
        for i, nm in enumerate(names):
            w.writerow([nm, JDR.get(nm, "neutral"),
                        round(deg_norm[i], 4), round(btw_norm[i], 4),
                        round(clo[i], 4), round(eig[i], 4)])
    print("wrote", node_path)

    edge_path = os.path.join(outdir, "jdr_construct_edges.csv")
    with open(edge_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight"])
        for (a, b), wt in zip(edges, weights):
            w.writerow([names[a], names[b], int(wt)])
    print("wrote", edge_path, "(%d edges)" % len(edges))

    print("\nJD-R mean centrality by class (construct level):")
    for cls in ("resource", "demand", "neutral"):
        idx = [i for i, nm in enumerate(names) if JDR.get(nm) == cls]
        if idx:
            print("  %-8s n=%2d  deg=%.3f  btw=%.4f  eig=%.3f" % (
                cls, len(idx),
                np.mean([deg_norm[i] for i in idx]),
                np.mean([btw_norm[i] for i in idx]),
                np.mean([eig[i] for i in idx])))
    return node_path, edge_path


# ---------------------------------------------------------------------------
# SECTION: figure
# Force-directed layout, nodes colored by JD-R class, sized by betweenness.
# ---------------------------------------------------------------------------
def section_figure(outdir):
    import igraph as ig
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    node_path = os.path.join(outdir, "jdr_construct_nodes.csv")
    edge_path = os.path.join(outdir, "jdr_construct_edges.csv")
    if not (os.path.exists(node_path) and os.path.exists(edge_path)):
        sys.stderr.write("run --section network first\n")
        sys.exit(1)

    nodes = {}
    order = []
    with open(node_path) as f:
        for r in csv.DictReader(f):
            nodes[r["construct"]] = r
            order.append(r["construct"])

    edges, weights = [], []
    with open(edge_path) as f:
        for r in csv.DictReader(f):
            edges.append((order.index(r["source"]), order.index(r["target"])))
            weights.append(float(r["weight"]))

    g = ig.Graph(n=len(order), edges=edges)
    g.es["weight"] = weights
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    layout = g.layout_fruchterman_reingold(weights="weight", niter=1000)
    coords = np.array(layout.coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    wmax = max(weights) if weights else 1.0
    for (a, b), wt in zip(edges, weights):
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                color="#B4B2A9", linewidth=0.3 + 2.0 * (wt / wmax), alpha=0.35,
                zorder=1)

    for i, nm in enumerate(order):
        cls = nodes[nm]["jdr_class"]
        btw = float(nodes[nm]["betweenness"])
        size = 200 + 2600 * btw
        ax.scatter(coords[i, 0], coords[i, 1], s=size, c=FILL[cls],
                   edgecolors=COLOR[cls], linewidths=1.4, zorder=2)
        ax.annotate(nm, (coords[i, 0], coords[i, 1]), ha="center", va="center",
                    fontsize=8, color="#2C2C2A", zorder=3)

    handles = [Line2D([0], [0], marker="o", color="w", label=c,
                      markerfacecolor=FILL[c], markeredgecolor=COLOR[c],
                      markersize=12, markeredgewidth=1.4)
               for c in ("resource", "demand", "neutral")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)
    ax.set_title("Construct co-occurrence network, JD-R mapping\n"
                 "node size = betweenness; color = demand / resource / neutral",
                 fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(outdir, "jdr_construct_network.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


SECTIONS = {
    "presence": "extract per-paper construct presence matrix (reads PDFs)",
    "network":  "build weighted construct graph, centrality, node+edge CSV",
    "figure":   "render force-directed JD-R-colored network PNG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expanduser("~/Desktop/MyResearch/RemoteWork/"))
    ap.add_argument("--outdir", default=os.path.expanduser("~/Desktop/MyResearch/RemoteWork/jdr/"))
    ap.add_argument("--section", default=None)
    ap.add_argument("--min-weight", type=int, default=1,
                    help="keep edges with co-occurrence weight >= this")
    ap.add_argument("--top-pct", type=float, default=None,
                    help="instead keep only top fraction of edges by weight, e.g. 0.15")
    ap.add_argument("--assoc", default="raw", choices=["raw", "jaccard", "pmi"],
                    help="edge weighting: raw co-occurrence, jaccard, or positive PMI")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, v in SECTIONS.items():
            print("  %-10s %s" % (k, v))
        return

    os.makedirs(a.outdir, exist_ok=True)
    s = a.section
    if s in (None, "all"):
        section_presence(a.corpus, a.outdir)
        section_network(a.outdir, a.min_weight, a.top_pct, a.assoc)
        section_figure(a.outdir)
    elif s == "presence":
        section_presence(a.corpus, a.outdir)
    elif s == "network":
        section_network(a.outdir, a.min_weight, a.top_pct, a.assoc)
    elif s == "figure":
        section_figure(a.outdir)
    else:
        sys.stderr.write("unknown section: %s\n" % s)
        sys.exit(1)


if __name__ == "__main__":
    main()
