#!/usr/bin/env python3
"""
visualize_network.py  --  RQ1+RQ2 paper-level community network (headline figure)

387-paper network. Nodes = papers, positioned by a 2D projection of the SPECTER
embeddings (UMAP if installed, else PCA) so community separation reflects the real
embedding structure RQ1 found -- not an arbitrary spring layout.

  node color  = community (blue -> red by RQ1 net orientation; benefit..risk)
  node size   = betweenness centrality (RQ2 bridge strength)
  ringed      = top-N bridge papers (highest betweenness)
  labeled     = top-K bridges by name + community centroid labels

Uses the SEED-0 CANONICAL files so colors and bridges share one partition:
  <clean>/rq1_analysis/communities.csv   (row, filename, community)  seed 0
  <clean>/rq1_analysis/centrality.csv    (row, filename, community, degree, betweenness, ...)
  <clean>/embeddings_specter.npy
  <clean>/rq1_analysis/community_table.csv  (community, net, orientation)  for coloring/labels

Output:
  <clean>/rq1_analysis/community_network.png  (300 dpi)
  <clean>/rq1_analysis/community_network.svg

Dependencies:
  pip install numpy matplotlib scikit-learn --break-system-packages
  pip install umap-learn --break-system-packages   (optional; nicer layout)

Usage:
  python3 visualize_network.py clean_389 \
     --labels "0=Performance & Productivity,1=Technostress & Digital Fatigue,2=Work-Life Boundary,3=Psychological Safety & Trust,4=Identity & Belonging,5=Surveillance & Monitoring" \
     --rings 10 --names 5 --edges faint
"""
import sys, csv
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. pip install numpy matplotlib scikit-learn --break-system-packages")


def opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def parse_labels():
    if "--labels" not in sys.argv:
        return {}
    spec = sys.argv[sys.argv.index("--labels") + 1]
    out = {}
    for kv in spec.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[int(k.strip())] = v.strip()
    return out


# blue (benefit) -> red (risk) ramp, assigned by community net orientation rank
RAMP = ["#1d4e89", "#2b6cb0", "#5b8fc9", "#c98b8b", "#d64545", "#b02525"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python3 visualize_network.py <clean_folder> [--labels ...] "
              "[--rings 10] [--names 5] [--edges faint|none]")
        return
    folder = Path(args[0])
    rq1 = folder / "rq1_analysis"
    n_rings = int(opt("--rings", 10))
    n_names = int(opt("--names", 5))
    edge_mode = opt("--edges", "faint")
    labels = parse_labels()

    # --- load canonical seed-0 data ---
    cent = list(csv.DictReader(open(rq1 / "centrality.csv")))
    names = [r["filename"] for r in cent]
    comm = np.array([int(r["community"]) for r in cent])
    bet = np.array([float(r["betweenness"]) for r in cent])
    N = len(names)

    X = np.load(folder / "embeddings_specter.npy")
    if X.shape[0] != N:
        sys.exit(f"MISALIGN: embeddings {X.shape[0]} vs centrality {N}")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # orientation (net) per community -> color order benefit..risk
    net = {}
    tbl = rq1 / "community_table.csv"
    if tbl.exists():
        for r in csv.DictReader(open(tbl)):
            try: net[int(r["community"])] = float(r["net"])
            except Exception: pass
    comms_sorted = sorted(set(comm.tolist()),
                          key=lambda c: -net.get(c, 0))   # benefit high -> risk low
    color_of = {c: RAMP[i % len(RAMP)] for i, c in enumerate(comms_sorted)}

    # --- 2D layout from embeddings ---
    try:
        import umap
        pos = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="cosine",
                        random_state=0).fit_transform(Xn)
        layout = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        pos = PCA(n_components=2, random_state=0).fit_transform(Xn)
        layout = "PCA"
    # normalize to a clean frame
    pos = pos - pos.min(0)
    pos = pos / (pos.max(0) + 1e-9) * 2 - 1

    fig, ax = plt.subplots(figsize=(13, 11))

    # --- edges (faint kNN, optional) ---
    if edge_mode == "faint":
        from sklearn.neighbors import NearestNeighbors
        k = 15
        nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(Xn)
        _, ind = nn.kneighbors(Xn)
        seen = set()
        for i in range(N):
            for jp in range(1, ind.shape[1]):
                j = int(ind[i, jp]); key = (min(i,j), max(i,j))
                if key in seen: continue
                seen.add(key)
                # only draw within-ish-range edges faintly to avoid full hairball
                ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                        color="#cbd5e0", lw=0.25, alpha=0.10, zorder=1)

    # --- nodes ---
    sizes = 40 + 1400 * (bet / (bet.max() + 1e-9))
    for c in comms_sorted:
        m = comm == c
        ax.scatter(pos[m,0], pos[m,1], s=sizes[m], c=color_of[c],
                   edgecolors="white", linewidths=0.4, alpha=0.9, zorder=2)

    # --- ring the top-N bridges ---
    top = np.argsort(-bet)[:n_rings]
    ax.scatter(pos[top,0], pos[top,1], s=sizes[top]*1.6,
               facecolors="none", edgecolors="black", linewidths=1.4, zorder=3)

    # --- label the top-K bridges ---
    def short(fn):
        base = fn.replace(".pdf","")
        return (base[:34] + "...") if len(base) > 36 else base
    for i in top[:n_names]:
        ax.annotate(short(names[i]), (pos[i,0], pos[i,1]),
                    fontsize=7.5, fontweight="bold", zorder=5,
                    xytext=(6, 6), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#999", alpha=0.85))

    # --- community centroid labels ---
    for c in comms_sorted:
        m = comm == c
        cx, cy = pos[m,0].mean(), pos[m,1].mean()
        name = labels.get(c, f"C{c}")
        o = net.get(c, 0)
        tag = f"{name}\n(n={int(m.sum())}, net={o:+.1f})"
        ax.text(cx, cy, tag, fontsize=10, fontweight="bold", ha="center", va="center",
                zorder=6, color="#1a202c",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4a5568", alpha=0.82))

    ax.set_title("RQ1 communities + RQ2 bridges: 387-paper remote/hybrid-work network\n"
                 f"(node size = betweenness; ringed = top {n_rings} bridges; "
                 f"color = community by benefit\u2192risk orientation; layout = {layout} of SPECTER embeddings)",
                 fontsize=12)
    ax.axis("off")

    leg = [Patch(facecolor=color_of[c], edgecolor="white",
                 label=f"{labels.get(c, f'C{c}')} ({'benefit' if net.get(c,0)>0 else 'risk'})")
           for c in comms_sorted]
    leg.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="black", markersize=12, label=f"Top-{n_rings} bridge paper"))
    ax.legend(handles=leg, loc="lower left", fontsize=8, frameon=True)

    fig.text(0.5, 0.005,
             "Layout position is illustrative (embedding projection); community membership "
             "and betweenness are the measured quantities (seed-0 partition).",
             ha="center", fontsize=8, color="#627d98", style="italic")

    plt.tight_layout()
    png = rq1 / "community_network.png"
    svg = rq1 / "community_network.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print("Saved:")
    print(" ", png)
    print(" ", svg)
    print(f"Layout: {layout} | nodes: {N} | rings: {n_rings} | labeled: {n_names}")


if __name__ == "__main__":
    main()
