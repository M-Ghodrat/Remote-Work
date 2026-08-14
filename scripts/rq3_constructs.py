#!/usr/bin/env python3
"""
rq3_constructs.py  --  RQ3: centrality of psychosocial constructs

Option C: TWO methods, compared.

(B) CONCEPT network  -- nodes ARE concepts; edges = PPMI-weighted co-occurrence
    across papers. Gives TRUE construct centrality (degree/betweenness/closeness/
    eigenvector of the node "trust", etc.). This is what RQ3 literally asks.

(A) ANCHORED paper sets -- for each construct, the papers about it; report the
    aggregate centrality of that paper-set in the PAPER network (from RQ1/RQ2) and
    which of the 6 communities those papers span (bridge strength).

Honest limits (printed in output):
  - Co-occurrence is ASSOCIATIONAL, not causal/directional.
  - Concept-network results depend on the concept vocabulary (fixed list below; edit it).
  - Anchored-set centrality describes construct-FOCUSED papers, not the word itself.

Inputs (in <clean> folder): corpus_index.csv, embeddings_specter.npy
Also needs: rq1_analysis/communities.csv and rq1_analysis/centrality.csv (from analyze_communities.py)
And the PDF folder (re-read in memory for concept detection).

Outputs (into <clean>/rq3_constructs/):
  concept_centrality.csv     B: degree/betweenness/closeness/eigenvector + ranks per construct
  concept_edges.csv          B: the concept co-occurrence edges (for the network viz)
  anchored_centrality.csv    A: mean paper-centrality of each construct's paper set
  bridge_strength.csv        A: construct -> primary community, connected communities, bridge strength
  comparison.csv             B-rank vs A-rank per construct
  summary.txt                narrative-ready summary + RQ3 answer skeleton

Dependencies:
    pip install numpy scikit-learn networkx pdfplumber --break-system-packages

Usage:
    python3 rq3_constructs.py <clean_folder> --pdfs <pdf_folder>
"""
import sys, csv, re, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

try:
    import networkx as nx
    import pdfplumber
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install numpy scikit-learn networkx pdfplumber --break-system-packages")

# ---- target constructs (RQ3) ----
TARGETS = ["trust", "psychological safety", "surveillance", "autonomy", "belonging", "technostress"]

# ---- broader concept vocabulary so targets sit in a real field ----
# each concept maps to regex alternatives (lowercased text match)
CONCEPTS = {
    "trust":               r"\btrust\b",
    "psychological safety": r"psychological safety|psychological\s+safety",
    "surveillance":        r"\bsurveillance\b|bossware",
    "monitoring":          r"\bmonitoring\b|employee monitoring|electronic monitoring",
    "autonomy":            r"\bautonomy\b|autonomous",
    "belonging":           r"\bbelonging\b|sense of belonging",
    "technostress":        r"technostress|techno-stress",
    "burnout":             r"\bburnout\b",
    "fatigue":             r"\bfatigue\b|digital fatigue|zoom fatigue",
    "isolation":           r"\bisolation\b|social isolation|loneliness",
    "engagement":          r"\bengagement\b|work engagement",
    "wellbeing":           r"wellbeing|well-being|well being",
    "communication":       r"\bcommunication\b",
    "collaboration":       r"\bcollaboration\b|collaborative",
    "leadership":          r"\bleadership\b|leader\b",
    "productivity":        r"\bproductivity\b|productive",
    "performance":         r"\bperformance\b",
    "identity":            r"\bidentity\b|organizational identification",
    "inclusion":           r"\binclusion\b|inclusive",
    "flexibility":         r"\bflexibility\b|flexible work",
    "work-life balance":   r"work-life balance|work life balance|work-family|work family",
    "privacy":             r"\bprivacy\b",
    "overload":            r"\boverload\b|information overload",
    "satisfaction":        r"\bsatisfaction\b|job satisfaction",
}
CONCEPT_RE = {c: re.compile(p, re.I) for c, p in CONCEPTS.items()}

REFS_H = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\b", re.I)

def read_paper_text(path, max_chars=8000):
    try:
        with pdfplumber.open(str(path)) as pdf:
            txt = "\n".join((pg.extract_text() or "") for pg in pdf.pages[:6])
    except Exception:
        return ""
    keep = []
    for ln in txt.splitlines():
        if REFS_H.match(ln):
            break
        keep.append(ln)
    return " ".join(keep)[:max_chars].lower()

def ranks(d, higher_better=True):
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=higher_better)
    return {k: i + 1 for i, (k, v) in enumerate(items)}

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python3 rq3_constructs.py <clean_folder> --pdfs <pdf_folder>")
        return
    folder = Path(args[0])
    pdfdir = Path(sys.argv[sys.argv.index("--pdfs") + 1]) if "--pdfs" in sys.argv else folder.parent

    names = [r["filename"] for r in csv.DictReader(open(folder / "corpus_index.csv"))]
    N = len(names)
    rq1 = folder / "rq1_analysis"
    comm = {r["filename"]: int(r["community"]) for r in csv.DictReader(open(rq1 / "communities.csv"))}
    paper_cent = {r["filename"]: r for r in csv.DictReader(open(rq1 / "centrality.csv"))}

    # ---- read papers, detect concept presence ----
    print(f"Reading {N} PDFs for concept detection...")
    present = []   # list of sets of concepts present per paper
    for i, nm in enumerate(names):
        p = pdfdir / nm
        t = read_paper_text(p) if p.exists() else ""
        s = {c for c, rx in CONCEPT_RE.items() if rx.search(t)} if t else set()
        present.append(s)
        if (i + 1) % 50 == 0: print(f"  {i+1}/{N}")
    readable = sum(1 for s in present if s)
    if readable < max(5, N * 0.2):
        sys.exit(f"Only {readable}/{N} PDFs readable in {pdfdir}. Fix --pdfs path.")

    concepts = list(CONCEPTS)
    df = {c: sum(1 for s in present if c in s) for c in concepts}   # doc frequency
    co = defaultdict(int)
    for s in present:
        sl = sorted(s)
        for a in range(len(sl)):
            for b in range(a + 1, len(sl)):
                co[(sl[a], sl[b])] += 1

    # ---- (B) concept co-occurrence network, PPMI-weighted ----
    Npapers = max(readable, 1)
    G = nx.Graph()
    G.add_nodes_from(concepts)
    edge_rows = []
    for (a, b), n_ab in co.items():
        if n_ab < 3:            # require >=3 co-occurrences to form an edge
            continue
        pa, pb, pab = df[a] / Npapers, df[b] / Npapers, n_ab / Npapers
        if pa <= 0 or pb <= 0:
            continue
        pmi = math.log(pab / (pa * pb) + 1e-12)
        ppmi = max(pmi, 0.0)
        if ppmi > 0:
            G.add_edge(a, b, weight=ppmi, cooccur=n_ab)
            edge_rows.append([a, b, n_ab, round(ppmi, 4)])

    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, weight="weight")
    clo = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0 for n in G}

    rdeg, rbet, rclo, reig = ranks(deg), ranks(bet), ranks(clo), ranks(eig)
    outdir = folder / "rq3_constructs"; outdir.mkdir(exist_ok=True)

    with open(outdir / "concept_centrality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "is_target", "degree", "betweenness", "closeness", "eigenvector",
                    "rank_degree", "rank_betweenness", "rank_closeness", "rank_eigenvector",
                    "composite_rank"])
        comp = {}
        for c in concepts:
            comp[c] = (rdeg[c] + rbet[c] + rclo[c] + reig[c]) / 4
        comp_rank = ranks(comp, higher_better=False)
        for c in concepts:
            w.writerow([c, c in TARGETS, round(deg[c],4), round(bet[c],4), round(clo[c],4),
                        round(eig.get(c,0),4), rdeg[c], rbet[c], rclo[c], reig[c], comp_rank[c]])

    with open(outdir / "concept_edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["concept_a","concept_b","cooccur","ppmi"])
        w.writerows(sorted(edge_rows, key=lambda r:-r[3]))

    # ---- (A) anchored paper-set centrality + bridge strength ----
    anchored = []
    bridge_rows = []
    for c in TARGETS:
        papers = [names[i] for i in range(N) if c in present[i]]
        if not papers:
            anchored.append([c, 0, 0, 0, 0, 0]); bridge_rows.append([c, "-", "-", 0]); continue
        def mean_metric(m):
            vals = [float(paper_cent[p][m]) for p in papers if p in paper_cent]
            return sum(vals)/len(vals) if vals else 0.0
        anchored.append([c, len(papers),
                         round(mean_metric("degree"),4), round(mean_metric("betweenness"),4),
                         round(mean_metric("closeness"),4), round(mean_metric("eigenvector"),4)])
        comm_counts = Counter(comm[p] for p in papers if p in comm)
        primary = comm_counts.most_common(1)[0][0] if comm_counts else "-"
        connected = sorted(comm_counts)
        # bridge strength: share of the construct's papers OUTSIDE its primary community
        total = sum(comm_counts.values())
        outside = total - comm_counts.get(primary, 0)
        bstr = round(outside / total, 3) if total else 0
        bridge_rows.append([c, primary, " ".join(map(str, connected)), bstr])

    with open(outdir / "anchored_centrality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["construct","n_papers","mean_degree","mean_betweenness","mean_closeness","mean_eigenvector"])
        w.writerows(anchored)
    with open(outdir / "bridge_strength.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["construct","primary_community","connected_communities","bridge_strength_share_outside_primary"])
        w.writerows(bridge_rows)

    # ---- comparison: B composite rank vs A betweenness rank (targets only) ----
    a_bet = {r[0]: r[3] for r in anchored}
    a_rank = ranks(a_bet, higher_better=True)
    with open(outdir / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["construct","B_composite_rank(concept-net)","A_betweenness_rank(anchored-papers)"])
        for c in TARGETS:
            w.writerow([c, comp_rank[c], a_rank.get(c, "-")])

    # ---- summary ----
    with open(outdir / "summary.txt", "w") as f:
        f.write("RQ3 -- centrality of psychosocial constructs (Option C)\n")
        f.write(f"Concept network: {G.number_of_nodes()} concept nodes, {G.number_of_edges()} edges "
                f"(PPMI-weighted, min 3 co-occurrences).\n")
        f.write("Co-occurrence is ASSOCIATIONAL, not causal. Concept-net depends on the fixed vocabulary.\n\n")
        f.write("(B) TRUE construct centrality -- target constructs ranked by composite:\n")
        tgt_comp = sorted([(c, comp_rank[c]) for c in TARGETS], key=lambda kv: kv[1])
        for c, r in tgt_comp:
            f.write(f"  #{r}  {c:22} deg={deg[c]:.3f} bet={bet[c]:.3f} clo={clo[c]:.3f} eig={eig.get(c,0):.3f}\n")
        f.write("\n(A) Construct-focused PAPERS -- bridge strength (share spanning multiple communities):\n")
        for c, prim, conn, bstr in bridge_rows:
            f.write(f"  {c:22} primary=C{prim}  spans={conn}  bridge_strength={bstr}\n")
        f.write("\nRQ3 ANSWER SKELETON (fill from tables):\n")
        f.write("  Most structurally central constructs (concept-net composite): "
                + ", ".join(c for c,_ in tgt_comp[:3]) + "\n")
        f.write("  Strongest cross-community bridges (anchored): "
                + ", ".join(sorted([r[0] for r in bridge_rows], key=lambda c:-dict((x[0],x[3]) for x in bridge_rows)[c])[:3]) + "\n")

    print("\nDone. Outputs in", outdir)
    print("Concept network:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
    print("Open summary.txt for the RQ3 answer skeleton.")

if __name__ == "__main__":
    main()
