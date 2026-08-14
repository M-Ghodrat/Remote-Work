#!/usr/bin/env python3
"""
analyze_communities.py  --  RQ1/RQ2/RQ3 second stage

Runs on the aligned 'clean' folder (corpus_index.csv + embeddings + the PDFs).
Lets the algorithm choose the natural number of communities (does NOT force 6).
Re-reads PDFs in memory for TF-IDF labelling (no text saved to disk).

Produces (into <folder>/rq1_analysis/):
  communities.csv        row, filename, community label
  community_profiles.txt per community: size, top freq terms, top TF-IDF terms,
                         representative papers, tightness, pos/neg density,
                         derived benefit/risk orientation
  centrality.csv         degree, betweenness, closeness, eigenvector per paper (bridges)
  community_table.csv    community | size | pos_score | neg_score | net | orientation
                         (descriptive/theoretical label columns left blank to fill in)

Method notes baked in:
  - Communities via Leiden (fallback Louvain) on a cosine kNN graph (kNN behaves;
    threshold graphs were over-dense in your data and are not used here).
  - k chosen by the algorithm; report the natural count.
  - Sentiment orientation = LEXICAL PREVALENCE of your positive/negative term lists.
    It measures which valence vocabulary appears, NOT net findings (negation not handled).
  - benefit/risk orientation is DERIVED: net = pos_density - neg_density.

Dependencies:
    pip install numpy scikit-learn networkx pdfplumber python-louvain --break-system-packages
    pip install leidenalg igraph --break-system-packages   (optional; enables Leiden)

Usage:
    python3 analyze_communities.py /path/to/clean --pdfs /path/to/RemoteWork
      --pdfs : folder holding the actual PDF files (defaults to clean folder's parent)
    optional: --knn 15   --model specter|mpnet   (default specter)
"""
import sys, csv, re
from pathlib import Path
from collections import Counter
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.neighbors import NearestNeighbors
    import networkx as nx
    import pdfplumber
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install numpy scikit-learn "
             "networkx pdfplumber python-louvain --break-system-packages")

HAVE_LEIDEN = True
try:
    import igraph as ig, leidenalg
except Exception:
    HAVE_LEIDEN = False
try:
    import community as community_louvain
    HAVE_LOUVAIN = True
except Exception:
    HAVE_LOUVAIN = False

POS_TERMS = ["autonomy","flexibility","trust","belonging","engagement","wellbeing",
             "well-being","productivity","satisfaction","collaboration","inclusion",
             "support","commitment"]
NEG_TERMS = ["burnout","technostress","fatigue","surveillance","isolation","overload",
             "conflict","exhaustion","stress","monitoring","strain","loneliness"]

STOP_EXTRA = {"remote","work","hybrid","telework","telecommuting","home","worker",
              "workers","employee","employees","study","research","paper","results",
              "using","based","also","may","one","two","among","within","covid","19",
              "pandemic","www","https","doi","org","et","al",
              # PDF / publisher / extraction noise
              "cid","mdpi","article","com","19pandemic","springer","elsevier","wiley",
              "sage","emerald","taylor","francis","journal","vol","volume","issue","pp",
              "abstract","keywords","author","authors","university","press","review",
              "license","creative","commons","downloaded","online","library",
              "3390","figure","table","fig","copyright","permission","reserved",
              "rights","reserved","lifebalance","19 pandemic"}

# token patterns to drop from vocab: anything containing a digit, or known glitch tokens
NOISE_RE = re.compile(r"^(?:cid|cid\s*cid|mdpi|v\d+|s\d+|p\d+|19pandemic)$", re.I)
def is_noise_term(t):
    t=t.strip()
    if any(ch.isdigit() for ch in t):   # drop any term with a digit (years, DOIs, ids)
        return True
    return bool(NOISE_RE.match(t))

# ---------- text re-read (in memory) for TF-IDF ----------
NUM = r"(?:\d{1,2}[\.\)]?\s+|[IVX]{1,4}[\.\)]\s+)?"
ABSTRACT_H = re.compile(rf"^\s*{NUM}(abstract|summary)\b", re.I)
INTRO_H    = re.compile(rf"^\s*{NUM}(introduction|background)\b", re.I)
REFS_H     = re.compile(rf"^\s*{NUM}(references|bibliography)\b", re.I)

def read_paper_text(path, max_chars=6000):
    """Light read: title region + abstract + early body, capped. For TF-IDF only."""
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = pdf.pages[:4]
            txt = "\n".join((pg.extract_text() or "") for pg in pages)
    except Exception:
        return ""
    # cut references if they appear early
    lines = txt.splitlines()
    keep = []
    for ln in lines:
        if REFS_H.match(ln):
            break
        keep.append(ln)
    return re.sub(r"\s+", " ", " ".join(keep))[:max_chars].lower()

def build_knn_graph(E, k):
    nn = NearestNeighbors(n_neighbors=min(k+1, len(E)), metric="cosine").fit(E)
    dist, ind = nn.kneighbors(E)
    G = nx.Graph(); G.add_nodes_from(range(len(E)))
    for i in range(len(E)):
        for jp in range(1, ind.shape[1]):
            j = int(ind[i, jp]); w = 1.0 - float(dist[i, jp])
            if w > 0: G.add_edge(i, j, weight=w)
    return G

def detect(G):
    if HAVE_LEIDEN and G.number_of_edges():
        edges = list(G.edges()); wts = [G[u][v]["weight"] for u,v in edges]
        g = ig.Graph(n=G.number_of_nodes(), edges=edges); g.es["weight"]=wts
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                        weights="weight", seed=0)
        lab=[0]*G.number_of_nodes()
        for cid,comm in enumerate(part):
            for v in comm: lab[v]=cid
        return lab, "leiden"
    if HAVE_LOUVAIN and G.number_of_edges():
        part = community_louvain.best_partition(G, weight="weight", random_state=0)
        return [part[i] for i in range(G.number_of_nodes())], "louvain"
    sys.exit("No community algorithm available.")

def main():
    args=[a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python3 analyze_communities.py <clean_folder> [--pdfs <pdf_folder>] [--knn 15] [--model specter]")
        return
    folder=Path(args[0])
    knn=15; model="specter"
    if "--knn" in sys.argv: knn=int(sys.argv[sys.argv.index("--knn")+1])
    if "--model" in sys.argv: model=sys.argv[sys.argv.index("--model")+1]
    pdfdir = Path(sys.argv[sys.argv.index("--pdfs")+1]) if "--pdfs" in sys.argv else folder.parent

    names=[r["filename"] for r in csv.DictReader(open(folder/"corpus_index.csv"))]
    E=np.load(folder/f"embeddings_{model}.npy")
    if E.shape[0]!=len(names):
        sys.exit(f"MISALIGNMENT: {model} has {E.shape[0]} rows, index has {len(names)}.")
    N=len(names); print(f"{N} papers | model={model} | Leiden={HAVE_LEIDEN}")

    G=build_knn_graph(E,knn)
    labels,algo=detect(G)
    k=len(set(labels))
    print(f"Communities (algorithm-chosen): {k}  via {algo}  | edges={G.number_of_edges()}")

    outdir=folder/"rq1_analysis"; outdir.mkdir(exist_ok=True)

    # centralities (RQ2 bridges)
    deg=nx.degree_centrality(G); bet=nx.betweenness_centrality(G,weight="weight")
    clo=nx.closeness_centrality(G)
    try: eig=nx.eigenvector_centrality_numpy(G,weight="weight")
    except Exception: eig={i:0 for i in G.nodes()}
    with open(outdir/"centrality.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["row","filename","community","degree","betweenness","closeness","eigenvector"])
        for i in range(N):
            w.writerow([i,names[i],labels[i],round(deg[i],4),round(bet[i],4),
                        round(clo[i],4),round(eig.get(i,0),4)])

    # communities.csv
    with open(outdir/"communities.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["row","filename","community"])
        for i in range(N): w.writerow([i,names[i],labels[i]])

    # re-read PDFs for TF-IDF
    print("Re-reading PDFs for TF-IDF labelling (in memory)...")
    texts=[]
    for i,nm in enumerate(names):
        p=pdfdir/nm
        texts.append(read_paper_text(p) if p.exists() else "")
        if (i+1)%50==0: print(f"  read {i+1}/{N}")
    missing=sum(1 for t in texts if not t)
    if missing: print(f"  note: {missing} PDFs not found in {pdfdir} (TF-IDF skips them)")
    if sum(1 for t in texts if t.strip()) < max(5, N*0.2):
        sys.exit(f"\nERROR: only {N-missing}/{N} PDFs readable in {pdfdir}.\n"
                 f"Pass the correct PDF folder with --pdfs /path/to/RemoteWork .\n"
                 f"(communities.csv and centrality.csv were still written.)")

    stop=list(TfidfVectorizer(stop_words="english").get_stop_words() | STOP_EXTRA)
    cv=CountVectorizer(stop_words=stop, ngram_range=(1,2), min_df=2, max_df=0.6)
    X=cv.fit_transform(texts); vocab=np.array(cv.get_feature_names_out())
    tf=TfidfVectorizer(stop_words=stop, ngram_range=(1,2), min_df=2, max_df=0.6)
    Xtf=tf.fit_transform(texts); vocab_tf=np.array(tf.get_feature_names_out())

    # centroids for representative papers
    rows_table=[]
    with open(outdir/"community_profiles.txt","w") as f:
        f.write(f"RQ1 community profiles -- {N} papers, {k} communities ({algo}, kNN={knn}, {model})\n")
        f.write("Sentiment orientation = lexical prevalence of term lists; NOT net findings.\n")
        f.write("="*70+"\n\n")
        for c in sorted(set(labels)):
            idxs=[i for i in range(N) if labels[i]==c]
            size=len(idxs)
            # representative papers: nearest to community centroid
            cen=E[idxs].mean(0); cen=cen/ (np.linalg.norm(cen)+1e-9)
            sims=E[idxs]@cen
            rep=[idxs[j] for j in np.argsort(-sims)[:8]]
            # top frequency terms (noise-filtered)
            sub=X[idxs].sum(0).A1
            order=np.argsort(-sub)
            topf=[vocab[j] for j in order if not is_noise_term(vocab[j])][:15]
            # top tf-idf terms (mean over cluster, noise-filtered)
            subtf=np.asarray(Xtf[idxs].mean(0)).ravel()
            ordertf=np.argsort(-subtf)
            toptf=[vocab_tf[j] for j in ordertf if not is_noise_term(vocab_tf[j])][:15]
            # sentiment densities (per 1000 tokens)
            jointext=" ".join(texts[i] for i in idxs)
            toks=jointext.split(); ntok=max(len(toks),1)
            pos=sum(jointext.count(t) for t in POS_TERMS)/ntok*1000
            neg=sum(jointext.count(t) for t in NEG_TERMS)/ntok*1000
            net=pos-neg
            orient = "benefit-oriented" if net>0.05 else ("risk-oriented" if net<-0.05 else "mixed")
            tight=float(sims.mean())
            rows_table.append([c,size,round(pos,3),round(neg,3),round(net,3),orient])

            f.write(f"COMMUNITY {c}  (n={size}, tightness={tight:.3f})\n")
            f.write(f"  pos_density={pos:.3f}  neg_density={neg:.3f}  net={net:+.3f}  -> {orient}\n")
            f.write("  top frequency terms : "+", ".join(topf)+"\n")
            f.write("  top TF-IDF terms    : "+", ".join(toptf)+"\n")
            f.write("  representative papers:\n")
            for r in rep: f.write(f"     - {names[r]}\n")
            f.write("\n")

    with open(outdir/"community_table.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["community","size","pos_score","neg_score","net","orientation",
                    "descriptive_label_FILL","theoretical_label_FILL"])
        for r in rows_table: w.writerow(r+["",""])

    # bridges summary
    top_bridge=sorted(range(N), key=lambda i:-bet[i])[:15]
    print("\nTop bridge papers (highest betweenness):")
    for i in top_bridge[:10]:
        print(f"  c{labels[i]}  bet={bet[i]:.4f}  {names[i][:70]}")

    print(f"\nDone. Outputs in {outdir}")
    print(f"Natural community count: {k}. Open community_profiles.txt to assign labels.")

if __name__=="__main__":
    main()
