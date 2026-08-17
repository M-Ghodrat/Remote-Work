#!/usr/bin/env python3
"""
rq4_prevalence.py  --  RQ4: benefit / risk THEME PREVALENCE (anchor-based)

Scope discipline (read before interpreting any output):
  - This measures whether benefit / risk THEMES are DISCUSSED, via lexical anchor
    presence. It does NOT measure whether a paper ENDORSES a benefit or finds a
    risk -- a paper can mention "surveillance" to critique it or "flexibility" to
    debunk it. Anchor presence == the theme is engaged, NOT a net good/bad verdict.
  - There is deliberately NO net-score / "% pro-remote" column anywhere. Do not add one.
  - Anchors are NOT invented here: POS_TERMS / NEG_TERMS are imported unchanged from
    analyze_communities.py -- the exact lists that produced the RQ1 orientation gradient.

Two levels:
  LEVEL 2 (HEADLINE, seed-independent): per-paper co-presence.
      Each paper -> benefit-only / risk-only / both / neither, corpus-wide.
      Independent of community labels, so NOT affected by the Leiden seed.
      This is the trustworthy result regardless of the seed-0/seed-42 question.
  LEVEL 1 (PROVISIONAL, seed-dependent): per-community prevalence.
      Share of each community's papers invoking benefit / risk / both.
      Depends on community labels -> MUST be recomputed once the canonical seed
      is settled before submission. Tagged PROVISIONAL in the output.

Threshold handling: we do NOT hard-code "1 hit = pole invoked". We report the
FULL distribution at thresholds 1, 2, 3 so you can pick the cut from evidence.

Inputs (in <clean_folder>):
  corpus_index.csv
  rq1_analysis/communities.csv     (filename, community)   -- from analyze_communities.py
Also needs the PDF folder (re-read in memory; no text saved to disk).
Also needs analyze_communities.py importable (same folder or PYTHONPATH).

Outputs (into <clean_folder>/rq4_prevalence/):
  paper_copresence.csv     LEVEL 2: per paper, benefit_hits, risk_hits, class (at thr=1)
  copresence_distribution.csv  LEVEL 2: counts of both/benefit-only/risk-only/neither at thr 1,2,3
  community_prevalence.csv PROVISIONAL LEVEL 1: per community, % benefit / % risk / % both (thr 1,2,3)
  anchor_hits.csv          which anchors fired how often (for the sensitivity note)
  summary.txt              narrative-ready summary + explicit provisional/seed caveats

Usage:
  python3 rq4_prevalence.py <clean_folder> --pdfs <pdf_folder>
  e.g. python3 rq4_prevalence.py clean_389 --pdfs ~/Desktop/MyResearch/RemoteWork
"""
import sys, csv
from pathlib import Path
from collections import Counter, defaultdict

# ---- import the SINGLE SOURCE OF TRUTH from analyze_communities.py ----
# anchors (POS_TERMS / NEG_TERMS) and the 4-page read used for RQ1 orientations.
try:
    from analyze_communities import POS_TERMS, NEG_TERMS, read_paper_text
except Exception as e:
    sys.exit(
        f"Could not import from analyze_communities.py: {e}\n"
        f"Run rq4_prevalence.py from the folder containing analyze_communities.py, "
        f"or add that folder to PYTHONPATH."
    )

THRESHOLDS = [1, 2, 3]   # 'pole invoked' = at least this many distinct anchor hits


def count_hits(text, terms):
    """Number of DISTINCT anchor terms from `terms` present in `text`.
    Distinct-term count (not raw frequency) so one term repeated 20x != 20 hits."""
    if not text:
        return 0, []
    fired = [t for t in terms if t in text]   # text is already lowercased by read_paper_text
    return len(fired), fired


def classify(b_hits, r_hits, thr):
    b = b_hits >= thr
    r = r_hits >= thr
    if b and r:   return "both"
    if b:         return "benefit_only"
    if r:         return "risk_only"
    return "neither"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python3 rq4_prevalence.py <clean_folder> --pdfs <pdf_folder>")
        return
    folder = Path(args[0])
    pdfdir = Path(sys.argv[sys.argv.index("--pdfs") + 1]) if "--pdfs" in sys.argv else folder.parent

    names = [r["filename"] for r in csv.DictReader(open(folder / "corpus_index.csv"))]
    N = len(names)

    comm_path = folder / "rq1_analysis" / "communities.csv"
    if not comm_path.exists():
        sys.exit(f"Missing {comm_path}. Run analyze_communities.py on this folder first.")
    comm = {r["filename"]: int(r["community"]) for r in csv.DictReader(open(comm_path))}

    print(f"RQ4 prevalence | {N} papers")
    print(f"Benefit anchors ({len(POS_TERMS)}): {', '.join(POS_TERMS)}")
    print(f"Risk anchors    ({len(NEG_TERMS)}): {', '.join(NEG_TERMS)}")
    print(f"Reading PDFs (4-page, refs-stripped -- same read as RQ1 orientations)...")

    per_paper = []          # (filename, community, b_hits, r_hits, fired_b, fired_r)
    anchor_counter = Counter()
    readable = 0
    for i, nm in enumerate(names):
        p = pdfdir / nm
        t = read_paper_text(p) if p.exists() else ""   # 4-page read imported from analyze_communities
        if t:
            readable += 1
        bH, firedB = count_hits(t, POS_TERMS)
        rH, firedR = count_hits(t, NEG_TERMS)
        for a in firedB: anchor_counter[("benefit", a)] += 1
        for a in firedR: anchor_counter[("risk", a)] += 1
        per_paper.append((nm, comm.get(nm, -1), bH, rH, firedB, firedR))
        if (i + 1) % 50 == 0: print(f"  {i+1}/{N}")

    if readable < max(5, N * 0.2):
        sys.exit(f"Only {readable}/{N} PDFs readable in {pdfdir}. Fix --pdfs path.")
    print(f"  readable: {readable}/{N}")

    outdir = folder / "rq4_prevalence"; outdir.mkdir(exist_ok=True)

    # ---------- LEVEL 2 (HEADLINE, seed-independent): per-paper co-presence ----------
    with open(outdir / "paper_copresence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "community", "benefit_hits", "risk_hits", "class_thr1",
                    "benefit_anchors_fired", "risk_anchors_fired"])
        for nm, c, bH, rH, fb, fr in per_paper:
            w.writerow([nm, c, bH, rH, classify(bH, rH, 1),
                        "|".join(fb), "|".join(fr)])

    dist_rows = []
    for thr in THRESHOLDS:
        cnt = Counter(classify(bH, rH, thr) for _, _, bH, rH, _, _ in per_paper)
        tot = sum(cnt.values())
        dist_rows.append({
            "threshold": thr,
            "both": cnt["both"], "benefit_only": cnt["benefit_only"],
            "risk_only": cnt["risk_only"], "neither": cnt["neither"],
            "pct_both": round(100 * cnt["both"] / tot, 1),
            "pct_benefit_only": round(100 * cnt["benefit_only"] / tot, 1),
            "pct_risk_only": round(100 * cnt["risk_only"] / tot, 1),
            "pct_neither": round(100 * cnt["neither"] / tot, 1),
        })
    with open(outdir / "copresence_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
        w.writeheader(); w.writerows(dist_rows)

    # ---------- LEVEL 1 (PROVISIONAL, seed-dependent): per-community prevalence ----------
    comms = sorted(set(c for _, c, _, _, _, _ in per_paper if c >= 0))
    with open(outdir / "community_prevalence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PROVISIONAL_seed_dependent_recompute_after_seed_fix"])
        w.writerow(["community", "threshold", "n_papers",
                    "pct_benefit", "pct_risk", "pct_both", "pct_neither"])
        for c in comms:
            papers = [(bH, rH) for _, cc, bH, rH, _, _ in per_paper if cc == c]
            n = len(papers)
            for thr in THRESHOLDS:
                b = sum(1 for bH, rH in papers if bH >= thr)
                r = sum(1 for bH, rH in papers if rH >= thr)
                both = sum(1 for bH, rH in papers if bH >= thr and rH >= thr)
                neither = sum(1 for bH, rH in papers if bH < thr and rH < thr)
                w.writerow([c, thr, n,
                            round(100*b/n, 1), round(100*r/n, 1),
                            round(100*both/n, 1), round(100*neither/n, 1)])

    # ---------- anchor hit frequencies (sensitivity note) ----------
    with open(outdir / "anchor_hits.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pole", "anchor", "n_papers_containing"])
        for (pole, a), n in sorted(anchor_counter.items(), key=lambda kv: -kv[1]):
            w.writerow([pole, a, n])

    # ---------- summary ----------
    with open(outdir / "summary.txt", "w") as f:
        f.write("RQ4 -- benefit / risk THEME PREVALENCE (anchor-based)\n")
        f.write("="*70 + "\n")
        f.write("SCOPE: anchor PRESENCE = theme is DISCUSSED, not endorsed. NOT a net verdict.\n")
        f.write(f"Anchors imported unchanged from analyze_communities.py "
                f"(POS={len(POS_TERMS)}, NEG={len(NEG_TERMS)}).\n")
        f.write(f"Papers: {N}, readable: {readable}. Read depth: 4 pages (matches RQ1 orientations).\n\n")

        f.write("LEVEL 2 -- per-paper co-presence  [HEADLINE; SEED-INDEPENDENT, trustworthy]\n")
        f.write("  (does NOT depend on community labels, so unaffected by the Leiden seed)\n")
        for d in dist_rows:
            f.write(f"  thr>={d['threshold']}: both={d['pct_both']}%  "
                    f"benefit_only={d['pct_benefit_only']}%  "
                    f"risk_only={d['pct_risk_only']}%  neither={d['pct_neither']}%  "
                    f"(n_both={d['both']})\n")
        f.write("\n  Read 'both' as: paper engages BOTH benefit and risk vocabulary.\n")
        f.write("  If 'both' dominates across thresholds -> field is polarized at the\n")
        f.write("  COMMUNITY level (RQ1) but individual papers engage both poles.\n\n")

        f.write("LEVEL 1 -- per-community prevalence  [PROVISIONAL; SEED-DEPENDENT]\n")
        f.write("  *** Community labels are from the current Leiden seed. The bridge/RQ3\n")
        f.write("  *** runs used a different seed. RECOMPUTE this section after the seed is\n")
        f.write("  *** standardized before submission. Level 2 above is unaffected.\n")
        f.write("  (see community_prevalence.csv for the table)\n\n")

        f.write("SENSITIVITY: see anchor_hits.csv -- check no single anchor dominates a pole.\n")
        f.write("If one term (e.g. 'stress') drives most risk hits, report that explicitly.\n")

    print("\nDone. Outputs in", outdir)
    print("LEVEL 2 (headline) co-presence distribution:")
    for d in dist_rows:
        print(f"  thr>={d['threshold']}: both={d['pct_both']}%  ben_only={d['pct_benefit_only']}%  "
              f"risk_only={d['pct_risk_only']}%  neither={d['pct_neither']}%")
    print("\nRead summary.txt. LEVEL 1 (per-community) is PROVISIONAL pending the seed fix.")


if __name__ == "__main__":
    main()
