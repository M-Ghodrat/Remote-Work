#!/usr/bin/env python3
"""
visualize_rq4.py  --  RQ4 benefit/risk co-presence figure

Reads the RQ4 outputs and renders two panels:

  PANEL A (headline, seed-independent): threshold sweep.
     Stacked bars at thr 1/2/3 showing the share of papers that are
     both / benefit-only / risk-only / neither. Visualizes the GRADED
     co-presence finding -- 'both' gives way to 'benefit-only' as the
     anchor threshold tightens. This is the trustworthy result.

  PANEL B (provisional, seed-dependent): per-community 'both' rate at thr>=2.
     Communities ordered by RQ1 net orientation (benefit -> risk). Shows that
     co-presence holds even in the most risk-polarized community (Surveillance).
     Bars colored by RQ1 orientation. LABELLED PROVISIONAL because it depends
     on the current Leiden seed (recompute after seed fix).

Honest framing baked in:
  - Panel A is the headline (seed-independent). Panel B is provisional.
  - 'both' = engages both benefit AND risk vocabulary, NOT a net verdict.
  - thr>=2 is treated as the defensible cut (washes out single generic-word hits
    like 'support'/'stress'); the full sweep is shown so nothing is cherry-picked.

Palette matches visualize_rq3.py: benefit = blue (#2b6cb0), risk = red (#d64545).

Inputs (in <clean_folder>/rq4_prevalence/):
   copresence_distribution.csv   (threshold, both, benefit_only, risk_only, neither, pct_*)
   community_prevalence.csv      (community, threshold, n_papers, pct_benefit/risk/both/neither)
Optional (for community names + orientation ordering in Panel B):
   <clean_folder>/rq1_analysis/community_table.csv  (community, net, orientation)
   --labels "0=Performance,1=Technostress,..."   to name communities on the x-axis

Output:
   <clean_folder>/rq4_prevalence/copresence_figure.png   (300 dpi)
   <clean_folder>/rq4_prevalence/copresence_figure.svg   (vector)

Dependencies:
   pip install matplotlib --break-system-packages

Usage:
   python3 visualize_rq4.py clean_389
   python3 visualize_rq4.py clean_389 --labels "0=Performance,1=Technostress,2=Work-Life,3=Psych Safety,4=Identity/Belonging,5=Surveillance"
"""
import sys, csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install matplotlib --break-system-packages")

# palette consistent with visualize_rq3.py
BENEFIT = "#2b6cb0"   # blue
RISK    = "#d64545"   # red
BOTH    = "#6b7280"   # neutral grey-slate (engages both)
NEITHER = "#d9e2ec"   # pale (engages neither)


def parse_labels(argv):
    if "--labels" not in argv:
        return {}
    spec = argv[argv.index("--labels") + 1]
    out = {}
    for kv in spec.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[int(k.strip())] = v.strip()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python3 visualize_rq4.py <clean_folder> [--labels \"0=Name,1=Name,...\"]")
        return
    base = Path(args[0]) / "rq4_prevalence"
    dist_f = base / "copresence_distribution.csv"
    comm_f = base / "community_prevalence.csv"
    if not dist_f.exists():
        sys.exit(f"Missing {dist_f}. Run rq4_prevalence.py first.")
    labels = parse_labels(sys.argv)

    # ---- Panel A data: threshold sweep ----
    dist = list(csv.DictReader(open(dist_f)))
    thr        = [int(r["threshold"]) for r in dist]
    pct_both   = [float(r["pct_both"]) for r in dist]
    pct_ben    = [float(r["pct_benefit_only"]) for r in dist]
    pct_risk   = [float(r["pct_risk_only"]) for r in dist]
    pct_neither= [float(r["pct_neither"]) for r in dist]

    # ---- Panel B data: per-community 'both' at thr>=2 (provisional) ----
    comm_rows = []
    if comm_f.exists():
        # first line is the PROVISIONAL banner; skip it
        raw = comm_f.read_text().splitlines()
        start = 1 if raw and raw[0].startswith("PROVISIONAL") else 0
        rdr = csv.DictReader(raw[start:])
        for r in rdr:
            if int(r["threshold"]) == 2:
                comm_rows.append((int(r["community"]),
                                  int(r["n_papers"]),
                                  float(r["pct_both"]),
                                  float(r["pct_benefit"]),
                                  float(r["pct_risk"])))

    # orientation (net) for ordering + coloring Panel B, from community_table.csv if present
    net = {}
    table_f = Path(args[0]) / "rq1_analysis" / "community_table.csv"
    if table_f.exists():
        for r in csv.DictReader(open(table_f)):
            try:
                net[int(r["community"])] = float(r["net"])
            except Exception:
                pass

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # ===================== PANEL A: threshold sweep =====================
    x = range(len(thr))
    b1 = axA.bar(x, pct_both, color=BOTH, label="Both poles")
    b2 = axA.bar(x, pct_ben, bottom=pct_both, color=BENEFIT, label="Benefit only")
    bottom2 = [a + b for a, b in zip(pct_both, pct_ben)]
    b3 = axA.bar(x, pct_risk, bottom=bottom2, color=RISK, label="Risk only")
    bottom3 = [a + b for a, b in zip(bottom2, pct_risk)]
    b4 = axA.bar(x, pct_neither, bottom=bottom3, color=NEITHER,
                 edgecolor="#bcccdc", label="Neither")

    # annotate the 'both' share inside each bar
    for i, v in enumerate(pct_both):
        axA.text(i, v / 2, f"{v:.0f}%", ha="center", va="center",
                 color="white", fontweight="bold", fontsize=11)

    axA.set_xticks(list(x))
    axA.set_xticklabels([f"\u2265{t} anchor{'s' if t > 1 else ''}\nper pole" for t in thr])
    axA.set_ylabel("Share of papers (%)")
    axA.set_ylim(0, 100)
    axA.set_title("A. Per-paper benefit/risk co-presence\n(graded by anchor threshold)",
                  fontsize=12, fontweight="bold")
    axA.legend(loc="upper right", fontsize=9, frameon=True)
    axA.spines[["top", "right"]].set_visible(False)
    # mark the defensible headline cut
    axA.axvline(1, ls=":", color="#627d98", lw=1)
    axA.text(1, 102, "defensible cut", ha="center", fontsize=8, color="#627d98")

    # ===================== PANEL B: per-community 'both' (provisional) =====================
    if comm_rows:
        # order by net orientation (benefit high -> risk low) if we have it
        if net:
            comm_rows.sort(key=lambda r: -net.get(r[0], 0))
        ys = [r[2] for r in comm_rows]            # pct_both at thr>=2
        names = []
        colors = []
        for cid, n_p, pboth, pben, prisk in comm_rows:
            nm = labels.get(cid, f"C{cid}")
            names.append(f"{nm}\n(n={n_p})")
            o = net.get(cid, 0)
            colors.append(BENEFIT if o > 0.05 else (RISK if o < -0.05 else BOTH))
        xs = range(len(comm_rows))
        axB.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.6)
        for i, v in enumerate(ys):
            axB.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
        axB.set_xticks(list(xs))
        axB.set_xticklabels(names, fontsize=8)
        axB.set_ylabel("Papers engaging BOTH poles (%)  [thr\u22652]")
        axB.set_ylim(0, 100)
        axB.set_title("B. Co-presence holds across communities  [PROVISIONAL: seed-dependent]\n"
                      "bar color = RQ1 orientation (blue benefit / red risk)",
                      fontsize=11, fontweight="bold")
        axB.spines[["top", "right"]].set_visible(False)
        leg = [Patch(facecolor=BENEFIT, edgecolor="black", label="Benefit-oriented community"),
               Patch(facecolor=RISK, edgecolor="black", label="Risk-oriented community")]
        axB.legend(handles=leg, loc="upper right", fontsize=8, frameon=True)
    else:
        axB.text(0.5, 0.5, "community_prevalence.csv not found\n(Panel B skipped)",
                 ha="center", va="center", transform=axB.transAxes)
        axB.axis("off")

    fig.suptitle("RQ4: Benefit and risk themes co-occur WITHIN papers "
                 "(structural polarization, not per-study)",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.04,
             "Anchor PRESENCE = theme discussed, NOT endorsed. Not a net good/bad verdict. "
             "Most common (softest) anchors: 'support' (benefit), 'stress' (risk).",
             ha="center", fontsize=8, color="#627d98", style="italic")

    plt.tight_layout()
    png = base / "copresence_figure.png"
    svg = base / "copresence_figure.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print("Saved:")
    print(" ", png)
    print(" ", svg)


if __name__ == "__main__":
    main()
