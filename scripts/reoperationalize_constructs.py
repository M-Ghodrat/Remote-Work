"""
reoperationalize_constructs.py

Redefines the automated construct-presence measure to better track SUBSTANTIVE
engagement (what the human coder judged) rather than mere lexical occurrence
(what construct_presence.csv currently encodes), then validates honestly.

THE OVERFITTING GUARD
  The human codes are already known. Tuning a threshold against all 278 matched
  papers and reporting the best kappa would be a training score, not a
  validation. So:
    - the matched papers are split ONCE, by fixed seed, into DEV (60%) and
      HELDOUT (40%), stratified is not attempted because presence rates differ
      per construct; the split is by paper and shared across constructs
    - every candidate operationalization is tuned on DEV only
    - the single winning definition per construct is then run ONCE on HELDOUT
    - the HELDOUT kappa is the reported number, whatever it is
  Do not re-run this script with different candidates after seeing HELDOUT
  results. That would reintroduce the leak. If candidates change, the split
  must be regenerated with a new seed and the old HELDOUT result discarded.

CANDIDATE OPERATIONALIZATIONS
  c1_any        construct appears at least once            (current baseline)
  c2_freq_N     construct appears at least N times         (N tuned on dev)
  c3_titleabs   construct appears in title or abstract      (proxy for focus)
  c4_density    occurrences per 1000 words above threshold  (length-normalized)
  c5_titleabs_or_freq   c3 OR c2                            (union rule)

  Human 0/1/2 is binarized as SUBSTANTIVE = (code == 2), because the human's
  "2 = central, a variable or focus" is the target concept. A second run with
  SUBSTANTIVE = (code >= 1) is reported for comparison but is NOT the target.

INPUTS
  construct_presence.csv   pipeline binary flags (used only for the baseline)
  coder_2.xlsx             human codes, column A filename, V2_* columns
  corpus PDFs              needed to recount occurrences; set --corpus

USAGE
  python3 reoperationalize_constructs.py --corpus "/path/to/pdfs" --section counts
  python3 reoperationalize_constructs.py --section tune
  python3 reoperationalize_constructs.py --section heldout
"""

import os
import re
import sys
import csv
import json
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

SEED = 0
DEV_FRAC = 0.60

CONSTRUCT_PATTERNS = {
    "autonomy":                [r"autonomy", r"autonomous"],
    "trust":                   [r"trust"],
    "surveillance_monitoring": [r"surveillance", r"monitoring", r"bossware"],
    "psych_safety":            [r"psychological safety", r"psychological-safety"],
    "technostress":            [r"technostress", r"techno-?stress"],
    "isolation":               [r"isolation", r"isolated", r"loneliness"],
    "work_life_balance":       [r"work-?life balance", r"work-?family balance"],
    "leadership":              [r"leadership", r"leader"],
}
CONSTRUCTS = list(CONSTRUCT_PATTERNS.keys())


def kappa(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    pe = 0.0
    for v in (0, 1):
        pe += (a == v).mean() * (b == v).mean()
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def extract(path):
    try:
        import pdfplumber
        title_abs, full = [], []
        with pdfplumber.open(path) as pdf:
            for i, pg in enumerate(pdf.pages):
                t = pg.extract_text() or ""
                full.append(t)
                if i == 0:
                    title_abs.append(t[:3000])
        return "\n".join(full), "\n".join(title_abs)
    except Exception as e:
        sys.stderr.write("skip %s : %s\n" % (os.path.basename(path), e))
        return "", ""


def section_counts(corpus, outdir):
    pats = {k: [re.compile(r"\b" + p + r"\b", re.I) for p in v]
            for k, v in CONSTRUCT_PATTERNS.items()}
    pdfs = sorted(glob.glob(os.path.join(corpus, "**", "*.pdf"), recursive=True))
    if not pdfs:
        sys.stderr.write("no PDFs under %s\n" % corpus)
        sys.exit(1)
    rows = []
    for i, p in enumerate(pdfs):
        full, ta = extract(p)
        wc = max(len(full.split()), 1)
        row = {"filename": os.path.basename(p), "word_count": wc}
        for c in CONSTRUCTS:
            n_full = sum(len(rx.findall(full)) for rx in pats[c])
            n_ta = sum(len(rx.findall(ta)) for rx in pats[c])
            row["n_%s" % c] = n_full
            row["ta_%s" % c] = 1 if n_ta > 0 else 0
            row["d_%s" % c] = 1000.0 * n_full / wc
        rows.append(row)
        if (i + 1) % 25 == 0:
            sys.stderr.write("  counts %d/%d\n" % (i + 1, len(pdfs)))
    df = pd.DataFrame(rows)
    out = os.path.join(outdir, "construct_counts.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "papers")


def load_joined(outdir, target_level):
    counts = pd.read_csv(os.path.join(outdir, "construct_counts.csv"))
    hum = pd.read_excel(os.path.join(outdir, "coder_2.xlsx"), sheet_name="Sheet1")
    hum = hum.rename(columns={hum.columns[0]: "filename"})
    hum["filename"] = hum["filename"].astype(str).str.strip()
    counts["filename"] = counts["filename"].astype(str).str.strip()
    m = hum.merge(counts, on="filename")
    print("matched papers:", len(m))

    for c in CONSTRUCTS:
        col = "V2_%s" % c
        if col not in m.columns:
            sys.stderr.write("missing human column %s\n" % col)
            sys.exit(1)
        v = m[col].fillna(0).astype(float)
        m["y_%s" % c] = (v >= 2).astype(int) if target_level == "central" else (v >= 1).astype(int)
    return m


def split(m):
    rng = random.Random(SEED)
    idx = list(range(len(m)))
    rng.shuffle(idx)
    cut = int(len(idx) * DEV_FRAC)
    dev = sorted(idx[:cut])
    hel = sorted(idx[cut:])
    return m.iloc[dev].reset_index(drop=True), m.iloc[hel].reset_index(drop=True)


def candidates(df, c):
    out = {}
    out["c1_any"] = (df["n_%s" % c] >= 1).astype(int)
    for n in (2, 3, 5, 8, 12, 20):
        out["c2_freq_%d" % n] = (df["n_%s" % c] >= n).astype(int)
    out["c3_titleabs"] = df["ta_%s" % c].astype(int)
    for d in (0.1, 0.25, 0.5, 1.0, 2.0):
        out["c4_density_%s" % d] = (df["d_%s" % c] >= d).astype(int)
    for n in (3, 5, 8):
        out["c5_ta_or_freq_%d" % n] = ((df["ta_%s" % c] == 1) |
                                       (df["n_%s" % c] >= n)).astype(int)
    return out


def section_tune(outdir, target_level):
    m = load_joined(outdir, target_level)
    dev, hel = split(m)
    print("dev n=%d  heldout n=%d  (seed=%d, frac=%.2f)" % (len(dev), len(hel), SEED, DEV_FRAC))
    print("target: human code %s\n" % ("== 2 (central)" if target_level == "central" else ">= 1 (present)"))

    chosen = {}
    print(f"{'construct':24s} {'best rule on DEV':22s} {'dev kappa':>10s} {'baseline':>9s}")
    for c in CONSTRUCTS:
        y = dev["y_%s" % c].values
        cands = candidates(dev, c)
        base = kappa(y, cands["c1_any"].values)
        best_name, best_k = None, -9
        for name, pred in cands.items():
            k = kappa(y, pred.values)
            if not np.isnan(k) and k > best_k:
                best_name, best_k = name, k
        chosen[c] = best_name
        print(f"{c:24s} {best_name:22s} {best_k:10.3f} {base:9.3f}")

    with open(os.path.join(outdir, "chosen_rules.json"), "w") as f:
        json.dump({"target_level": target_level, "seed": SEED,
                   "dev_frac": DEV_FRAC, "rules": chosen}, f, indent=2)
    print("\nwrote chosen_rules.json")
    print("Rules are now LOCKED. Run --section heldout once. Do not retune.")


def section_heldout(outdir, target_level):
    rp = os.path.join(outdir, "chosen_rules.json")
    if not os.path.exists(rp):
        sys.stderr.write("run --section tune first\n")
        sys.exit(1)
    spec = json.load(open(rp))
    if spec["target_level"] != target_level:
        sys.stderr.write("target_level mismatch with locked rules\n")
        sys.exit(1)

    m = load_joined(outdir, target_level)
    dev, hel = split(m)
    print("HELD-OUT EVALUATION  n=%d" % len(hel))
    print("These numbers are the reportable validation result.\n")
    print(f"{'construct':24s} {'rule':22s} {'agree%':>7s} {'kappa':>8s} {'baseline':>9s}")
    ks, bs = [], []
    for c in CONSTRUCTS:
        y = hel["y_%s" % c].values
        cands = candidates(hel, c)
        rule = spec["rules"][c]
        pred = cands[rule].values
        k = kappa(y, pred)
        b = kappa(y, cands["c1_any"].values)
        agree = (y == pred).mean() * 100
        ks.append(k)
        bs.append(b)
        print(f"{c:24s} {rule:22s} {agree:6.1f} {k:8.3f} {b:9.3f}")
    print()
    print("MEAN held-out kappa  = %.3f" % np.nanmean(ks))
    print("MEAN baseline kappa  = %.3f" % np.nanmean(bs))
    print()
    print("Interpretation guide: >=0.61 substantial, 0.41-0.60 moderate,")
    print("0.21-0.40 fair, <=0.20 slight. If the held-out mean does not reach")
    print("0.41, the automated construct measure is not validated and construct")
    print("level claims should not enter the manuscript.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--section", required=True,
                    choices=["counts", "tune", "heldout"])
    ap.add_argument("--target", default="central", choices=["central", "present"])
    a = ap.parse_args()

    if a.section == "counts":
        if not a.corpus:
            sys.stderr.write("--corpus required for counts\n")
            sys.exit(1)
        section_counts(a.corpus, a.outdir)
    elif a.section == "tune":
        section_tune(a.outdir, a.target)
    else:
        section_heldout(a.outdir, a.target)


if __name__ == "__main__":
    main()
