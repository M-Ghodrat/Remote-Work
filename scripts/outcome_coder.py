#!/usr/bin/env python3
"""
outcome_coder.py
=================
Codebook-based outcome classifier for the remote/hybrid work corpus.

This script implements a deterministic, lexicon-based content coding of
outcome engagement across a corpus of research papers. It is fully
reproducible: given the same PDFs and the same codebook below, it
produces identical output. No machine-learning model is involved.

METHOD
------
For each paper, the script reads the first four pages of text (the region
containing the abstract, introduction, and typically the hypotheses /
framing), lowercases it, truncates at the references section, and records
the PRESENCE of each outcome construct as a binary indicator, based on
whether any of that construct's surface forms appears in the text.

This measures THEMATIC ENGAGEMENT (whether a paper engages an outcome),
not measurement (whether the outcome is an empirical dependent variable).
The distinction is deliberate and is stated as a limitation in the paper:
lexical presence is an upper bound on substantive treatment.

CODEBOOK
--------
Outcomes are organized into three categories, reflecting documented
coding decisions:

  * EMPLOYEE  : outcomes experienced by the individual worker
  * ORG       : outcomes accruing to the organization (firm or team).
                Note: productivity and job performance are coded here as
                organization-relevant despite individual measurement, on
                the rationale that they are the outcomes organizations
                value; engagement is coded EMPLOYEE but is dual in nature.
  * CONTROL   : organizational control / governance mechanisms
                (monitoring, evaluation, policy), distinct from outcomes.

A "firm-level" subset of ORG is defined separately to support the
firm-level coverage analysis (the hard, bottom-line outcomes).

USAGE
-----
  python outcome_coder.py
Expects:
  - corpus_index.csv         (column: filename)
  - rq1_analysis/communities.csv  (columns: filename, community)
  - PDFs under PDFDIR
Produces:
  - outcome_coding.csv       (one row per paper, binary outcome columns)
  - prints coverage summary and firm-level coverage
"""

import csv
import os
import re
from pathlib import Path
from collections import Counter
import pdfplumber

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
HERE = Path(os.environ.get("REMOTE_WORK_OUTPUT_DIR", Path(__file__).parent))
PDFDIR = Path(os.environ.get("REMOTE_WORK_PDF_DIR", Path.home() / "Desktop/MyResearch/RemoteWork"))
PAGES_READ = 4            # number of leading pages scanned per paper
CORPUS_INDEX = HERE / "corpus_index.csv"
COMMUNITIES = HERE / "rq1_analysis" / "communities.csv"
OUTPUT = HERE / "outcome_coding.csv"

# ---------------------------------------------------------------------------
# CODEBOOK
# Each construct maps to a list of surface forms. A construct is coded
# present (1) if ANY surface form occurs in the scanned text.
# ---------------------------------------------------------------------------
EMPLOYEE = {
    "emp_job_satisfaction": ["job satisfaction", "employee satisfaction", "work satisfaction"],
    "emp_engagement":       ["work engagement", "employee engagement", "job engagement"],
    "emp_wellbeing":        ["wellbeing", "well-being", "mental health", "psychological health"],
    "emp_burnout":          ["burnout", "exhaustion", "emotional exhaustion"],
    "emp_technostress":     ["technostress", "techno-stress", "digital fatigue"],
    "emp_stress":           ["stress", "strain"],
    "emp_work_life":        ["work-life balance", "work-life conflict", "work-family", "work\u2013life"],
    "emp_turnover_intent":  ["turnover intention", "intention to leave", "intention to quit"],
    "emp_isolation":        ["isolation", "loneliness"],
    "emp_autonomy":         ["autonomy", "job autonomy", "perceived control"],
    "emp_flexibility":      ["flexibility", "schedule flexibility", "flexible work"],
    "emp_belonging":        ["belonging", "sense of belonging", "inclusion"],
}

ORG = {
    "org_firm_performance": ["organizational performance", "organisational performance",
                             "firm performance", "business performance", "organizational effectiveness"],
    "org_financial":        ["financial performance", "profitability", "profit",
                             "return on investment", "roi", "revenue"],
    "org_cost":             ["cost saving", "cost-saving", "cost reduction",
                             "overhead cost", "operational cost"],
    "org_real_estate":      ["real estate", "office space", "office footprint", "lease", "premises"],
    "org_retention_rate":   ["employee retention", "retention rate", "turnover rate", "attrition"],
    "org_absenteeism":      ["absenteeism", "absence rate", "sickness absence"],
    "org_culture":          ["organizational culture", "organisational culture", "company culture"],
    "org_productivity":     ["productivity", "work output"],
    "org_job_performance":  ["job performance", "work performance", "task performance", "employee performance"],
    "org_commitment":       ["organizational commitment", "organisational commitment", "affective commitment"],
    "org_team_effective":   ["team effectiveness", "team performance", "team efficiency",
                             "virtual team performance"],
    "org_collaboration":    ["collaboration", "coordination", "teamwork", "knowledge sharing"],
    "org_innovation":       ["innovation", "innovative behavior", "innovative behaviour",
                             "creativity", "creative performance"],
    "org_project_success":  ["project success", "goal accomplishment", "project delivery"],
    "org_leadership":       ["leadership effectiveness", "e-leadership", "remote leadership", "supervisory"],
}

CONTROL = {
    "ctrl_monitoring":  ["monitoring", "surveillance", "electronic monitoring", "digital monitoring"],
    "ctrl_evaluation":  ["outcome-based evaluation", "results-based", "management by objectives",
                         "performance management"],
    "ctrl_policy":      ["remote work policy", "telework policy", "work-from-home mandate",
                         "work from home policy"],
    "ctrl_oversight":   ["oversight"],
}

# The hard, bottom-line firm outcomes (subset of ORG) used for the
# firm-level coverage analysis.
FIRM_LEVEL = ["org_firm_performance", "org_financial", "org_cost",
              "org_real_estate", "org_retention_rate", "org_absenteeism"]

# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------
REFS_HEADING = re.compile(r"^\s*(?:\d{1,2}[\.\)]?\s+)?(references|bibliography)\b", re.I)

def read_paper_text(path: Path) -> str:
    """Return lowercased text of the first PAGES_READ pages, truncated
    at the references heading. Returns '' if the PDF cannot be read."""
    try:
        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join((page.extract_text() or "")
                             for page in pdf.pages[:PAGES_READ])
    except Exception:
        return ""
    kept_lines = []
    for line in text.splitlines():
        if REFS_HEADING.match(line):
            break
        kept_lines.append(line)
    return re.sub(r"\s+", " ", " ".join(kept_lines)).lower()

def code_presence(text: str, codebook: dict) -> dict:
    """Return {construct: 0/1} for each construct in the codebook."""
    return {name: int(any(form in text for form in forms))
            for name, forms in codebook.items()}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    filenames = [r["filename"] for r in csv.DictReader(open(CORPUS_INDEX))]
    community = {r["filename"]: int(r["community"])
                 for r in csv.DictReader(open(COMMUNITIES))}

    all_constructs = list(EMPLOYEE) + list(ORG) + list(CONTROL)
    rows = []
    for i, name in enumerate(filenames):
        pdf_path = PDFDIR / name
        text = read_paper_text(pdf_path) if pdf_path.exists() else ""
        emp = code_presence(text, EMPLOYEE)
        org = code_presence(text, ORG)
        ctrl = code_presence(text, CONTROL)
        row = {"filename": name, "community": community.get(name, -1)}
        row.update(emp); row.update(org); row.update(ctrl)
        row["n_employee"] = sum(emp.values())
        row["n_org"] = sum(org.values())
        row["n_control"] = sum(ctrl.values())
        row["any_employee"] = int(row["n_employee"] > 0)
        row["any_org"] = int(row["n_org"] > 0)
        row["any_control"] = int(row["n_control"] > 0)
        row["any_firm_level"] = int(any(org[k] for k in FIRM_LEVEL))
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  coded {i + 1}/{len(filenames)} papers")

    cols = (["filename", "community"] + all_constructs +
            ["n_employee", "n_org", "n_control",
             "any_employee", "any_org", "any_control", "any_firm_level"])
    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ----- coverage summary -----
    N = len(rows)
    print(f"\nCoded {N} papers. Output: {OUTPUT}\n")
    print("Per-construct coverage (% of papers engaging the construct):")
    for c in sorted(all_constructs, key=lambda x: -sum(r[x] for r in rows)):
        pct = 100 * sum(r[c] for r in rows) / N
        print(f"  {c:24s} {pct:5.1f}%")
    print("\nFirm-level outcome coverage (the bottom-line subset):")
    for c in FIRM_LEVEL:
        pct = 100 * sum(r[c] for r in rows) / N
        print(f"  {c:24s} {pct:5.1f}%")
    corpus_avg = sum(100 * sum(r[c] for r in rows) / N
                     for c in all_constructs) / len(all_constructs)
    print(f"\nCorpus-average coverage across all constructs: {corpus_avg:.1f}%")

if __name__ == "__main__":
    main()
