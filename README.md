# Computational Systematic Review of the Remote and Hybrid Work Literature

Code and derived data for a computational systematic review of the remote and
hybrid work literature (2022-2026), mapping a corpus of 387 papers using
document embeddings, community detection, concept-network analysis, and
codebook-based outcome coding.

## Pipeline

Run in order:

1. `analyze_communities.py`  - SPECTER embeddings, kNN graph, Leiden community detection (RQ1)
2. `rq3_constructs.py`       - concept co-occurrence network and construct centrality (RQ3)
3. `rq4_prevalence.py`       - per-paper benefit/risk co-presence (RQ4)
4. `outcome_coder.py`        - codebook-based outcome coding (coverage and firm-level gap)
5. `firm_gap.py`             - firm-level coverage figure
6. visualization scripts     - `visualize_network.py`, `visualize_rq3.py`, `visualize_rq4.py`

## Key outputs (derived data)

- `corpus_index.csv`            - the 387-paper corpus index
- `rq1_analysis/communities.csv`- community assignments
- `rq3_constructs/concept_centrality.csv` - construct centrality
- `rq4_prevalence/copresence_distribution.csv` - benefit/risk co-presence
- `outcome_coding.csv`          - per-paper outcome coding (from outcome_coder.py)

## Outcome codebook

The outcome codebook is defined explicitly inside `outcome_coder.py` as a set of
constructs, each with its surface forms, organized into employee-level,
organization-level, and organizational-control categories. A paper is coded as
engaging a construct when any of that construct's forms appears in the scanned
text. The method records thematic engagement (presence), not empirical
measurement; coverage figures are an upper bound on substantive treatment.

## Requirements

See `requirements.txt`. Core dependencies: sentence-transformers, umap-learn,
leidenalg, python-igraph, networkx, scikit-learn, scipy, numpy, pandas,
matplotlib, pdfplumber.

## Note on the corpus

The source PDFs are copyrighted and are not redistributed in this repository.
The corpus index lists the included papers by identifier.
