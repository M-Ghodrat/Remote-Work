# Remote Work Computational Review

A reproducible pipeline for the computational systematic review of remote and
hybrid work research. One entry point (`main.py`) validates the corpus, builds
document embeddings, detects paper communities, analyzes constructs and
benefit/risk prevalence, codes outcomes, builds the JD-R network, and renders
figures.

## Canonical corpus

`data/corpus_index.csv` is the reconciled 389-paper index. Every filename in
this index was verified against the source collection. Source PDFs are
copyrighted and are intentionally excluded from Git.

The older 387-paper repository supplied the most complete analysis code. The
389-paper index supplies the canonical corpus. The later 390-paper experiment
was not selected because its index contains a nonexistent `paper.pdf` entry.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python main.py run-all --pdf-dir "/absolute/path/to/RemoteWork/pdfs"
```

The default output directory is `outputs/canonical_389`. Existing expensive
artifacts are reused. Pass `--force` to rebuild them.

To inspect the execution plan without running analysis:

```bash
python main.py run-all --pdf-dir "/absolute/path/to/RemoteWork/pdfs" --dry-run
```

## Run one stage

```bash
python main.py prepare      --pdf-dir /path/to/pdfs
python main.py embed        --pdf-dir /path/to/pdfs
python main.py communities  --pdf-dir /path/to/pdfs
python main.py constructs   --pdf-dir /path/to/pdfs
python main.py prevalence   --pdf-dir /path/to/pdfs
python main.py outcomes     --pdf-dir /path/to/pdfs
python main.py jdr          --pdf-dir /path/to/pdfs
python main.py figures      --pdf-dir /path/to/pdfs
```

Each stage runs its prerequisites and reuses completed outputs. Both SPECTER
and MPNet can be selected with `--model specter|mpnet`; downstream historical
figures currently expect SPECTER.

## Pipeline

```text
PDFs + canonical index
        |
        v
document embeddings
        |
        v
paper kNN network + Leiden communities (RQ1/RQ2)
        |------------------|------------------|
        v                  v                  v
construct network     benefit/risk       outcome coding
(RQ3)                 co-presence (RQ4)  and firm gap
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                    figures and tables

PDFs ------------------------------------------------> JD-R construct network
```

## Reproducibility note

The original program that generated the historical embedding arrays was not
present in the research folder. `src/remote_work/embeddings.py` reconstructs
that missing stage using `allenai/specter` or
`sentence-transformers/all-mpnet-base-v2`, four leading PDF pages, normalized
embeddings, and index-order alignment. This is a documented reconstruction,
not a claim that the missing historical implementation is byte-identical.

The repository retains the research scripts under `scripts/` so the analytical
rules remain auditable. `main.py` supplies portable paths and orchestration;
no script depends on the former `~/Desktop/MyResearch/RemoteWork` location.

## Main outputs

- `rq1_analysis/`: communities, centralities, profiles
- `rq3_constructs/`: concept network and construct centralities
- `rq4_prevalence/`: benefit/risk co-presence and sensitivity thresholds
- `outcome_coding.csv`: per-paper outcome engagement
- `outcome_figures/` and `outcome_networks/`: outcome results
- `jdr/`: construct presence, JD-R nodes, edges, and figure

## Tests

```bash
python -m unittest discover -s tests
```

The tests cover the most important preflight guard: the corpus index must be
unique and every indexed PDF must exist before expensive analysis starts.
