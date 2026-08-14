from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Context:
    root: Path
    pdf_dir: Path
    output_dir: Path
    index_file: Path
    model: str = "specter"
    force: bool = False
    dry_run: bool = False

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"


def _run(ctx: Context, *args: object, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    command = [str(a) for a in args]
    print("+", " ".join(command))
    if ctx.dry_run:
        return
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(command, cwd=cwd, env=merged, check=True)


def _names(index_file: Path) -> list[str]:
    with index_file.open(newline="", encoding="utf-8-sig") as handle:
        # Preserve filename whitespace exactly: a few source PDFs genuinely begin
        # with a space and the historical index is aligned to those disk names.
        return [row["filename"] for row in csv.DictReader(handle)]


def prepare(ctx: Context) -> None:
    if not ctx.pdf_dir.is_dir():
        raise SystemExit(f"PDF directory does not exist: {ctx.pdf_dir}")
    names = _names(ctx.index_file)
    missing = [name for name in names if not (ctx.pdf_dir / name).is_file()]
    if missing:
        preview = "\n  - ".join(missing[:10])
        raise SystemExit(f"{len(missing)} indexed PDFs are missing:\n  - {preview}")
    if len(names) != len(set(names)):
        raise SystemExit("The corpus index contains duplicate filenames.")
    if ctx.dry_run:
        print(f"Would prepare corpus: {len(names)} indexed PDFs; 0 missing")
        return
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    target = ctx.output_dir / "corpus_index.csv"
    if ctx.force or not target.exists():
        shutil.copy2(ctx.index_file, target)
    print(f"Corpus ready: {len(names)} indexed PDFs; 0 missing")


def embed(ctx: Context) -> None:
    prepare(ctx)
    target = ctx.output_dir / f"embeddings_{ctx.model}.npy"
    if target.exists() and not ctx.force:
        print(f"Reuse existing embeddings: {target}")
        return
    _run(ctx, sys.executable, ctx.root / "src" / "remote_work" / "embeddings.py",
         "--index", ctx.output_dir / "corpus_index.csv", "--pdf-dir", ctx.pdf_dir,
         "--output", target, "--model", ctx.model)


def communities(ctx: Context) -> None:
    embed(ctx)
    expected = [ctx.output_dir / "rq1_analysis" / name for name in ("communities.csv", "centrality.csv")]
    if all(path.exists() for path in expected) and not ctx.force:
        print("Reuse existing community analysis")
        return
    _run(ctx, sys.executable, ctx.scripts / "analyze_communities.py", ctx.output_dir,
         "--pdfs", ctx.pdf_dir, "--model", ctx.model)


def constructs(ctx: Context) -> None:
    communities(ctx)
    if (ctx.output_dir / "rq3_constructs" / "summary.txt").exists() and not ctx.force:
        print("Reuse existing construct analysis")
        return
    _run(ctx, sys.executable, ctx.scripts / "rq3_constructs.py", ctx.output_dir,
         "--pdfs", ctx.pdf_dir)


def prevalence(ctx: Context) -> None:
    communities(ctx)
    if (ctx.output_dir / "rq4_prevalence" / "summary.txt").exists() and not ctx.force:
        print("Reuse existing prevalence analysis")
        return
    _run(ctx, sys.executable, ctx.scripts / "rq4_prevalence.py", ctx.output_dir,
         "--pdfs", ctx.pdf_dir, cwd=ctx.scripts)


def outcomes(ctx: Context) -> None:
    communities(ctx)
    if (ctx.output_dir / "outcome_coding.csv").exists() and not ctx.force:
        print("Reuse existing outcome coding")
        return
    env = {"REMOTE_WORK_OUTPUT_DIR": str(ctx.output_dir), "REMOTE_WORK_PDF_DIR": str(ctx.pdf_dir)}
    _run(ctx, sys.executable, ctx.scripts / "outcome_coder.py", env=env)


def jdr(ctx: Context) -> None:
    out = ctx.output_dir / "jdr"
    if (out / "jdr_construct_nodes.csv").exists() and not ctx.force:
        print("Reuse existing JD-R analysis")
        return
    _run(ctx, sys.executable, ctx.scripts / "build_jdr_construct_network.py",
         "--corpus", ctx.pdf_dir, "--outdir", out, "--section", "all")


def figures(ctx: Context) -> None:
    constructs(ctx)
    prevalence(ctx)
    outcomes(ctx)
    env = {"REMOTE_WORK_OUTPUT_DIR": str(ctx.output_dir), "REMOTE_WORK_PDF_DIR": str(ctx.pdf_dir)}
    if (ctx.output_dir / "outcome_figures" / "coverage_bar.png").exists() and not ctx.force:
        print("Reuse existing figures")
        return
    for script, args in (
        ("visualize_network.py", [ctx.output_dir]),
        ("visualize_rq3.py", [ctx.output_dir]),
        ("visualize_rq4.py", [ctx.output_dir]),
        ("visualize_similarity.py", []),
        ("outcome_figures.py", []),
        ("outcome_networks.py", []),
        ("firm_gap.py", []),
        ("level_descent.py", []),
    ):
        _run(ctx, sys.executable, ctx.scripts / script, *args, env=env)


STAGES = {
    "prepare": prepare,
    "embed": embed,
    "communities": communities,
    "constructs": constructs,
    "prevalence": prevalence,
    "outcomes": outcomes,
    "jdr": jdr,
    "figures": figures,
}


def run_all(ctx: Context) -> None:
    # Functions enforce their own prerequisites and reuse expensive embeddings.
    for name in ("prepare", "embed", "communities", "constructs", "prevalence", "outcomes", "jdr", "figures"):
        print(f"\n=== {name.upper()} ===")
        STAGES[name](ctx)
