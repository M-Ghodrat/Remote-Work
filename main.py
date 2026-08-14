#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from remote_work.pipeline import Context, STAGES, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the remote-work systematic-review pipeline.")
    parser.add_argument("stage", choices=["run-all", *STAGES])
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Folder containing the source PDFs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "canonical_389")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "corpus_index.csv")
    parser.add_argument("--model", choices=["specter", "mpnet"], default="specter")
    parser.add_argument("--force", action="store_true", help="Replace reusable generated inputs")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running them")
    args = parser.parse_args()

    ctx = Context(ROOT, args.pdf_dir.expanduser().resolve(), args.output_dir.expanduser().resolve(),
                  args.index.expanduser().resolve(), args.model, args.force, args.dry_run)
    if args.stage == "run-all":
        run_all(ctx)
    else:
        STAGES[args.stage](ctx)


if __name__ == "__main__":
    main()
