from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


MODELS = {
    "specter": "allenai/specter",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def extract_text(path: Path, pages: int = 4, max_chars: int = 12000) -> str:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages[:pages])
    return " ".join(text.split())[:max_chars]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned document embeddings.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="specter")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    with args.index.open(newline="", encoding="utf-8-sig") as handle:
        names = [row["filename"] for row in csv.DictReader(handle)]
    texts = []
    for number, name in enumerate(names, 1):
        texts.append(extract_text(args.pdf_dir / name))
        if number % 50 == 0:
            print(f"Extracted {number}/{len(names)}")
    model = SentenceTransformer(MODELS[args.model])
    matrix = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(matrix, dtype=np.float32))
    print(f"Saved {matrix.shape} to {args.output}")


if __name__ == "__main__":
    main()
