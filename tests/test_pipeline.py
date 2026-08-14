import csv
import tempfile
import unittest
from pathlib import Path

from remote_work.pipeline import Context, prepare


def make_context(tmp_path: Path, filenames: list[str]) -> Context:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "filename"])
        writer.writerows(enumerate(filenames))
    return Context(tmp_path, pdf_dir, tmp_path / "out", index)


class PrepareTests(unittest.TestCase):
    def test_prepare_copies_aligned_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = make_context(Path(directory), ["a.pdf", "b.pdf"])
            for name in ("a.pdf", "b.pdf"):
                (ctx.pdf_dir / name).touch()
            prepare(ctx)
            self.assertEqual((ctx.output_dir / "corpus_index.csv").read_bytes(), ctx.index_file.read_bytes())

    def test_prepare_rejects_missing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = make_context(Path(directory), ["missing.pdf"])
            with self.assertRaisesRegex(SystemExit, "1 indexed PDFs are missing"):
                prepare(ctx)

    def test_prepare_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = make_context(Path(directory), ["same.pdf", "same.pdf"])
            (ctx.pdf_dir / "same.pdf").touch()
            with self.assertRaisesRegex(SystemExit, "duplicate"):
                prepare(ctx)


if __name__ == "__main__":
    unittest.main()
