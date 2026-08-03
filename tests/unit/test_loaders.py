from unittest.mock import patch

from src.self_rag.ingestion.loaders import pdf_paths


def test_pdf_paths_finds_files_in_subdirectories_even_with_top_level_files(tmp_path):
    """Regression test: pdf_paths() used to prefer top-level *.pdf files and only
    fall back to a recursive search when none existed at the top level - so once
    a single file was uploaded directly into data_dir, PDFs in a subdirectory
    (e.g. data/raw_pdfs/) were silently excluded from every rebuild."""
    (tmp_path / "raw_pdfs").mkdir()
    nested = tmp_path / "raw_pdfs" / "policy.pdf"
    nested.write_bytes(b"%PDF-1.4 fake")
    top_level = tmp_path / "uploaded.pdf"
    top_level.write_bytes(b"%PDF-1.4 fake")

    with patch("src.self_rag.ingestion.loaders.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        found = pdf_paths()

    assert set(found) == {nested, top_level}


def test_pdf_paths_returns_empty_list_when_data_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with patch("src.self_rag.ingestion.loaders.settings") as mock_settings:
        mock_settings.data_dir = missing
        assert pdf_paths() == []
