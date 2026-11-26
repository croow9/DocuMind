from pathlib import Path

from app.ingestion.parser import parse_file, read_pdf


def test_pdf_file_text_is_joined_by_pages_and_empty_pages_are_dropped(
    tmp_path: Path,
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes("First page", "", "Third page"))

    assert parse_file(path) == "First page\n\nThird page"


def test_read_pdf_gives_title_and_page_texts_and_can_stop_early(
    tmp_path: Path,
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes("a", "b", "c", title="Spec"))

    assert read_pdf(str(path)) == ("Spec", ["a", "b", "c"])
    assert read_pdf(str(path), max_pages=2) == ("Spec", ["a", "b"])


def test_pdf_without_properties_has_an_empty_title(
    tmp_path: Path,
    pdf_bytes,  # type: ignore[no-untyped-def]
) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes("a"))

    assert read_pdf(str(path))[0] == ""
