from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def parse_file(path: Path) -> str:
    """Извлечь чистый текст из файла документа, стратегия — по расширению."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"Неподдерживаемый формат файла: {suffix or 'без расширения'}")


def read_pdf(
    source: str | BinaryIO, max_pages: int | None = None
) -> tuple[str, list[str]]:
    """Название из свойств PDF (или пустая строка) и текст каждой из страниц.

    Страница без текстового слоя (скан) даёт пустую строку. max_pages обрезает
    разбор: у огромного документа остальное не нужно, а читать его долго.
    """
    reader = PdfReader(source)
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    title = (reader.metadata.title if reader.metadata else None) or ""
    return title.strip(), [(page.extract_text() or "").strip() for page in pages]


def _parse_pdf(path: Path) -> str:
    _, pages = read_pdf(str(path))
    return "\n\n".join(page for page in pages if page)
