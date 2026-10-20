from pathlib import Path

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


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = (page.extract_text() or "" for page in reader.pages)
    return "\n\n".join(page.strip() for page in pages if page.strip())
