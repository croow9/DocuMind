from pathlib import Path
from uuid import UUID

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def document_file_path(document_id: UUID, suffix: str) -> Path:
    """Путь хранения файла документа: data/<id документа><оригинальное расширение>."""
    return DATA_DIR / f"{document_id}{suffix}"
