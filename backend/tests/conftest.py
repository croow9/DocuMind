import os
from collections.abc import Callable

import pytest

# Тесты не ходят в реальные сервисы — настройкам достаточно любых значений.
# setdefault не трогает переменные, уже заданные окружением или локальным .env,
# поэтому локально и в CI поведение одинаковое.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test"
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-with-at-least-32-bytes")


def build_pdf(*page_texts: str, title: str = "") -> bytes:
    """Минимальный настоящий PDF: по странице на каждый текст (только ASCII, без скобок).

    Рукописный, чтобы тестам не нужна была библиотека для создания PDF; pypdf
    читает его как обычный файл. Пустая строка даёт страницу без текста.
    """
    count = len(page_texts)
    font_id = 3 + 2 * count
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
    ]
    for i, text in enumerate(page_texts):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    info = ""
    if title:
        objects.append(f"<< /Title ({title}) >>".encode())
        info = f" /Info {len(objects)} 0 R"

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R{info} >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return out


@pytest.fixture
def pdf_bytes() -> Callable[..., bytes]:
    return build_pdf
