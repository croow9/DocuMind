from pathlib import Path

from app.ingestion.chunker import MAX_CHUNK_LENGTH, split_into_chunks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_sample_doc_splits_by_headings() -> None:
    text = (FIXTURES / "sample_doc.md").read_text(encoding="utf-8")

    chunks = split_into_chunks(text)

    assert len(chunks) == 7
    assert chunks[3].heading_path == "Установка > Linux"
    # длинная секция Linux разрезана на два куска с тем же путём
    assert chunks[4].heading_path == "Установка > Linux"
    assert chunks[5].heading_path == "Установка > macOS"
    assert [c.ordinal for c in chunks] == list(range(7))
    assert all(len(c.content) <= MAX_CHUNK_LENGTH for c in chunks)


def test_empty_text_gives_no_chunks() -> None:
    assert split_into_chunks("") == []


def test_text_before_first_heading_has_empty_path() -> None:
    chunks = split_into_chunks("Вступление без заголовка.")

    assert len(chunks) == 1
    assert chunks[0].heading_path == ""
    assert chunks[0].content == "Вступление без заголовка."


def test_hash_without_space_is_not_a_heading() -> None:
    text = "#Заголовок без пробела не считается заголовком."

    chunks = split_into_chunks(text)

    assert len(chunks) == 1
    assert chunks[0].heading_path == ""


def test_top_level_heading_resets_nesting() -> None:
    text = "# Один\n\nтекст\n\n## Два\n\nтекст\n\n# Три\n\nтекст"

    chunks = split_into_chunks(text)

    assert [c.heading_path for c in chunks] == ["Один", "Один > Два", "Три"]
