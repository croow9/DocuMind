from dataclasses import dataclass

MAX_CHUNK_LENGTH = 1500
_SEPARATOR = " > "


@dataclass(frozen=True)
class ChunkData:
    """Кусок текста для индексации: содержимое и путь по заголовкам документа."""

    content: str
    heading_path: str
    ordinal: int


def split_into_chunks(text: str) -> list[ChunkData]:
    """Разбить markdown-текст на куски по заголовкам.

    Секция между заголовками — один кусок; секция длиннее MAX_CHUNK_LENGTH
    режется по абзацам. heading_path — заголовки, вложенные друг в друга,
    например «Установка > Linux».
    """
    chunks: list[ChunkData] = []
    headings: list[str] = []
    section: list[str] = []

    def flush() -> None:
        content = "\n".join(section).strip()
        section.clear()
        if not content:
            return
        path = _SEPARATOR.join(headings)
        for part in _split_long(content):
            chunks.append(
                ChunkData(content=part, heading_path=path, ordinal=len(chunks))
            )

    for line in text.splitlines():
        level = _heading_level(line)
        if level is None:
            section.append(line)
            continue
        flush()
        title = line.lstrip().lstrip("#").strip()
        # заголовок уровня N укорачивает стек до N-1 и сам становится текущим:
        # после '## B' заголовок '# A' сбрасывает вложенность до первого уровня
        headings[level - 1 :] = [title]

    flush()
    return chunks


def _heading_level(line: str) -> int | None:
    stripped = line.lstrip()
    marks = len(stripped) - len(stripped.lstrip("#"))
    if 1 <= marks <= 6 and stripped[marks : marks + 1] == " ":
        return marks
    return None


def _split_long(content: str) -> list[str]:
    if len(content) <= MAX_CHUNK_LENGTH:
        return [content]

    parts: list[str] = []
    current = ""
    for paragraph in content.split("\n\n"):
        if len(paragraph) > MAX_CHUNK_LENGTH:
            if current:
                parts.append(current)
                current = ""
            for i in range(0, len(paragraph), MAX_CHUNK_LENGTH):
                parts.append(paragraph[i : i + MAX_CHUNK_LENGTH])
            continue
        if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_LENGTH:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts
