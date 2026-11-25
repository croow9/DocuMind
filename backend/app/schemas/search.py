from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RetrievedChunk(BaseModel):
    """Кусок документа, найденный поиском: всё, что нужно для промпта и цитаты.

    Векторов и служебных колонок здесь нет: после поиска они не нужны, а вектор
    это 1536 чисел на каждый чанк.
    """

    # from_attributes: объект собирается прямо из строки результата SQL-запроса
    model_config = ConfigDict(frozen=True, from_attributes=True)

    chunk_id: UUID
    document_id: UUID
    document_title: str
    heading_path: str | None
    content: str
