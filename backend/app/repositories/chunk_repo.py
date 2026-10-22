from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChunkORM


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_all(self, chunks: Sequence[ChunkORM]) -> None:
        self.session.add_all(chunks)
        await self.session.flush()
