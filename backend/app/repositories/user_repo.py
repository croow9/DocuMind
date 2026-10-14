from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserORM


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: UUID) -> UserORM | None:
        result = await self.session.execute(
            select(UserORM).where(UserORM.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, user_email: str) -> UserORM | None:
        result = await self.session.execute(
            select(UserORM).where(
                UserORM.email == user_email,
            )
        )
        return result.scalar_one_or_none()

    async def create(self, user: UserORM) -> UserORM:
        self.session.add(user)
        await self.session.flush()
        return user
