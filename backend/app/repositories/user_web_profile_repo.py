from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserWebProfileORM


class UserWebProfileRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(
        self, profile_id: UUID, owner_id: UUID
    ) -> UserWebProfileORM | None:
        result = await self.session.execute(
            select(UserWebProfileORM).where(
                UserWebProfileORM.id == profile_id,
                UserWebProfileORM.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_owner(self, owner_id: UUID) -> Sequence[UserWebProfileORM]:
        # списков мало (их число ограничено), поэтому без постраничной выдачи
        result = await self.session.execute(
            select(UserWebProfileORM)
            .where(UserWebProfileORM.owner_id == owner_id)
            .order_by(UserWebProfileORM.created_at, UserWebProfileORM.id)
        )
        return result.scalars().all()

    async def count_for_owner(self, owner_id: UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(UserWebProfileORM)
            .where(UserWebProfileORM.owner_id == owner_id)
        )
        return result.scalar_one()

    async def create(self, profile: UserWebProfileORM) -> UserWebProfileORM:
        self.session.add(profile)
        await self.session.flush()
        return profile

    async def delete(self, profile: UserWebProfileORM) -> None:
        await self.session.delete(profile)
