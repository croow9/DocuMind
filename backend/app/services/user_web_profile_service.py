from collections.abc import Sequence
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserWebProfileORM
from app.rag.web_profiles import WebProfile
from app.repositories.user_web_profile_repo import UserWebProfileRepository
from app.schemas.user_web_profile import UserWebProfileCreate, UserWebProfileUpdate

# Защита от бесконтрольного роста: у списков нет стоимости, но каждый можно
# выбрать в вопросе, а значит, это ещё и нагрузка на платный поиск
MAX_PROFILES_PER_USER = 10

NOT_FOUND_TEXT = "Список сайтов не найден"
DUPLICATE_TITLE_TEXT = "Список с таким названием уже есть"


def web_profile_from_row(row: UserWebProfileORM) -> WebProfile:
    """Личный список в виде профиля, с которым умеет работать поиск."""
    return WebProfile(
        # имя нужно только для логов; из id, чтобы не совпасть с профилями из файла
        name=f"my-{row.id.hex[:12]}",
        title=row.title,
        description=row.description,
        domains=tuple(row.domains),
        # доменам пользователя мы не доверяем настолько, чтобы ходить на них сами
        fetch_pages=False,
    )


class UserWebProfileService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.profiles = UserWebProfileRepository(session)

    async def create(
        self, owner_id: UUID, payload: UserWebProfileCreate
    ) -> UserWebProfileORM:
        # Проверка и вставка не атомарны: два одновременных запроса могут
        # проскочить лимит. Для защиты от «слишком много» это терпимо
        if await self.profiles.count_for_owner(owner_id) >= MAX_PROFILES_PER_USER:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Можно создать не больше {MAX_PROFILES_PER_USER} списков",
            )
        profile = UserWebProfileORM(
            owner_id=owner_id,
            title=payload.title,
            description=payload.description,
            domains=payload.domains,
        )
        try:
            await self.profiles.create(profile)
            await self.session.commit()
        except IntegrityError:
            await self._duplicate_title()
        return profile

    async def list_profiles(self, owner_id: UUID) -> Sequence[UserWebProfileORM]:
        return await self.profiles.list_for_owner(owner_id)

    async def get(self, owner_id: UUID, profile_id: UUID) -> UserWebProfileORM:
        profile = await self.profiles.get_by_id(profile_id, owner_id)
        if profile is None:
            # чужой список неотличим от несуществующего: не раскрываем его наличие
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND_TEXT
            )
        return profile

    async def update(
        self, owner_id: UUID, profile_id: UUID, payload: UserWebProfileUpdate
    ) -> UserWebProfileORM:
        profile = await self.get(owner_id, profile_id)
        if payload.title is not None:
            profile.title = payload.title
        if payload.description is not None:
            profile.description = payload.description
        if payload.domains is not None:
            profile.domains = payload.domains
        try:
            await self.session.commit()
        except IntegrityError:
            await self._duplicate_title()
        return profile

    async def delete(self, owner_id: UUID, profile_id: UUID) -> None:
        profile = await self.get(owner_id, profile_id)
        await self.profiles.delete(profile)
        await self.session.commit()

    async def _duplicate_title(self) -> None:
        await self.session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=DUPLICATE_TITLE_TEXT
        )
