from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.deps import SessionDep, UserDep
from app.schemas.user_web_profile import (
    UserWebProfileCreate,
    UserWebProfileRead,
    UserWebProfileUpdate,
)
from app.services.user_web_profile_service import UserWebProfileService

router = APIRouter(prefix="/my-web-profiles", tags=["My web profiles"])


def get_user_web_profile_service(db: SessionDep) -> UserWebProfileService:
    return UserWebProfileService(db)


ServiceDep = Annotated[UserWebProfileService, Depends(get_user_web_profile_service)]


@router.post("", response_model=UserWebProfileRead, status_code=status.HTTP_201_CREATED)
async def create_my_web_profile(
    payload: UserWebProfileCreate, user: UserDep, service: ServiceDep
) -> UserWebProfileRead:
    """Свой список сайтов для поиска в интернете.

    Домены можно вставлять и адресами страниц: `https://docs.site.com/guide`
    сохранится как `docs.site.com`. Поиск идёт только по публичным сайтам: закрытые
    и внутренние поисковик не увидит. Id списка идёт в `my_web_profile_id` вопроса.
    """
    return UserWebProfileRead.model_validate(await service.create(user.id, payload))


@router.get("", response_model=list[UserWebProfileRead])
async def list_my_web_profiles(
    user: UserDep, service: ServiceDep
) -> list[UserWebProfileRead]:
    profiles = await service.list_profiles(user.id)
    return [UserWebProfileRead.model_validate(p) for p in profiles]


@router.get("/{profile_id}", response_model=UserWebProfileRead)
async def get_my_web_profile(
    profile_id: UUID, user: UserDep, service: ServiceDep
) -> UserWebProfileRead:
    return UserWebProfileRead.model_validate(await service.get(user.id, profile_id))


@router.patch("/{profile_id}", response_model=UserWebProfileRead)
async def update_my_web_profile(
    profile_id: UUID,
    payload: UserWebProfileUpdate,
    user: UserDep,
    service: ServiceDep,
) -> UserWebProfileRead:
    """Меняет только переданные поля; `domains` заменяет список целиком."""
    return UserWebProfileRead.model_validate(
        await service.update(user.id, profile_id, payload)
    )


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_web_profile(
    profile_id: UUID, user: UserDep, service: ServiceDep
) -> None:
    await service.delete(user.id, profile_id)
