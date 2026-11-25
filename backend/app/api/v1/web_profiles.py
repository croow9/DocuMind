from fastapi import APIRouter

from app.api.deps import UserDep, WebProfilesDep
from app.schemas.chat import WebProfileRead

router = APIRouter(prefix="/web-profiles", tags=["Web profiles"])


@router.get("", response_model=list[WebProfileRead])
async def list_web_profiles(
    _user: UserDep, profiles: WebProfilesDep
) -> list[WebProfileRead]:
    """Наборы сайтов для поиска в интернете: значение `name` идёт в `web_profile`."""
    return [
        WebProfileRead(
            name=profile.name,
            title=profile.title,
            description=profile.description,
            domains_count=len(profile.domains),
            is_default=profile is profiles.default,
        )
        for profile in profiles.all()
    ]
