from fastapi import APIRouter, status

from app.api.deps import SessionDep
from app.schemas.user import UserCreate, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
)
async def register(user_in: UserCreate, db: SessionDep) -> UserRead:
    service = AuthService(db)
    user = await service.register_new_user(user_in)
    return UserRead.model_validate(user)
