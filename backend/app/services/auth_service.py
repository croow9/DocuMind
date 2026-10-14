from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models import UserORM
from app.repositories.user_repo import UserRepository
from app.schemas.user import UserCreate


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)

    async def register_new_user(self, user_data: UserCreate) -> UserORM:
        hashed_pwd = hash_password(user_data.password)

        new_user = UserORM(email=user_data.email, hashed_password=hashed_pwd)

        try:
            await self.user_repo.create(new_user)
            await self.session.commit()
        except IntegrityError as e:
            await self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Пользователь с таким email уже зарегистрирован",
            ) from e

        return new_user

    async def authenticate_user(self, email: str, password_raw: str) -> str:
        user = await self.user_repo.get_by_email(email)

        if user is None or not verify_password(password_raw, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный email или пароль",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return create_access_token(user.id)
