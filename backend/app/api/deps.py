from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import Cache
from app.core.security import decode_access_token
from app.db.session import get_db
from app.ingestion.embedder import EmbeddingClient
from app.messaging.producer import KafkaProducerManager
from app.models import UserORM
from app.rag.llm import ChatClient
from app.rag.web_profiles import WebProfiles
from app.repositories.user_repo import UserRepository
from app.services.web_search_service import WebSearchService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

SessionDep = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: SessionDep,
) -> UserORM:
    try:
        user_id = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или просроченный токен",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = await UserRepository(db).get_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_producer(request: Request) -> KafkaProducerManager:
    return request.app.state.kafka_producer


def get_embedder(request: Request) -> EmbeddingClient:
    return request.app.state.embedder


def get_chat_client(request: Request) -> ChatClient:
    return request.app.state.chat_client


def get_web_search(request: Request) -> WebSearchService:
    return request.app.state.web_search


def get_web_profiles(request: Request) -> WebProfiles:
    return request.app.state.web_profiles


def get_cache(request: Request) -> Cache:
    return request.app.state.cache


UserDep = Annotated[UserORM, Depends(get_current_user)]
ProducerDep = Annotated[KafkaProducerManager, Depends(get_producer)]
EmbedderDep = Annotated[EmbeddingClient, Depends(get_embedder)]
ChatClientDep = Annotated[ChatClient, Depends(get_chat_client)]
WebSearchDep = Annotated[WebSearchService, Depends(get_web_search)]
WebProfilesDep = Annotated[WebProfiles, Depends(get_web_profiles)]
CacheDep = Annotated[Cache, Depends(get_cache)]
