from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.auth import router as auth_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.core.config import settings
from app.db.session import engine
from app.ingestion.embedder import build_embedder
from app.messaging.producer import KafkaProducerManager
from app.rag.llm import build_chat_client
from app.services.web_search_service import (
    build_http_client,
    build_web_search_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    producer = KafkaProducerManager(bootstrap_servers=settings.kafka_bootstrap_servers)
    await producer.start()
    app.state.kafka_producer = producer
    # клиенты OpenAI держат пул соединений: создаём один раз, а не на каждый запрос
    app.state.embedder = build_embedder()
    app.state.chat_client = build_chat_client()
    http_client = build_http_client()
    app.state.web_search = build_web_search_service(app.state.embedder, http_client)

    try:
        yield
    finally:
        try:
            await producer.stop()
        finally:
            try:
                await http_client.aclose()
            finally:
                await engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")


@app.get("/")
def health() -> dict[str, str]:
    return {"message": "I'm a live!"}
