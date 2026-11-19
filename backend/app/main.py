from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.auth import router as auth_router
from app.api.v1.documents import router as documents_router
from app.core.config import settings
from app.db.session import engine
from app.messaging.producer import KafkaProducerManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    producer = KafkaProducerManager(bootstrap_servers=settings.kafka_bootstrap_servers)
    await producer.start()
    app.state.kafka_producer = producer

    try:
        yield
    finally:
        try:
            await producer.stop()
        finally:
            await engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")


@app.get("/")
def health() -> dict[str, str]:
    return {"message": "I'm a live!"}
