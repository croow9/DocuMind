from fastapi import FastAPI

from app.api.v1.auth import router as auth_router
from app.api.v1.documents import router as documents_router

app = FastAPI()

app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")


@app.get("/")
def health():
    return {"message": "I'm a live!"}
