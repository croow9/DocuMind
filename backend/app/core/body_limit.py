from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.sizes import MEBIBYTE, human_size

# Запас на служебную часть multipart-запроса (границы, заголовки частей): файл
# размером ровно с лимит весит в запросе чуть больше. Точный размер файла проверяет
# сервис документов, а здесь только грубая защита от гигантских тел
MULTIPART_OVERHEAD_BYTES = MEBIBYTE


def _too_large_text(max_bytes: int) -> str:
    return f"Запрос слишком большой: допустимо до {human_size(max_bytes)}"


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None  # мусор в заголовке: проверит счётчик принятых байт
    return None


class BodySizeLimitMiddleware:
    """Отвечает 413 на запросы с телом больше max_bytes.

    Форму с файлом FastAPI разбирает целиком ещё до вызова нашей функции, так что
    проверкой внутри неё ни память, ни диск уже не защитить. Поэтому проверок две:
    заявленная длина (Content-Length) отклоняет честного клиента, не прочитав ни
    байта, а счётчик принятых байт останавливает того, кто длину не указал или
    соврал (например, передаёт тело кусками).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan и websocket не трогаем
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            response = JSONResponse(
                {"detail": _too_large_text(self.max_bytes)},
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
            await response(scope, receive, send)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # HTTPException из чтения тела FastAPI превращает в обычный
                    # ответ 413, а не в обрыв соединения
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=_too_large_text(self.max_bytes),
                    )
            return message

        await self.app(scope, counting_receive, send)
