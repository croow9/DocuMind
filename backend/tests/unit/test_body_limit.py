from collections.abc import Iterator

import pytest
from fastapi import FastAPI, HTTPException, Request, UploadFile
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from app.core.body_limit import MULTIPART_OVERHEAD_BYTES, BodySizeLimitMiddleware
from app.core.config import settings
from app.main import app as real_app

LIMIT = 200


def make_app(reached: list[int]) -> FastAPI:
    """Маленькое приложение с ограничителем; reached запоминает, что дошло до функции."""
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=LIMIT)

    @app.post("/raw")
    async def raw(request: Request) -> dict[str, int]:
        body = await request.body()
        reached.append(len(body))
        return {"size": len(body)}

    @app.post("/upload")
    async def upload(file: UploadFile) -> dict[str, int]:
        content = await file.read()
        reached.append(len(content))
        return {"size": len(content)}

    return app


@pytest.fixture
def reached() -> list[int]:
    return []


@pytest.fixture
def client(reached: list[int]) -> Iterator[TestClient]:
    yield TestClient(make_app(reached))


def chunks(total: int, step: int = 10) -> Iterator[bytes]:
    """Тело кусками: у такого запроса нет Content-Length, длина клиентом не заявлена."""
    sent = 0
    while sent < total:
        piece = min(step, total - sent)
        yield b"x" * piece
        sent += piece


# --- заявленная длина (Content-Length)


def test_a_body_exactly_at_the_limit_passes(client: TestClient) -> None:
    response = client.post("/raw", content=b"x" * LIMIT)

    assert response.status_code == 200
    assert response.json() == {"size": LIMIT}


def test_a_declared_length_over_the_limit_is_413_and_the_handler_never_runs(
    client: TestClient, reached: list[int]
) -> None:
    response = client.post("/raw", content=b"x" * (LIMIT + 1))

    assert response.status_code == 413
    assert reached == []


def test_the_413_answer_is_json_with_a_readable_reason(client: TestClient) -> None:
    response = client.post("/raw", content=b"x" * (LIMIT + 1))

    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Запрос слишком большой: допустимо до 200 Б"}


# --- тело без заявленной длины или с неправдой в ней


def test_a_chunked_body_at_the_limit_passes(client: TestClient) -> None:
    response = client.post("/raw", content=chunks(LIMIT))

    assert response.status_code == 200
    assert response.json() == {"size": LIMIT}


def test_a_chunked_body_over_the_limit_is_413_and_the_handler_never_finishes(
    client: TestClient, reached: list[int]
) -> None:
    response = client.post("/raw", content=chunks(LIMIT + 1))

    assert response.status_code == 413
    assert response.json() == {"detail": "Запрос слишком большой: допустимо до 200 Б"}
    assert reached == []


def multipart(size: int) -> tuple[bytes, dict[str, str]]:
    boundary = "testboundary"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + b"x" * size + tail
    return body, {"content-type": f"multipart/form-data; boundary={boundary}"}


def test_a_chunked_file_form_over_the_limit_is_a_clean_413_not_a_500(
    client: TestClient, reached: list[int]
) -> None:
    # форму с файлом фреймворк разбирает до нашей функции: исключение из чтения тела
    # должно стать обычным ответом, а не оборванным соединением или 500
    body, headers = multipart(LIMIT)  # вместе со служебной частью заведомо больше

    response = client.post("/upload", content=chunks_of(body), headers=headers)

    assert response.status_code == 413
    assert reached == []


def test_a_chunked_file_form_within_the_limit_is_read_normally(
    client: TestClient, reached: list[int]
) -> None:
    body, headers = multipart(10)
    assert len(body) <= LIMIT

    response = client.post("/upload", content=chunks_of(body), headers=headers)

    assert response.status_code == 200
    assert response.json() == {"size": 10}


def chunks_of(body: bytes, step: int = 10) -> Iterator[bytes]:
    for start in range(0, len(body), step):
        yield body[start : start + step]


# --- прямая проверка на уровне ASGI


def http_scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    return {"type": "http", "headers": headers}


async def drain(middleware: BodySizeLimitMiddleware, scope: Scope, receive: Receive):
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


async def test_a_lying_content_length_does_not_help_the_client() -> None:
    # заявлено 5 байт, а приходит 250: ограничитель считает то, что пришло
    seen: list[Message] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(await receive())

    async def receive() -> Message:
        return {"type": "http.request", "body": b"x" * 250, "more_body": False}

    middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)

    with pytest.raises(HTTPException) as exc_info:
        await drain(middleware, http_scope([(b"content-length", b"5")]), receive)

    assert exc_info.value.status_code == 413
    assert seen == []


async def test_a_declared_length_over_the_limit_is_answered_without_reading_a_byte() -> (
    None
):
    inner_calls: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        inner_calls.append("вызвано")

    async def receive() -> Message:
        raise AssertionError("тело не должно читаться")

    middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)
    header = [(b"content-length", str(LIMIT + 1).encode())]

    sent = await drain(middleware, http_scope(header), receive)

    assert inner_calls == []
    assert sent[0]["status"] == 413


async def test_the_bytes_of_several_messages_add_up() -> None:
    # тело приходит кусками по 120 байт при лимите 200: по отдельности каждый кусок
    # допустим, а вместе они превышают лимит
    pieces = iter([True, True, True, True, False])  # more_body у пяти кусков

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        while (await receive())["more_body"]:
            pass

    async def receive() -> Message:
        return {"type": "http.request", "body": b"x" * 120, "more_body": next(pieces)}

    middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)

    with pytest.raises(HTTPException) as exc_info:
        await drain(middleware, http_scope([]), receive)

    assert exc_info.value.status_code == 413


async def test_a_garbage_content_length_is_ignored_and_the_body_is_counted() -> None:
    reads: list[Message] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        reads.append(await receive())

    async def receive() -> Message:
        return {"type": "http.request", "body": b"ok", "more_body": False}

    middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)

    await drain(middleware, http_scope([(b"content-length", b"abc")]), receive)

    assert reads == [{"type": "http.request", "body": b"ok", "more_body": False}]


async def test_non_http_scopes_pass_through_untouched() -> None:
    called: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        called.append(scope["type"])

    async def receive() -> Message:
        raise AssertionError("не должен читаться")

    middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)

    # у lifespan нет ни заголовков, ни тела
    await drain(middleware, {"type": "lifespan"}, receive)

    assert called == ["lifespan"]


# --- как ограничитель подключён в настоящем приложении


def test_the_real_app_allows_a_file_of_the_limit_plus_multipart_overhead() -> None:
    [middleware] = [m for m in real_app.user_middleware]

    assert middleware.cls is BodySizeLimitMiddleware
    assert middleware.kwargs["max_bytes"] == (
        settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    )


def test_the_real_app_rejects_a_huge_upload_before_authentication_and_the_database() -> (
    None
):
    # без токена обычный запрос получил бы 401: 413 значит, что отказ случился раньше
    too_big = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES + 1

    response = TestClient(real_app).post(
        "/api/v1/documents/upload",
        content=b"x" * too_big,
        headers={"content-type": "multipart/form-data; boundary=b"},
    )

    assert response.status_code == 413
