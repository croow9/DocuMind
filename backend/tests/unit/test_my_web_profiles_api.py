from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from app.api.deps import get_current_user
from app.api.v1.my_web_profiles import get_user_web_profile_service
from app.main import app
from app.models import UserWebProfileORM

NOW = datetime(2026, 1, 1, tzinfo=UTC)
OWNER = SimpleNamespace(id=uuid4())
URL = "/api/v1/my-web-profiles"


def make_row() -> UserWebProfileORM:
    return UserWebProfileORM(
        id=uuid4(),
        owner_id=OWNER.id,
        title="Мой список",
        description="",
        domains=["docs.site.com"],
        created_at=NOW,
    )


@pytest.fixture
def service() -> Iterator[AsyncMock]:
    fake = AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: OWNER
    app.dependency_overrides[get_user_web_profile_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client(service: AsyncMock) -> TestClient:
    return TestClient(app)


def test_create_returns_201_and_the_list_with_its_domains(
    client: TestClient, service: AsyncMock
) -> None:
    row = make_row()
    service.create.return_value = row

    response = client.post(
        URL, json={"title": "Мой список", "domains": ["https://docs.site.com/a"]}
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": str(row.id),
        "title": "Мой список",
        "description": "",
        "domains": ["docs.site.com"],
        "created_at": "2026-01-01T00:00:00Z",
    }
    owner_id, payload = service.create.await_args.args
    assert owner_id == OWNER.id
    assert payload.domains == ["docs.site.com"]  # адрес уже сведён к домену


@pytest.mark.parametrize(
    "body",
    [
        {"title": "Т", "domains": []},
        {"title": "Т", "domains": ["printer.local"]},
        {"title": "", "domains": ["a.com"]},
        {"domains": ["a.com"]},
        {"title": "Т", "domains": ["a.com"], "owner_id": str(uuid4())},
    ],
)
def test_invalid_body_is_422_and_nothing_is_created(
    client: TestClient, service: AsyncMock, body: dict[str, object]
) -> None:
    response = client.post(URL, json=body)

    assert response.status_code == 422
    service.create.assert_not_awaited()


def test_limit_and_duplicate_errors_from_the_service_reach_the_client(
    client: TestClient, service: AsyncMock
) -> None:
    service.create.side_effect = HTTPException(409, "Список с таким названием уже есть")

    response = client.post(URL, json={"title": "Т", "domains": ["a.com"]})

    assert response.status_code == 409
    assert response.json() == {"detail": "Список с таким названием уже есть"}


def test_list_returns_the_owners_lists(client: TestClient, service: AsyncMock) -> None:
    service.list_profiles.return_value = [make_row(), make_row()]

    response = client.get(URL)

    assert response.status_code == 200
    assert len(response.json()) == 2
    service.list_profiles.assert_awaited_once_with(OWNER.id)


def test_get_one_and_404_for_a_foreign_list(
    client: TestClient, service: AsyncMock
) -> None:
    row = make_row()
    service.get.return_value = row
    assert client.get(f"{URL}/{row.id}").json()["id"] == str(row.id)

    service.get.side_effect = HTTPException(404, "Список сайтов не найден")
    response = client.get(f"{URL}/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Список сайтов не найден"}


def test_patch_passes_only_the_given_fields(
    client: TestClient, service: AsyncMock
) -> None:
    row = make_row()
    service.update.return_value = row

    response = client.patch(f"{URL}/{row.id}", json={"domains": ["b.com"]})

    assert response.status_code == 200
    owner_id, profile_id, payload = service.update.await_args.args
    assert (owner_id, profile_id) == (OWNER.id, row.id)
    assert payload.model_fields_set == {"domains"}


def test_empty_patch_is_422(client: TestClient, service: AsyncMock) -> None:
    response = client.patch(f"{URL}/{uuid4()}", json={})

    assert response.status_code == 422
    service.update.assert_not_awaited()


def test_delete_returns_204_without_body(
    client: TestClient, service: AsyncMock
) -> None:
    profile_id = uuid4()

    response = client.delete(f"{URL}/{profile_id}")

    assert response.status_code == 204 and response.content == b""
    service.delete.assert_awaited_once_with(OWNER.id, profile_id)


def test_not_a_uuid_in_the_path_is_422(client: TestClient) -> None:
    assert client.get(f"{URL}/not-a-uuid").status_code == 422


def test_every_endpoint_requires_a_logged_in_user() -> None:
    app.dependency_overrides.clear()
    client = TestClient(app)
    some_id = uuid4()

    responses = [
        client.post(URL, json={"title": "Т", "domains": ["a.com"]}),
        client.get(URL),
        client.get(f"{URL}/{some_id}"),
        client.patch(f"{URL}/{some_id}", json={"title": "Т"}),
        client.delete(f"{URL}/{some_id}"),
    ]

    assert [r.status_code for r in responses] == [401] * 5
