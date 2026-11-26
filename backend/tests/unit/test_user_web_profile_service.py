from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from app.models import UserWebProfileORM
from app.schemas.user_web_profile import UserWebProfileCreate, UserWebProfileUpdate
from app.services.user_web_profile_service import (
    DUPLICATE_TITLE_TEXT,
    MAX_PROFILES_PER_USER,
    NOT_FOUND_TEXT,
    UserWebProfileService,
    web_profile_from_row,
)

OWNER_ID = uuid4()
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_row(**overrides: object) -> UserWebProfileORM:
    data: dict[str, object] = {
        "id": uuid4(),
        "owner_id": OWNER_ID,
        "title": "Мой список",
        "description": "Описание",
        "domains": ["docs.site.com", "api.site.com"],
        "created_at": NOW,
    }
    return UserWebProfileORM(**{**data, **overrides})


def make_service(row: UserWebProfileORM | None = None, count: int = 0):  # type: ignore[no-untyped-def]
    service = UserWebProfileService(AsyncMock())
    service.profiles = AsyncMock()
    service.profiles.get_by_id.return_value = row
    service.profiles.count_for_owner.return_value = count
    return service


def duplicate_error() -> IntegrityError:
    return IntegrityError("INSERT", {}, Exception("duplicate key"))


# --- create


async def test_create_saves_the_list_for_its_owner_and_commits() -> None:
    service = make_service()
    payload = UserWebProfileCreate(
        title="Мой список",
        description="О Rust",
        domains=["https://doc.rust-lang.org/book"],
    )

    row = await service.create(OWNER_ID, payload)

    service.profiles.create.assert_awaited_once_with(row)
    service.session.commit.assert_awaited_once()
    assert (row.owner_id, row.title, row.description, row.domains) == (
        OWNER_ID,
        "Мой список",
        "О Rust",
        ["doc.rust-lang.org"],
    )


async def test_the_last_allowed_list_can_still_be_created() -> None:
    service = make_service(count=MAX_PROFILES_PER_USER - 1)

    await service.create(OWNER_ID, UserWebProfileCreate(title="Т", domains=["a.com"]))

    service.profiles.create.assert_awaited_once()


async def test_limit_of_lists_is_409_and_nothing_is_created() -> None:
    service = make_service(count=MAX_PROFILES_PER_USER)

    with pytest.raises(HTTPException) as exc_info:
        await service.create(
            OWNER_ID, UserWebProfileCreate(title="Т", domains=["a.com"])
        )

    assert exc_info.value.status_code == 409
    assert str(MAX_PROFILES_PER_USER) in exc_info.value.detail
    service.profiles.create.assert_not_awaited()
    service.session.commit.assert_not_awaited()


async def test_the_limit_is_counted_for_this_owner_only() -> None:
    service = make_service()

    await service.create(OWNER_ID, UserWebProfileCreate(title="Т", domains=["a.com"]))

    service.profiles.count_for_owner.assert_awaited_once_with(OWNER_ID)


@pytest.mark.parametrize("fails_on", ["create", "commit"])
async def test_duplicate_title_is_409_and_the_transaction_is_rolled_back(
    fails_on: str,
) -> None:
    service = make_service()
    if fails_on == "create":
        service.profiles.create.side_effect = duplicate_error()
    else:
        service.session.commit.side_effect = duplicate_error()

    with pytest.raises(HTTPException) as exc_info:
        await service.create(
            OWNER_ID, UserWebProfileCreate(title="Т", domains=["a.com"])
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == DUPLICATE_TITLE_TEXT
    service.session.rollback.assert_awaited_once()


# --- get / list


async def test_get_returns_the_owners_list() -> None:
    row = make_row()
    service = make_service(row)

    assert await service.get(OWNER_ID, row.id) is row
    service.profiles.get_by_id.assert_awaited_once_with(row.id, OWNER_ID)


async def test_foreign_or_missing_list_is_404() -> None:
    service = make_service(None)

    with pytest.raises(HTTPException) as exc_info:
        await service.get(OWNER_ID, uuid4())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == NOT_FOUND_TEXT


async def test_list_is_asked_for_the_owner() -> None:
    service = make_service()
    service.profiles.list_for_owner.return_value = [make_row()]

    assert len(await service.list_profiles(OWNER_ID)) == 1
    service.profiles.list_for_owner.assert_awaited_once_with(OWNER_ID)


# --- update


async def test_update_changes_only_the_given_fields() -> None:
    row = make_row()
    service = make_service(row)

    await service.update(OWNER_ID, row.id, UserWebProfileUpdate(title="Новое имя"))

    assert (row.title, row.description, row.domains) == (
        "Новое имя",
        "Описание",
        ["docs.site.com", "api.site.com"],
    )
    service.session.commit.assert_awaited_once()


async def test_update_replaces_the_domains_as_a_whole() -> None:
    row = make_row()
    service = make_service(row)

    await service.update(
        OWNER_ID, row.id, UserWebProfileUpdate(domains=["only.site.com"])
    )

    assert row.domains == ["only.site.com"]
    # название и описание при этом остались как были
    assert (row.title, row.description) == ("Мой список", "Описание")


async def test_update_can_clear_the_description() -> None:
    row = make_row()
    service = make_service(row)

    await service.update(OWNER_ID, row.id, UserWebProfileUpdate(description=""))

    assert row.description == ""


async def test_update_of_a_foreign_list_is_404_and_commits_nothing() -> None:
    service = make_service(None)

    with pytest.raises(HTTPException) as exc_info:
        await service.update(OWNER_ID, uuid4(), UserWebProfileUpdate(title="Т"))

    assert exc_info.value.status_code == 404
    service.session.commit.assert_not_awaited()


async def test_update_to_an_existing_title_is_409_with_rollback() -> None:
    row = make_row()
    service = make_service(row)
    service.session.commit.side_effect = duplicate_error()

    with pytest.raises(HTTPException) as exc_info:
        await service.update(OWNER_ID, row.id, UserWebProfileUpdate(title="Занято"))

    assert exc_info.value.status_code == 409
    service.session.rollback.assert_awaited_once()


# --- delete


async def test_delete_removes_the_list_and_commits() -> None:
    row = make_row()
    service = make_service(row)

    await service.delete(OWNER_ID, row.id)

    service.profiles.delete.assert_awaited_once_with(row)
    service.session.commit.assert_awaited_once()


async def test_delete_of_a_foreign_list_is_404_and_deletes_nothing() -> None:
    service = make_service(None)

    with pytest.raises(HTTPException) as exc_info:
        await service.delete(OWNER_ID, uuid4())

    assert exc_info.value.status_code == 404
    service.profiles.delete.assert_not_awaited()
    service.session.commit.assert_not_awaited()


# --- превращение строки БД в профиль для поиска


def test_row_becomes_a_profile_that_never_fetches_pages_itself() -> None:
    row = make_row()

    profile = web_profile_from_row(row)

    assert profile.fetch_pages is False
    assert profile.domains == ("docs.site.com", "api.site.com")
    assert (profile.title, profile.description) == ("Мой список", "Описание")


def test_profile_name_is_stable_unique_per_list_and_never_a_file_profile_name() -> None:
    first, second = make_row(), make_row()

    assert web_profile_from_row(first).name == web_profile_from_row(first).name
    assert web_profile_from_row(first).name != web_profile_from_row(second).name
    assert web_profile_from_row(first).name.startswith("my-")
