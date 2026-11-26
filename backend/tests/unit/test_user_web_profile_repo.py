from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy.sql import Select

from app.models import UserWebProfileORM
from app.repositories.user_web_profile_repo import UserWebProfileRepository

OWNER_ID = uuid4()


def make_repo() -> tuple[UserWebProfileRepository, AsyncMock]:
    session = AsyncMock()
    session.add = MagicMock()  # в настоящей сессии add обычный, не async
    session.execute.return_value = MagicMock()
    return UserWebProfileRepository(session), session


def executed(session: AsyncMock) -> Select:  # type: ignore[type-arg]
    [call] = session.execute.await_args_list
    return call.args[0]


def bound_values(statement: Select) -> list[object]:  # type: ignore[type-arg]
    return list(statement.compile().params.values())


# Проверка владельца — единственное, что отделяет чужой список от своего. Сессия
# здесь подставная, поэтому смотрим на сам запрос: в нём должно быть условие по
# владельцу и именно с тем id, который передан


async def test_get_by_id_is_limited_to_the_owner() -> None:
    repo, session = make_repo()
    profile_id = uuid4()

    await repo.get_by_id(profile_id, OWNER_ID)

    statement = executed(session)
    assert "user_web_profiles.owner_id = " in str(statement)
    assert "user_web_profiles.id = " in str(statement)
    assert set(bound_values(statement)) == {profile_id, OWNER_ID}


async def test_listing_is_limited_to_the_owner() -> None:
    repo, session = make_repo()

    await repo.list_for_owner(OWNER_ID)

    statement = executed(session)
    assert "user_web_profiles.owner_id = " in str(statement)
    assert bound_values(statement) == [OWNER_ID]


async def test_counting_is_limited_to_the_owner() -> None:
    repo, session = make_repo()

    await repo.count_for_owner(OWNER_ID)

    statement = executed(session)
    assert "count" in str(statement).lower()
    assert "user_web_profiles.owner_id = " in str(statement)
    assert bound_values(statement) == [OWNER_ID]


async def test_create_adds_the_row_and_flushes_to_get_database_errors_early() -> None:
    repo, session = make_repo()
    row = UserWebProfileORM(owner_id=OWNER_ID, title="Т", domains=["a.com"])

    assert await repo.create(row) is row

    session.add.assert_called_once_with(row)
    session.flush.assert_awaited_once()


async def test_delete_hands_the_row_to_the_session() -> None:
    repo, session = make_repo()
    row = UserWebProfileORM(owner_id=OWNER_ID, title="Т", domains=["a.com"])

    await repo.delete(row)

    session.delete.assert_awaited_once_with(row)
