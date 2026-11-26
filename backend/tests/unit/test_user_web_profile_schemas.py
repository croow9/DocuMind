import pytest
from pydantic import ValidationError

from app.rag.web_profiles import MAX_DOMAINS_PER_USER_PROFILE
from app.schemas.chat import MessageCreate
from app.schemas.user_web_profile import UserWebProfileCreate, UserWebProfileUpdate


def create(**overrides: object) -> UserWebProfileCreate:
    data: dict[str, object] = {"title": "Мой список", "domains": ["docs.site.com"]}
    return UserWebProfileCreate.model_validate({**data, **overrides})


def test_pasted_addresses_become_domains_and_repeats_collapse() -> None:
    profile = create(
        domains=[
            "https://Docs.Site.com/guide",
            "docs.site.com",
            "  api.site.com/v1 ",
            "DOCS.site.com.",
        ]
    )

    assert profile.domains == ["docs.site.com", "api.site.com"]


def test_title_and_description_are_trimmed_and_description_is_optional() -> None:
    profile = create(title="  Мой список  ")

    assert profile.title == "Мой список" and profile.description == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": ""},
        {"title": "   "},
        {"title": "x" * 101},
        {"description": "x" * 501},
        {"domains": []},
        {"domains": ["not a domain"]},
        {"domains": ["docs.site.com:8080"]},
        {"domains": ["user@docs.site.com"]},
        {"domains": ["10.0.0.10"]},
        {"domains": ["printer.local"]},
        {"domains": ["github.io"]},  # весь общий хостинг, а не один проект
        {"domains": ["пример.рф"]},
        {"unknown": "поле"},
    ],
)
def test_invalid_list_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        create(**overrides)


def test_one_project_on_a_shared_hosting_is_fine() -> None:
    assert create(domains=["fastapi.github.io"]).domains == ["fastapi.github.io"]


def test_the_number_of_sites_is_limited_after_repeats_are_removed() -> None:
    limit = MAX_DOMAINS_PER_USER_PROFILE
    ok = [f"site{i}.com" for i in range(limit)]

    assert len(create(domains=ok).domains) == limit
    assert len(create(domains=ok + ok[:5]).domains) == limit  # повторы не в счёт
    with pytest.raises(ValidationError):
        create(domains=[*ok, "one-more.com"])


def test_absurdly_long_input_is_cut_off_before_it_is_parsed() -> None:
    with pytest.raises(ValidationError):
        create(domains=["a.com"] * (MAX_DOMAINS_PER_USER_PROFILE * 4 + 1))


def test_update_needs_at_least_one_field() -> None:
    with pytest.raises(ValidationError):
        UserWebProfileUpdate()
    with pytest.raises(ValidationError):
        UserWebProfileUpdate(title=None, description=None, domains=None)


def test_update_with_only_description_is_allowed_even_when_empty() -> None:
    # пустое описание — это «очистить», а не «ничего не менять»
    assert UserWebProfileUpdate(description="").description == ""


def test_update_normalizes_domains_like_create() -> None:
    update = UserWebProfileUpdate(domains=["https://A.com/x", "a.com", "b.com"])

    assert update.domains == ["a.com", "b.com"]


@pytest.mark.parametrize("domains", [[], ["not a domain"], ["a.local"]])
def test_update_rejects_bad_domains(domains: list[str]) -> None:
    with pytest.raises(ValidationError):
        UserWebProfileUpdate(domains=domains)


# --- поле запроса к чату


def test_personal_list_needs_web_search() -> None:
    from uuid import uuid4

    with pytest.raises(ValidationError, match="web_search"):
        MessageCreate(question="как?", my_web_profile_id=uuid4())


def test_personal_list_cannot_be_combined_with_a_profile_from_the_file() -> None:
    from uuid import uuid4

    with pytest.raises(ValidationError, match="вместе нельзя"):
        MessageCreate(
            question="как?",
            web_search=True,
            web_profile="backend",
            my_web_profile_id=uuid4(),
        )


def test_personal_list_with_web_search_is_accepted() -> None:
    from uuid import uuid4

    list_id = uuid4()

    payload = MessageCreate(question="как?", web_search=True, my_web_profile_id=list_id)

    assert payload.my_web_profile_id == list_id


def test_personal_list_cannot_be_combined_with_own_documents() -> None:
    from uuid import uuid4

    with pytest.raises(ValidationError):
        MessageCreate(
            question="как?",
            web_search=True,
            document_ids=[uuid4()],
            my_web_profile_id=uuid4(),
        )


def test_a_personal_list_holds_fifty_sites_and_not_fifty_one() -> None:
    # число закреплено явно: это продуктовое ограничение, а не деталь реализации
    assert len(create(domains=[f"s{i}.com" for i in range(50)]).domains) == 50
    with pytest.raises(ValidationError):
        create(domains=[f"s{i}.com" for i in range(51)])
    with pytest.raises(ValidationError):
        UserWebProfileUpdate(domains=[f"s{i}.com" for i in range(51)])


def test_raw_input_is_cut_off_in_update_too() -> None:
    with pytest.raises(ValidationError):
        UserWebProfileUpdate(domains=["a.com"] * (MAX_DOMAINS_PER_USER_PROFILE * 4 + 1))
