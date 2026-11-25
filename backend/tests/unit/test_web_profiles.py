import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import settings
from app.ingestion.web_pages import is_allowed_url
from app.rag.web_profiles import (
    MAX_DOMAINS_PER_PROFILE,
    PROFILES_PATH,
    WebProfile,
    WebProfiles,
    load_web_profiles,
)


def make_profile(name: str = "backend", **overrides: object) -> WebProfile:
    data: dict[str, object] = {
        "name": name,
        "title": "Заголовок",
        "domains": ["docs.python.org"],
        **overrides,
    }
    return WebProfile.model_validate(data)


# --- WebProfile: каждый домен в списке — это разрешение ходить на сайт


def test_domains_are_trimmed_and_lowercased() -> None:
    profile = make_profile(domains=["  Docs.Python.ORG ", "fastapi.tiangolo.com"])

    assert profile.domains == ("docs.python.org", "fastapi.tiangolo.com")


@pytest.mark.parametrize(
    "domain",
    [
        "https://docs.python.org",  # схема не нужна: её проверяет is_allowed_url
        "docs.python.org/3/",  # путь
        "docs.python.org:8443",  # порт
        "python",  # не домен
        "127.0.0.1",  # адрес вместо имени
        "10.0.0.10",
        "docs python.org",
        "-bad.example.com",
        "",
    ],
)
def test_not_a_bare_domain_is_rejected(domain: str) -> None:
    with pytest.raises(ValidationError, match="это не домен"):
        make_profile(domains=[domain])


@pytest.mark.parametrize("domain", ["github.io", "ReadTheDocs.io", "pages.dev"])
def test_whole_shared_hosting_is_rejected(domain: str) -> None:
    with pytest.raises(ValidationError, match="общий хостинг"):
        make_profile(domains=[domain])


def test_single_project_on_shared_hosting_is_fine() -> None:
    assert make_profile(domains=["fastapi.github.io"]).domains == ("fastapi.github.io",)


def test_empty_duplicate_and_too_long_lists_are_rejected() -> None:
    with pytest.raises(ValidationError, match="пуст"):
        make_profile(domains=[])
    with pytest.raises(ValidationError, match="повторяются"):
        make_profile(domains=["docs.python.org", "DOCS.python.org"])
    too_many = [f"site{i}.example.com" for i in range(MAX_DOMAINS_PER_PROFILE + 1)]
    with pytest.raises(ValidationError, match="больше"):
        make_profile(domains=too_many)
    # ровно по лимиту ещё можно
    at_limit = [f"site{i}.example.com" for i in range(MAX_DOMAINS_PER_PROFILE)]
    assert len(make_profile(domains=at_limit).domains) == MAX_DOMAINS_PER_PROFILE


@pytest.mark.parametrize("name", ["", "Backend", "1abc", "a b", "a" * 33, "../x"])
def test_profile_name_must_be_a_simple_slug(name: str) -> None:
    with pytest.raises(ValidationError):
        make_profile(name=name)


# --- WebProfiles: реестр


def test_profile_is_found_by_name_and_none_means_default() -> None:
    backend, analyst = make_profile("backend"), make_profile("analyst")
    profiles = WebProfiles([backend, analyst], default="analyst")

    assert profiles.get("backend") is backend
    assert profiles.get(None) is analyst
    assert profiles.get("nope") is None
    assert profiles.all() == [backend, analyst]


def test_registry_rejects_duplicate_names_and_missing_default() -> None:
    with pytest.raises(ValueError, match="повторяются"):
        WebProfiles([make_profile("a"), make_profile("a")], default="a")
    with pytest.raises(ValueError, match="по умолчанию"):
        WebProfiles([make_profile("a")], default="b")


# --- загрузка из файла


def write_profiles(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_profiles_are_loaded_from_a_file(tmp_path: Path) -> None:
    path = write_profiles(
        tmp_path,
        [{"name": "a", "title": "A", "domains": ["docs.python.org"]}],
    )

    profiles = load_web_profiles("a", path)

    assert profiles.default.domains == ("docs.python.org",)


def test_broken_file_stops_loading_instead_of_being_ignored(tmp_path: Path) -> None:
    path = write_profiles(
        tmp_path, [{"name": "a", "title": "A", "domains": ["https://evil.com/"]}]
    )

    with pytest.raises(ValidationError):
        load_web_profiles("a", path)


def test_shipped_profiles_file_is_valid_and_has_the_default_profile() -> None:
    profiles = load_web_profiles(settings.web_default_profile)

    assert {profile.name for profile in profiles.all()} >= {"backend", "analyst"}
    assert PROFILES_PATH.is_file()


def test_shipped_profiles_allow_what_they_promise() -> None:
    profiles = load_web_profiles(settings.web_default_profile)
    backend, analyst = profiles.get("backend"), profiles.get("analyst")
    assert backend is not None and analyst is not None

    assert is_allowed_url("https://fastapi.tiangolo.com/", backend.domains)
    assert not is_allowed_url("https://fastapi.tiangolo.com/", analyst.domains)
    assert is_allowed_url("https://www.omg.org/spec/BPMN/", analyst.domains)
