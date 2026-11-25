import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

# Лимит поискового API (Tavily принимает до 300 доменов), а заодно защита от
# случайно вставленного огромного списка
MAX_DOMAINS_PER_PROFILE = 300

PROFILES_PATH = Path(__file__).with_name("web_profiles.json")

# Имя хоста без схемы, пути и порта; последняя часть не из цифр, поэтому адреса
# вроде 10.0.0.10 не проходят. Допускаются национальные зоны вида xn--p1ai
_DOMAIN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:[a-z]{2,63}|xn--[a-z0-9-]{1,59})$"
)

# Общие хостинги: на github.io или readthedocs.io живут чужие сайты, и разрешить
# такой домен целиком значит разрешить любой из них. Можно добавить один проект
# (fastapi.github.io), но не весь хостинг
SHARED_HOSTINGS = frozenset(
    {
        "github.io", "readthedocs.io", "readthedocs.org", "gitlab.io", "netlify.app",
        "vercel.app", "pages.dev", "web.app", "firebaseapp.com", "herokuapp.com",
        "blogspot.com", "wordpress.com", "medium.com", "notion.site",
    }
)  # fmt: skip


class WebProfile(BaseModel):
    """Набор официальных сайтов для одного направления (бэкенд, системный анализ…)."""

    model_config = ConfigDict(frozen=True)

    # имя идёт в запросе клиента и в ключи кэша, поэтому только простые символы
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    title: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    domains: tuple[str, ...]

    @field_validator("domains")
    @classmethod
    def _check_domains(cls, domains: tuple[str, ...]) -> tuple[str, ...]:
        if not domains:
            raise ValueError("список сайтов пуст")
        if len(domains) > MAX_DOMAINS_PER_PROFILE:
            raise ValueError(f"сайтов больше {MAX_DOMAINS_PER_PROFILE}")
        cleaned = tuple(domain.strip().lower() for domain in domains)
        for domain in cleaned:
            if not _DOMAIN.match(domain):
                raise ValueError(f"это не домен: {domain!r}")
            if domain in SHARED_HOSTINGS:
                raise ValueError(f"общий хостинг {domain!r} разрешать нельзя")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("домены повторяются")
        return cleaned


class WebProfiles:
    """Все профили и профиль по умолчанию; проверяется один раз, при загрузке."""

    def __init__(self, profiles: Sequence[WebProfile], default: str) -> None:
        by_name = {profile.name: profile for profile in profiles}
        if len(by_name) != len(profiles):
            raise ValueError("имена профилей повторяются")
        if default not in by_name:
            raise ValueError(f"профиль по умолчанию {default!r} не найден")
        self._by_name = by_name
        self.default = by_name[default]

    def get(self, name: str | None) -> WebProfile | None:
        """Профиль по имени; без имени (None) — профиль по умолчанию."""
        if name is None:
            return self.default
        return self._by_name.get(name)

    def all(self) -> list[WebProfile]:
        return list(self._by_name.values())


_PROFILE_LIST = TypeAdapter(list[WebProfile])


def load_web_profiles(default: str, path: Path = PROFILES_PATH) -> WebProfiles:
    """Читает профили из файла; ошибка в нём останавливает запуск приложения.

    Лучше упасть сразу с понятной причиной, чем выяснить про опечатку в домене
    на первом же вопросе пользователя.
    """
    profiles = _PROFILE_LIST.validate_json(path.read_text(encoding="utf-8"))
    return WebProfiles(profiles, default)
