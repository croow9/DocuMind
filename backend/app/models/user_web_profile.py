from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ARRAY, DateTime, ForeignKey, String, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class UserWebProfileORM(Base):
    """Личный список сайтов пользователя для поиска по документации в интернете."""

    __tablename__ = "user_web_profiles"
    # одноимённые списки у одного пользователя только запутали бы выбор; индекс
    # этого ограничения заодно ускоряет «все списки владельца»
    __table_args__ = (
        UniqueConstraint("owner_id", "title", name="uq_user_web_profiles_owner_title"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(500), default="")
    # уже очищенные домены (см. clean_domains); порядок — как ввёл пользователь
    domains: Mapped[list[str]] = mapped_column(ARRAY(String(253)))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
