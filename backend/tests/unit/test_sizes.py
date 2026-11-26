import pytest

from app.core.sizes import KIBIBYTE, MEBIBYTE, human_size


def test_units_are_binary() -> None:
    assert KIBIBYTE == 1024
    assert MEBIBYTE == 1024 * 1024


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 Б"),
        (100, "100 Б"),
        (1023, "1023 Б"),
        (1024, "1 КБ"),
        (1025, "2 КБ"),  # округление вверх: «до 1 КБ» было бы неправдой
        (512 * 1024, "512 КБ"),
        (1024 * 1024, "1 МБ"),
        (20 * 1024 * 1024, "20 МБ"),
        (20 * 1024 * 1024 + 1, "21 МБ"),
    ],
)
def test_human_size(size: int, expected: str) -> None:
    assert human_size(size) == expected
