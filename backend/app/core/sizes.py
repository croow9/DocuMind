from math import ceil

KIBIBYTE = 1024
MEBIBYTE = 1024 * KIBIBYTE


def human_size(num_bytes: int) -> str:
    """Размер для сообщения человеку: «20 МБ», «512 КБ», «100 Б» (округление вверх)."""
    if num_bytes >= MEBIBYTE:
        return f"{ceil(num_bytes / MEBIBYTE)} МБ"
    if num_bytes >= KIBIBYTE:
        return f"{ceil(num_bytes / KIBIBYTE)} КБ"
    return f"{num_bytes} Б"
