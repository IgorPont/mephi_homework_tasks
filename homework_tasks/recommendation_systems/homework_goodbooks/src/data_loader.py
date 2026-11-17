"""
Модуль загрузки исходных данных Goodbooks-10k.

Все данные загружаются из CSV-файлов, расположенных
в директории data/raw/.
"""

from pathlib import Path

import pandas as pd

from homework_tasks.recommendation_systems.homework_goodbooks.src.config import (
    BOOK_TAGS_PATH,
    BOOKS_PATH,
    RATINGS_PATH,
    TAGS_PATH,
    TO_READ_PATH,
)


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Загружает DataFrame из CSV-файла.

    Аргументы:
        - file_path: Путь к исходному CSV-файлу

    Возвращает:
        - Загруженный DataFrame

    Исключения:
        - FileNotFoundError: Если файл не найден
    """

    if not file_path.is_file():
        raise FileNotFoundError(f"Не найден файл с данными: {file_path}")

    return pd.read_csv(file_path)


def load_ratings() -> pd.DataFrame:
    """
    Загружает оценки пользователей.
    """

    return load_dataframe(RATINGS_PATH)


def load_books() -> pd.DataFrame:
    """
    Загружает метаданные книг.
    """

    return load_dataframe(BOOKS_PATH)


def load_tags() -> pd.DataFrame:
    """
    Загружает справочник тегов.
    """

    return load_dataframe(TAGS_PATH)


def load_book_tags() -> pd.DataFrame:
    """
    Загружает связи книг с тегами.
    """

    return load_dataframe(BOOK_TAGS_PATH)


def load_to_read() -> pd.DataFrame:
    """
    Загружает список книг, отмеченных к прочтению.
    """

    return load_dataframe(TO_READ_PATH)


def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Загружает все исходные таблицы проекта.

    Возвращает:
        - Словарь с исходными DataFrame
    """

    return {
        "ratings": load_ratings(),
        "books": load_books(),
        "tags": load_tags(),
        "book_tags": load_book_tags(),
        "to_read": load_to_read(),
    }
