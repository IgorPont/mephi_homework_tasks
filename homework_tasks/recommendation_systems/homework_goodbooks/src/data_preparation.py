"""
Модуль подготовки данных для книжной рекомендательной системы.

Содержит функции очистки и преобразования исходных данных
перед обучением рекомендательных моделей.
"""

import numpy as np
import pandas as pd

from homework_tasks.recommendation_systems.homework_goodbooks.src.config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_RATINGS_PATH,
    TEST_SIZE,
    TRAIN_RATINGS_PATH,
    MAX_TAGS_PER_BOOK,
)

# Служебные теги, не описывающие содержание произведений
SERVICE_TAGS = {
    "to-read",
    "currently-reading",
    "read",
    "favorites",
    "favourites",
    "owned",
    "owned-books",
    "books-i-own",
    "library",
    "default",
    "to-buy",
    "book-club",
    "series",
    "kindle",
    "ebook",
    "audiobook",
    "audiobooks",
    "re-read",
    "reread",
    "read-in-2012",
    "my-books",
    "my-ebooks",
    "all-time-favorites",
    "favorite",
    "favorite-books",
}


def prepare_book_titles(books: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает названия книг для контентной модели.

    При отсутствии original_title используется title.

    Аргументы:
        - books: Метаданные книг

    Возвращает:
        - DataFrame с подготовленными названиями книг
    """

    books_content = books[
        ["id", "book_id", "title", "original_title", "authors"]
    ].copy()

    books_content["book_title"] = books_content["original_title"].fillna(
        books_content["title"]
    )

    return books_content


def prepare_book_content(
    books: pd.DataFrame,
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
    max_tags_per_book: int = MAX_TAGS_PER_BOOK,
) -> pd.DataFrame:
    """
    Формирует текстовые профили книг из названий и тегов.

    Для каждой книги выбираются наиболее часто используемые
    содержательные теги.

    Аргументы:
        - books: Метаданные книг
        - book_tags: Связи книг с тегами
        - tags: Справочник тегов
        - max_tags_per_book: Максимальное количество тегов на книгу

    Возвращает:
        - DataFrame с текстовым представлением каждой книги
    """

    if max_tags_per_book < 1:
        raise ValueError("max_tags_per_book должен быть положительным")

    books_content = prepare_book_titles(books)

    book_tags_with_names = book_tags.merge(
        tags,
        on="tag_id",
        how="left",
        validate="many_to_one",
    )

    # Удаляем служебные теги и записи без названия
    content_tags = book_tags_with_names[
        book_tags_with_names["tag_name"].notna()
        & ~book_tags_with_names["tag_name"].isin(SERVICE_TAGS)
        ].copy()

    # Объединяем повторяющиеся связи книга–тег, суммируя их частоты
    content_tags = (
        content_tags.groupby(
            ["goodreads_book_id", "tag_id", "tag_name"],
            as_index=False,
        )
        .agg(count=("count", "sum"))
    )

    # Отбираем наиболее часто используемые теги для каждой книги
    content_tags = content_tags.sort_values(
        ["goodreads_book_id", "count", "tag_id"],
        ascending=[True, False, True],
    )

    top_content_tags = content_tags.groupby(
        "goodreads_book_id",
        group_keys=False,
    ).head(max_tags_per_book)

    # Собираем названия тегов в одну строку для каждой книги
    book_tag_text = (
        top_content_tags.groupby("goodreads_book_id")["tag_name"]
        .agg(" ".join)
        .reset_index(name="tags_text")
    )

    # books.book_id соответствует book_tags.goodreads_book_id
    books_content = books_content.merge(
        book_tag_text,
        left_on="book_id",
        right_on="goodreads_book_id",
        how="left",
        validate="one_to_one",
    )

    books_content["tags_text"] = books_content["tags_text"].fillna("")

    # Формируем единый текстовый профиль: название + выбранные теги
    books_content["content_text"] = (
        books_content["book_title"].fillna("")
        + " "
        + books_content["tags_text"]
    ).str.strip()

    return books_content.drop(columns="goodreads_book_id")


def prepare_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает оценки пользователей для построения моделей.

    Повторяющиеся оценки одной книги пользователем объединяются
    с помощью среднего арифметического.

    Аргументы:
        - ratings: Исходные оценки пользователей

    Возвращает:
        - DataFrame с уникальными парами пользователь–книга
    """

    # Повторяющиеся пары объединяем,
    # поскольку время выставления оценок неизвестно
    prepared_ratings = (
        ratings.groupby(["user_id", "book_id"], as_index=False)
        .agg(rating=("rating", "mean"))
    )

    return prepared_ratings


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет оценки пользователей на train и test.

    Разбиение выполняется отдельно для каждого пользователя.
    В тренировочной выборке сохраняется хотя бы одна оценка
    каждого пользователя.

    Аргументы:
        - ratings: Подготовленные оценки пользователей
        - test_size: Доля оценок для тестовой выборки
        - random_state: Фиксированное значение для воспроизводимости

    Возвращает:
        - train_ratings: Обучающая выборка
        - test_ratings: Тестовая выборка
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться между 0 и 1")

    if ratings.duplicated(subset=["user_id", "book_id"]).any():
        raise ValueError("Перед разделением необходимо устранить дубликаты оценок")

    # Перемешиваем взаимодействия воспроизводимым образом
    rng = np.random.default_rng(random_state)
    shuffled_ratings = ratings.copy()
    shuffled_ratings["_random"] = rng.random(len(shuffled_ratings))

    shuffled_ratings = shuffled_ratings.sort_values("_random")

    # Определяем порядковый номер и количество оценок каждого пользователя
    user_groups = shuffled_ratings.groupby("user_id")

    interaction_index = user_groups.cumcount()
    interaction_count = user_groups["book_id"].transform("size")

    # Для пользователей с двумя и более оценками оставляем часть в test,
    # но обязательно сохраняем хотя бы одну оценку в train
    test_count = np.minimum(
        np.maximum(np.floor(interaction_count * test_size).astype(int), 1),
        interaction_count - 1,
    )

    test_mask = interaction_index < test_count

    train_ratings = shuffled_ratings.loc[~test_mask].drop(
        columns="_random"
    )
    test_ratings = shuffled_ratings.loc[test_mask].drop(
        columns="_random"
    )

    return (
        train_ratings.reset_index(drop=True),
        test_ratings.reset_index(drop=True),
    )


def save_ratings_split(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
) -> None:
    """
    Сохраняет обучающую и тестовую выборки в CSV-файлы.

    Аргументы:
        - train_ratings: Обучающая выборка
        - test_ratings: Тестовая выборка
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_ratings.to_csv(TRAIN_RATINGS_PATH, index=False)
    test_ratings.to_csv(TEST_RATINGS_PATH, index=False)

    print(f"Обучающая выборка сохранена: {TRAIN_RATINGS_PATH}")
    print(f"Тестовая выборка сохранена: {TEST_RATINGS_PATH}")
