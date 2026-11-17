"""
Метрики качества рекомендательных алгоритмов.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


def precision_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 5,
) -> float:
    """
    Вычисляет долю релевантных рекомендаций среди первых K позиций.
    """

    if k < 1:
        raise ValueError("k должен быть положительным")

    top_k_ids = recommended_ids[:k]
    relevant_count = sum(
        book_id in relevant_ids
        for book_id in top_k_ids
    )

    return relevant_count / k


def recall_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 5,
) -> float:
    """
    Вычисляет долю найденных релевантных книг среди всех релевантных.
    """

    if k < 1:
        raise ValueError("k должен быть положительным")

    if not relevant_ids:
        return 0.0

    top_k_ids = recommended_ids[:k]
    relevant_count = sum(
        book_id in relevant_ids
        for book_id in top_k_ids
    )

    return relevant_count / len(relevant_ids)


def ndcg_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 5,
) -> float:
    """
    Вычисляет нормализованный DCG для бинарной релевантности.
    """

    if k < 1:
        raise ValueError("k должен быть положительным")

    if not relevant_ids:
        return 0.0

    top_k_ids = recommended_ids[:k]

    relevance = np.array(
        [book_id in relevant_ids for book_id in top_k_ids],
        dtype=float,
    )

    positions = np.arange(2, len(relevance) + 2)
    dcg = np.sum(relevance / np.log2(positions))

    ideal_relevant_count = min(k, len(relevant_ids))
    ideal_positions = np.arange(2, ideal_relevant_count + 2)

    idcg = np.sum(1.0 / np.log2(ideal_positions))

    return float(dcg / idcg)


def evaluate_recommender(
    user_ids: np.ndarray,
    relevant_test_ratings: pd.DataFrame,
    recommend: Callable[[int], pd.DataFrame],
    k: int = 5,
) -> pd.DataFrame:
    """
    Оценивает рекомендательный алгоритм на заданном наборе пользователей.

    Args:
        user_ids: Идентификаторы оцениваемых пользователей
        relevant_test_ratings: Релевантные взаимодействия из test
        recommend: Функция, возвращающая рекомендации для пользователя
        k: Количество позиций рекомендательного списка

    Returns:
        DataFrame с метриками и количеством рекомендаций для каждого пользователя.
    """

    if k < 1:
        raise ValueError("k должен быть положительным")

    if len(user_ids) != len(set(user_ids)):
        raise ValueError("Идентификаторы пользователей должны быть уникальными")

    relevant_books_by_user = (
        relevant_test_ratings.groupby("user_id")["book_id"]
        .agg(set)
        .to_dict()
    )

    results = []

    for user_id in user_ids:
        user_id = int(user_id)

        recommendations = recommend(user_id)
        recommended_ids = recommendations["book_id"].tolist()

        if len(recommended_ids) > k:
            raise ValueError(
                f"Модель вернула больше {k} рекомендаций для user_id={user_id}"
            )

        if len(recommended_ids) != len(set(recommended_ids)):
            raise ValueError(
                f"Модель вернула повторяющиеся книги для user_id={user_id}"
            )

        if any(pd.isna(book_id) for book_id in recommended_ids):
            raise ValueError(
                f"Модель вернула пустой book_id для user_id={user_id}"
            )

        relevant_ids = relevant_books_by_user.get(user_id, set())

        results.append(
            {
                "user_id": user_id,
                "recommendations_count": len(recommended_ids),
                "precision": precision_at_k(recommended_ids, relevant_ids, k),
                "recall": recall_at_k(recommended_ids, relevant_ids, k),
                "ndcg": ndcg_at_k(recommended_ids, relevant_ids, k),
            }
        )

    return pd.DataFrame(results)
