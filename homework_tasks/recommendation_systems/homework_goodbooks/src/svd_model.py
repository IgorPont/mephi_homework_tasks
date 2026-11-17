"""
Обучение модели матричной факторизации для книжных рекомендаций.
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from homework_tasks.recommendation_systems.homework_goodbooks.src.config import (
    RANDOM_STATE,
)


def train_svd_model(
    train_ratings: pd.DataFrame,
    n_factors: int = 50,
    n_epochs: int = 20,
    random_state: int = RANDOM_STATE,
) -> SVD:
    """
    Обучает модель SVD на известных пользовательских оценках.

    Args:
        train_ratings: Обучающая выборка с user_id, book_id и rating
        n_factors: Количество скрытых факторов
        n_epochs: Количество эпох обучения
        random_state: Начальное состояние генератора случайных чисел

    Returns:
        Обученная модель SVD.
    """

    if n_factors < 1 or n_epochs < 1:
        raise ValueError("n_factors и n_epochs должны быть положительными")

    reader = Reader(rating_scale=(1, 5))

    dataset = Dataset.load_from_df(
        train_ratings[["user_id", "book_id", "rating"]],
        reader=reader,
    )

    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        random_state=random_state,
    )

    model.fit(trainset)

    return model


def predict_svd_rating(
    model: SVD,
    user_id: int,
    book_id: int,
) -> float:
    """
    Предсказывает пользовательскую оценку книги с помощью SVD.

    Args:
        model: Обученная модель SVD
        user_id: Идентификатор пользователя
        book_id: Идентификатор книги

    Returns:
        Предсказанная оценка либо np.nan для неизвестного
        пользователя или книги
    """
    try:
        model.trainset.to_inner_uid(user_id)
        model.trainset.to_inner_iid(book_id)
    except ValueError:
        return float("nan")

    prediction = model.predict(
        uid=user_id,
        iid=book_id,
    )

    return float(prediction.est)


def get_svd_recommendations(
    model: SVD,
    user_id: int,
    books: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Формирует Top-N персональных рекомендаций на основе SVD.

    Args:
        model: Обученная модель SVD
        user_id: Идентификатор пользователя
        books: Метаданные книг
        top_n: Количество рекомендаций

    Returns:
        DataFrame с рекомендуемыми книгами и предсказанными оценками
    """

    if top_n < 1:
        raise ValueError("top_n должен быть положительным")

    try:
        user_index = model.trainset.to_inner_uid(user_id)
    except ValueError as exc:
        raise ValueError(f"Пользователь с user_id={user_id} не найден") from exc

    trainset = model.trainset

    # Рассчитываем прогнозы для всех книг, известных модели
    predicted_ratings = (
        trainset.global_mean
        + model.bu[user_index]
        + model.bi
        + model.qi @ model.pu[user_index]
    )

    # Surprise ограничивает предсказания диапазоном допустимых оценок
    predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)

    # Исключаем книги, оцененные пользователем в обучающей выборке
    rated_book_indices = [
        book_index
        for book_index, _ in trainset.ur[user_index]
    ]

    predicted_ratings[rated_book_indices] = np.nan

    candidate_indices = np.flatnonzero(np.isfinite(predicted_ratings))

    sorted_indices = candidate_indices[
        np.argsort(
            -predicted_ratings[candidate_indices],
            kind="stable",
        )
    ][:top_n]

    recommendations = pd.DataFrame(
        {
            "book_id": [
                trainset.to_raw_iid(int(book_index))
                for book_index in sorted_indices
            ],
            "predicted_rating": predicted_ratings[sorted_indices],
        }
    )

    recommendations = recommendations.merge(
        books[["id", "title", "authors"]],
        left_on="book_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )

    return recommendations[
        ["book_id", "title", "authors", "predicted_rating"]
    ].reset_index(drop=True)
