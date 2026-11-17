"""
Модуль алгоритмов книжных рекомендаций.

Содержит функции построения рекомендательных моделей
и получения списков рекомендуемых книг.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_popularity_ranking(
    train_ratings: pd.DataFrame,
    min_ratings_count: int = 20,
) -> pd.DataFrame:
    """
    Формирует рейтинг книг по средним пользовательским оценкам.

    Аргументы:
        - train_ratings: Обучающая выборка с оценками пользователей
        - min_ratings_count: Минимальное количество оценок книги

    Возвращает:
        - DataFrame с рейтингом книг, отсортированным по средней оценке
    """

    book_rating_stats = (
        train_ratings.groupby("book_id")
        .agg(
            ratings_count=("rating", "count"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    # Исключаем книги с недостаточным количеством оценок
    popular_books = book_rating_stats[
        book_rating_stats["ratings_count"] >= min_ratings_count
        ].copy()

    # При равенстве средней оценки используем количество оценок,
    # а затем идентификатор книги для воспроизводимого порядка
    popular_books = popular_books.sort_values(
        ["average_rating", "ratings_count", "book_id"],
        ascending=[False, False, True],
    )

    return popular_books.reset_index(drop=True)


def get_popular_recommendations(
    popularity_ranking: pd.DataFrame,
    books: pd.DataFrame,
    top_n: int = 10,
    user_id: int | None = None,
    train_ratings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Возвращает рекомендации на основе общего рейтинга книг.

    Для известного пользователя исключает книги,
    которые он уже оценил в обучающей выборке.

    Аргументы:
        - popularity_ranking: Рейтинг популярных книг
        - books: Метаданные книг
        - top_n: Количество рекомендаций
        - user_id: Идентификатор пользователя, если указан
        - train_ratings: Обучающая выборка для исключения оцененных книг

    Возвращает:
        - DataFrame с рекомендуемыми книгами
    """

    recommendations = popularity_ranking

    if user_id is not None:
        if train_ratings is None:
            raise ValueError(
                "Для рекомендаций пользователю необходимо передать train_ratings"
            )

        # Исключаем книги, уже оцененные пользователем в train
        rated_book_ids = train_ratings.loc[
            train_ratings["user_id"] == user_id,
            "book_id",
        ]

        recommendations = recommendations[
            ~recommendations["book_id"].isin(rated_book_ids)
        ]

    recommendations = recommendations.head(top_n).merge(
        books[["id", "title", "authors"]],
        left_on="book_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )

    return recommendations[
        ["book_id", "title", "authors", "ratings_count", "average_rating"]
    ]


def get_similar_books(
    book_id: int,
    books_content: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Возвращает наиболее похожие книги на основе TF-IDF.

    Сходство рассчитывается с помощью косинусной близости
    между текстовыми профилями книг.

    Аргументы:
        - book_id: Внутренний идентификатор книги (books.id)
        - books_content: Метаданные и текстовые профили книг
        - tfidf_matrix: Матрица TF-IDF
        - top_n: Количество рекомендуемых книг

    Возвращает:
        - DataFrame с похожими книгами и значениями сходства
    """

    if top_n < 1:
        raise ValueError("top_n должен быть положительным")

    if len(books_content) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Количество книг не соответствует количеству строк TF-IDF матрицы"
        )

    # Определяем позицию книги в TF-IDF матрице по внутреннему ID
    book_positions = np.flatnonzero(
        books_content["id"].to_numpy() == book_id
    )

    if len(book_positions) != 1:
        raise ValueError(f"Книга с book_id={book_id} не найдена или ID не уникален")

    book_index = int(book_positions[0])

    # Рассчитываем сходство выбранной книги со всем каталогом
    similarity_scores = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix,
    ).ravel()

    # Исключаем исходную книгу и сортируем остальные по убыванию сходства
    similar_indices = np.argsort(-similarity_scores, kind="stable")
    similar_indices = similar_indices[
        similar_indices != book_index
        ][:top_n]

    recommendations = books_content.iloc[
        similar_indices
    ][["id", "title", "authors"]].copy()

    recommendations["similarity"] = similarity_scores[similar_indices]

    return recommendations.reset_index(drop=True)


def get_content_recommendations(
    user_id: int,
    train_ratings: pd.DataFrame,
    books_content: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    top_n: int = 10,
    relevance_threshold: float = 4,
) -> pd.DataFrame:
    """
    Формирует персональные рекомендации по текстовому профилю пользователя.
    """

    if top_n < 1:
        raise ValueError("top_n должен быть положительным")

    if len(books_content) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Количество книг не соответствует количеству строк TF-IDF матрицы"
        )

    if not books_content["id"].is_unique:
        raise ValueError("Идентификаторы книг должны быть уникальными")

    user_ratings = train_ratings.loc[
        train_ratings["user_id"] == user_id
        ]

    if user_ratings.empty:
        raise ValueError(f"Пользователь с user_id={user_id} не найден")

    columns = ["book_id", "title", "authors", "similarity"]

    positive_book_ids = user_ratings.loc[
        user_ratings["rating"] >= relevance_threshold,
        "book_id",
    ]

    if positive_book_ids.empty:
        return pd.DataFrame(columns=columns)

    # Сопоставляем ID книг с позициями строк TF-IDF матрицы
    book_to_position = pd.Series(
        np.arange(len(books_content)),
        index=books_content["id"],
    )

    positive_positions = book_to_position.reindex(
        positive_book_ids
    ).dropna().to_numpy(dtype=int)

    if len(positive_positions) == 0:
        return pd.DataFrame(columns=columns)

    # Усредняем текстовые векторы положительно оцененных книг
    user_profile = csr_matrix(
        tfidf_matrix[positive_positions].mean(axis=0)
    )

    similarities = cosine_similarity(
        tfidf_matrix,
        user_profile,
    ).ravel()

    # Исключаем все оцененные книги, включая оценки ниже порога
    rated_positions = book_to_position.reindex(
        user_ratings["book_id"]
    ).dropna().to_numpy(dtype=int)

    similarities[rated_positions] = np.nan

    candidate_indices = np.flatnonzero(
        np.isfinite(similarities) & (similarities > 0)
    )

    sorted_indices = candidate_indices[
        np.argsort(
            -similarities[candidate_indices],
            kind="stable",
        )
    ][:top_n]

    recommendations = books_content.iloc[
        sorted_indices
    ][["id", "title", "authors"]].copy()

    recommendations = recommendations.rename(
        columns={"id": "book_id"}
    )

    recommendations["similarity"] = similarities[sorted_indices]

    return recommendations.reset_index(drop=True)


def predict_item_rating(
    user_id: int,
    book_id: int,
    user_item_matrix: csr_matrix,
    item_user_matrix: csr_matrix,
    user_to_index: dict,
    book_to_index: dict,
    top_k: int = 10,
    min_positive_neighbors: int = 1,
) -> float:
    """
    Предсказывает оценку пользователя для книги с помощью Item-Based CF.

    Аргументы:
        - user_id: Идентификатор пользователя
        - book_id: Идентификатор целевой книги
        - user_item_matrix: Матрица пользователь–книга
        - item_user_matrix: Матрица книга–пользователь
        - user_to_index: Соответствие ID пользователей индексам
        - book_to_index: Соответствие ID книг индексам
        - top_k: Максимальное количество похожих книг
        - min_positive_neighbors: Минимальное количество соседей
          с положительным сходством

    Возвращает:
        - Предсказанную оценку или np.nan, если предсказание невозможно
    """

    if top_k < 1:
        raise ValueError("top_k должен быть положительным")

    if not 1 <= min_positive_neighbors <= top_k:
        raise ValueError(
            "min_positive_neighbors должен быть в диапазоне от 1 до top_k"
        )

    if user_id not in user_to_index or book_id not in book_to_index:
        return np.nan

    user_index = user_to_index[user_id]
    book_index = book_to_index[book_id]

    # Получаем только книги с известными оценками пользователя
    user_ratings = user_item_matrix.getrow(user_index)
    rated_indices = user_ratings.indices
    rated_values = user_ratings.data

    # Целевую книгу исключаем из истории при расчете предсказания
    candidate_mask = rated_indices != book_index
    rated_indices = rated_indices[candidate_mask]
    rated_values = rated_values[candidate_mask]

    if len(rated_indices) == 0:
        return np.nan

    # Сравниваем целевую книгу только с книгами из истории пользователя
    similarities = cosine_similarity(
        item_user_matrix[book_index],
        item_user_matrix[rated_indices],
    ).ravel()

    # Учитываем только книги с положительным сходством
    positive_mask = similarities > 0

    if np.count_nonzero(positive_mask) < min_positive_neighbors:
        return np.nan

    similarities = similarities[positive_mask]
    rated_values = rated_values[positive_mask]

    # Выбираем до top_k наиболее похожих книг
    neighbor_indices = np.argsort(-similarities, kind="stable")[:top_k]

    neighbor_similarities = similarities[neighbor_indices]
    neighbor_ratings = rated_values[neighbor_indices]

    predicted_rating = np.average(
        neighbor_ratings,
        weights=neighbor_similarities,
    )

    return float(predicted_rating)


def get_item_recommendations(
    user_id: int,
    user_item_matrix: csr_matrix,
    item_user_matrix: csr_matrix,
    user_to_index: dict,
    book_ids: np.ndarray,
    books: pd.DataFrame,
    top_n: int = 10,
    top_k: int = 10,
    min_positive_neighbors: int = 1,
) -> pd.DataFrame:
    """
    Формирует Top-N рекомендаций с помощью Item-Based CF.

    Аргументы:
        - user_id: Идентификатор пользователя
        - user_item_matrix: Матрица пользователь–книга
        - item_user_matrix: Матрица книга–пользователь
        - user_to_index: Соответствие ID пользователей индексам
        - book_ids: Идентификаторы книг в порядке столбцов матрицы
        - books: Метаданные книг
        - top_n: Количество рекомендаций
        - top_k: Максимальное количество похожих книг для прогноза
        - min_positive_neighbors: Минимальное количество соседей
          с положительным сходством

    Возвращает:
        - DataFrame с рекомендуемыми книгами и предсказанными оценками
    """

    if top_n < 1 or top_k < 1:
        raise ValueError("top_n и top_k должны быть положительными")

    if not 1 <= min_positive_neighbors <= top_k:
        raise ValueError(
            "min_positive_neighbors должен быть в диапазоне от 1 до top_k"
        )

    if user_id not in user_to_index:
        raise ValueError(f"Пользователь с user_id={user_id} не найден")

    if user_item_matrix.shape[1] != item_user_matrix.shape[0]:
        raise ValueError("Размерности матриц взаимодействий не совпадают")

    if len(book_ids) != item_user_matrix.shape[0]:
        raise ValueError("Количество идентификаторов книг не соответствует матрице")

    user_index = user_to_index[user_id]
    user_ratings = user_item_matrix.getrow(user_index)

    rated_indices = user_ratings.indices
    rated_values = user_ratings.data

    if len(rated_indices) == 0:
        return pd.DataFrame(
            columns=["book_id", "title", "authors", "predicted_rating"]
        )

    # Рассчитываем сходство всех книг с книгами из истории пользователя
    similarities = cosine_similarity(
        item_user_matrix,
        item_user_matrix[rated_indices],
    )

    # Выбираем до top_k наиболее похожих оцененных книг для каждого кандидата
    neighbor_indices = np.argsort(
        -similarities,
        axis=1,
        kind="stable",
    )[:, :top_k]

    neighbor_similarities = np.take_along_axis(
        similarities,
        neighbor_indices,
        axis=1,
    )

    neighbor_ratings = rated_values[neighbor_indices]

    # Учитываем только кандидатов с достаточным числом положительных соседей
    positive_neighbors_count = (neighbor_similarities > 0).sum(axis=1)

    valid_neighbors_mask = (
        positive_neighbors_count >= min_positive_neighbors
    )

    # Рассчитываем взвешенные прогнозы по положительным коэффициентам сходства
    weighted_sum = (
        neighbor_similarities * neighbor_ratings
    ).sum(axis=1)

    similarity_sum = neighbor_similarities.sum(axis=1)

    predicted_ratings = np.divide(
        weighted_sum,
        similarity_sum,
        out=np.full(len(book_ids), np.nan, dtype=float),
        where=(similarity_sum > 0) & valid_neighbors_mask,
    )

    # Исключаем книги, которые пользователь уже оценил
    predicted_ratings[rated_indices] = np.nan

    candidate_indices = np.flatnonzero(
        np.isfinite(predicted_ratings)
    )

    # Сортируем доступные книги по убыванию предсказанной оценки
    sorted_indices = candidate_indices[
        np.argsort(
            -predicted_ratings[candidate_indices],
            kind="stable",
        )
    ][:top_n]

    recommendations = pd.DataFrame(
        {
            "book_id": book_ids[sorted_indices],
            "predicted_rating": predicted_ratings[sorted_indices],
        }
    )

    # Добавляем названия и авторов книг
    recommendations = recommendations.merge(
        books[["id", "title", "authors"]],
        left_on="book_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )

    return recommendations[
        ["book_id", "title", "authors", "predicted_rating"]
    ]
