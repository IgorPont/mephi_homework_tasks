"""
Запуск экспериментов для задач регрессии и классификации.

Модуль содержит функции, которые:
    - делят данные на train/test
    - обучают несколько моделей
    - считают качество на кросс-валидации
    - считают качество на тестовой выборке
    - возвращают итоговую таблицу метрик
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.config import (
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
)
from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
)


def run_regression_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, object],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Запускает эксперимент для задачи регрессии.

    Аргументы:
        - X: Матрица признаков.
        - y: Целевая переменная.
        - models: Словарь моделей для сравнения.
        - test_size: Размер тестовой выборки.
        - random_state: Фиксированное зерно случайности.
        - cv_folds: Количество фолдов для кросс-валидации.

    Возвращает:
        - DataFrame с метриками моделей.
        - Словарь обученных моделей.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    results = []
    fitted_models = {}

    for model_name, model in models.items():
        start_time = time.time()

        estimator = clone(model)

        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        metrics = evaluate_regression_model(y_test, y_pred)

        elapsed_time = time.time() - start_time

        result = {
            "model": model_name,
            "cv_rmse_mean": float(-cv_scores.mean()),
            "cv_rmse_std": float(cv_scores.std()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "fit_time_sec": float(elapsed_time),
            **metrics,
        }

        results.append(result)
        fitted_models[model_name] = estimator

    results_df = (
        pd.DataFrame(results)
        .sort_values(by=["rmse", "mae"], ascending=True)
        .reset_index(drop=True)
    )

    return results_df, fitted_models


def run_classification_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, object],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Запускает эксперимент для задачи бинарной классификации.

    Аргументы:
        - X: Матрица признаков
        - y: Целевая переменная
        - models: Словарь моделей для сравнения
        - test_size: Размер тестовой выборки
        - random_state: Фиксированное зерно случайности
        - cv_folds: Количество фолдов для кросс-валидации

    Возвращает:
        - DataFrame с метриками моделей
        - Словарь обученных моделей
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    results = []
    fitted_models = {}

    for model_name, model in models.items():
        start_time = time.time()

        estimator = clone(model)

        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_proba = _get_positive_class_probability(estimator, X_test)

        metrics = evaluate_classification_model(y_test, y_pred, y_proba)

        elapsed_time = time.time() - start_time

        result = {
            "model": model_name,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "fit_time_sec": float(elapsed_time),
            **metrics,
        }

        results.append(result)
        fitted_models[model_name] = estimator

    results_df = (
        pd.DataFrame(results)
        .sort_values(by=["f1", "balanced_accuracy", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    return results_df, fitted_models


def _get_positive_class_probability(model: object, X_test: pd.DataFrame) -> np.ndarray | None:
    """
    Возвращает вероятность положительного класса для бинарной классификации.

    Аргументы:
        - model: Обученная модель
        - X_test: Тестовая матрица признаков

    Возвращает:
        - Массив вероятностей класса 1, если модель поддерживает predict_proba
        - None, если модель не поддерживает predict_proba
    """

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X_test)

    if probabilities.shape[1] < 2:
        return None

    return probabilities[:, 1]
