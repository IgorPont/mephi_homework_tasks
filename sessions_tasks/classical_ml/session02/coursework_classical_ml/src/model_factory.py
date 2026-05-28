"""
Фабрика моделей для задач регрессии и классификации.

Модуль содержит функции, которые создают наборы моделей для сравнения.
Цель -> не дублировать код в отдельных ноутбуках.
"""

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR, SVC

from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.config import RANDOM_STATE


def build_scaled_regression_pipeline(model) -> Pipeline:
    """
    Создает pipeline для моделей регрессии, чувствительных к масштабу признаков.

    Аргументы:
        - model: Модель регрессии

    Возвращает:
        - Pipeline с заполнением пропусков, стандартизацией и моделью
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_tree_regression_pipeline(model) -> Pipeline:
    """
    Создает pipeline для древесных моделей регрессии.

    Аргументы:
        - model: Модель регрессии

    Возвращает:
        - Pipeline с заполнением пропусков и моделью
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def build_scaled_classification_pipeline(model) -> Pipeline:
    """
    Создает pipeline для моделей классификации, чувствительных к масштабу признаков.

    Аргументы:
        - model: Модель классификации

    Возвращает:
        - Pipeline с заполнением пропусков, стандартизацией и моделью
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_tree_classification_pipeline(model) -> Pipeline:
    """
    Создает pipeline для древесных моделей классификации.

    Аргументы:
        - model: Модель классификации

    Возвращает:
        - Pipeline с заполнением пропусков и моделью
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def wrap_regressor_with_log_target(model) -> TransformedTargetRegressor:
    """
    Оборачивает модель регрессии в логарифмическое преобразование целевой переменной.

    Это необходимо для IC50, CC50 и SI, так как в EDA было видно,
    что целевые переменные имеют сильную правостороннюю асимметрию и выбросы.

    Аргументы:
        - model: Базовая модель регрессии.

    Возвращает:
        - TransformedTargetRegressor, который обучается на log1p(y),
            а предсказания возвращает обратно через expm1
    """

    return TransformedTargetRegressor(
        regressor=model,
        func=FunctionTransformer(func=lambda y: __import__("numpy").log1p(y)).transform,
        inverse_func=FunctionTransformer(func=lambda y: __import__("numpy").expm1(y)).transform,
    )


def get_regression_models(use_log_target: bool = False) -> dict[str, object]:
    """
    Создает набор моделей для задачи регрессии.

    Аргументы:
        - use_log_target: Нужно ли применять log1p-преобразование целевой переменной.

    Возвращает:
        - Словарь вида:
            название модели -> модель или pipeline
    """

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": build_scaled_regression_pipeline(LinearRegression()),
        "ridge": build_scaled_regression_pipeline(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "knn": build_scaled_regression_pipeline(KNeighborsRegressor(n_neighbors=7)),
        "svr_rbf": build_scaled_regression_pipeline(SVR(kernel="rbf", C=10.0, epsilon=0.1)),
        "random_forest": build_tree_regression_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "extra_trees": build_tree_regression_pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": build_tree_regression_pipeline(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
        ),
        "catboost": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
    }

    if not use_log_target:
        return models

    return {
        f"{model_name}_log_target": wrap_regressor_with_log_target(model)
        for model_name, model in models.items()
    }


def get_classification_models() -> dict[str, object]:
    """
    Создает набор моделей для задачи бинарной классификации.

    Возвращает:
        - Словарь вида:
            название модели -> модель или pipeline
    """

    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": build_scaled_classification_pipeline(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "knn": build_scaled_classification_pipeline(KNeighborsClassifier(n_neighbors=7)),
        "svc_rbf": build_scaled_classification_pipeline(
            SVC(
                kernel="rbf",
                C=10.0,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": build_tree_classification_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "extra_trees": build_tree_classification_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting": build_tree_classification_pipeline(
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
    }
