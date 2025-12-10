"""
Модуль: Линейная регрессия со стохастическим градиентным спуском (SGD) на датасете алмазов.

Цель
-----
Построить модель линейной регрессии для предсказания цены алмазов с помощью
`sklearn.linear_model.SGDRegressor`, подбирая гиперпараметры через `GridSearchCV`.
Мы воспроизводим шаги из задания:
    1) загрузка датасета `diamonds` (Seaborn),
    2) удаление признаков: depth, table, x, y, z,
    3) логарифмирование признаков: carat и целевой price (через log1p),
    4) one-hot кодирование категориальных (drop_first=True),
    5) стандартизация числовых признаков,
    6) разбиение на train/test (test_size=0.33, random_state=42),
    7) подбор параметров SGD по заданной сетке,
    8) оценка по MSE и вывод значения, округленного до 3 знаков после запятой.

Почему именно такие шаги
------------------------
    - Логарифмирование `log1p` уменьшает влияние «тяжелых хвостов» и стабилизирует дисперсию.
    - Для линейной модели со стохастическим обучением масштабы признаков критичны —
      поэтому числовые признаки стандартизируются.
    - One-hot c `drop_first=True` повторяет поведение `pd.get_dummies(..., drop_first=True)`
      и минимизирует мультиколлинеарность (исключаем дамми-ловушку).
    - Сетка параметров взята из условия (loss, penalty, alpha, l1_ratio, learning_rate, eta0).

Зависимости/заметки
-------------------
    - `seaborn.load_dataset("diamonds")` загружает датасет из интернета (из репозитория Seaborn).
      Если интернет недоступен, сохраните CSV локально и замените загрузку в `load_data()`.
    - `OneHotEncoder(sparse_output=False)` требует scikit-learn >= 1.2.
      Для более старых версий используйте `sparse=False`.

Вывод
-----
Скрипт печатает:
    - лучшие гиперпараметры (`Best params:`),
    - MSE на тесте и округленный до 3 знаков вариант.

Запуск
------
    poetry run python mephi_homework_tasks/homework_tasks/ml/sgd_diamonds.py
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def load_data() -> pd.DataFrame:
    """
    Загружает датасет `diamonds` из Seaborn.

    Важно:
        Функция `seaborn.load_dataset` подтягивает датасет из онлайн-источника.
        Если интернет недоступен, замените реализацию на чтение локального CSV:
            return pd.read_csv("path/to/diamonds.csv")

    Возвращает:
        pd.DataFrame: исходный DataFrame с колонками, принятыми в датасете Seaborn.
    """
    import seaborn as sns  # локальный импорт, чтобы модуль не требовал seaborn при статическом анализе

    df = sns.load_dataset("diamonds")
    return df


def preprocess_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Выполняет препроцессинг на уровне DataFrame (как требует условие задания):
      - удаляет `depth`, `table`, `x`, `y`, `z`;
      - логарифмирует `carat` и целевую `price` через log1p;
      - разделяет на признаки X и целевую y.

    Параметры:
        df (pd.DataFrame): исходный датафрейм.

    Возвращает:
        Tuple[pd.DataFrame, pd.Series]: (X, y) — признаки и целевая.
    """
    # 1) удалить ненужные признаки
    df = df.drop(columns=["depth", "table", "x", "y", "z"])

    # 2) логарифмирование устойчивой функцией log1p (корректно для нулей)
    df["carat"] = np.log1p(df["carat"])
    df["price"] = np.log1p(df["price"])

    # 3) разделение на X/y
    X = df.drop(columns="price")
    y = df["price"]
    return X, y


def make_preprocess(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Создает препроцессор признаков для пайплайна sklearn.

    Что делает:
        - числовые: идентичный трансформер (на случай будущего расширения) + StandardScaler,
        - категориальные: OneHotEncoder(drop='first', handle_unknown='ignore').

    Параметры:
        numeric_features (List[str]): список названий числовых признаков.
        categorical_features (List[str]): список названий категориальных признаков.

    Возвращает:
        ColumnTransformer: объект для шага "prep" в Pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            # Явная тождественная трансформация с корректными именами фич.
            ("identity", FunctionTransformer(lambda z: z, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Для sklearn < 1.2 замените sparse_output=False на sparse=False
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocess


def make_param_grid() -> Dict[str, Any]:
    """
    Возвращает сетку гиперпараметров для GridSearchCV согласно условию задания.

    Возвращает:
        Dict[str, Any]: словарь параметров для `GridSearchCV`.
    """
    return {
        "sgd__loss": ["squared_error", "epsilon_insensitive"],
        "sgd__penalty": ["elasticnet"],
        "sgd__alpha": np.logspace(-3, 3, 10),
        "sgd__l1_ratio": np.linspace(0, 1, 10),
        "sgd__learning_rate": ["constant"],
        "sgd__eta0": np.logspace(-4, -1, 4),
    }


def make_pipeline(preprocess: ColumnTransformer, random_state: int = 42) -> Pipeline:
    """
    Формирует sklearn-пайплайн: препроцессинг → SGDRegressor.

    Параметры:
        preprocess (ColumnTransformer): шаг препроцессинга признаков.
        random_state (int): сид для воспроизводимости.

    Возвращает:
        Pipeline: готовый пайплайн для обучения/поиска параметров.
    """
    model = SGDRegressor(
        random_state=random_state,
        max_iter=2000,
        tol=1e-3,
    )
    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("sgd", model),
        ]
    )
    return pipe


def run_experiment(random_state: int = 42) -> Dict[str, Any]:
    """
    Полный цикл эксперимента:
      1) загрузка данных,
      2) препроцессинг фрейма,
      3) train/test split,
      4) сборка пайплайна и сетки параметров,
      5) GridSearchCV (cv=5, scoring='neg_mean_squared_error'),
      6) финальная оценка на тесте.

    Параметры:
        random_state (int): сид для разбиения и модели.

    Возвращает:
        Dict[str, Any]: метаданные эксперимента:
            {
              "best_params": dict,
              "mse": float,
              "mse_rounded_3": str,
            }
    """
    # 1) данные
    df = load_data()

    # 2) препроцессинг DataFrame по условию
    X, y = preprocess_frame(df)

    # 3) разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state
    )

    # 4) схемы признаков
    numeric_features = ["carat"]
    categorical_features = ["cut", "color", "clarity"]

    preprocess = make_preprocess(numeric_features, categorical_features)
    pipe = make_pipeline(preprocess, random_state=random_state)

    # 5) подбор параметров
    param_grid = make_param_grid()
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    # 6) итоговая оценка
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    result = {
        "best_params": search.best_params_,
        "mse": mse,
        "mse_rounded_3": f"{mse:.3f}",
    }
    return result


if __name__ == "__main__":
    res = run_experiment(random_state=42)
    print("Best params:", res["best_params"])
    print("MSE (test):", res["mse"])
    print("MSE rounded to 3 decimals:", res["mse_rounded_3"])
