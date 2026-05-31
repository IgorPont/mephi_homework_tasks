"""
Модуль обучения модели для проекта анализа сайта "СберАвтоподписка".

Скрипт выполняет:
    - загрузку подготовленного датасета
    - разделение данных на train/test
    - обучение baseline-модели
    - обучение LogisticRegression
    - расчет метрик качества
    - сохранение обученного sklearn Pipeline
    - сохранение метрик и предсказаний

Запуск из корня проекта mephi_homework_tasks:
    poetry run python -m sessions_tasks.classical_ml.session02.sber_auto_subscription.src.train_model

Пример вывода:

    Загружаю подготовленный датасет...
    Размер датасета: (1860042, 34)
    Распределение target: {0: 0.9662, 1: 0.0338}
    Количество числовых признаков: 21
    Количество категориальных признаков: 12
    X_train: (1488033, 33)
    X_test: (372009, 33)
    y_train target rate: 3.3818%
    y_test target rate: 3.3816%

    Обучаю baseline-модель...

    DummyClassifier
    Accuracy: 0.9662
    ROC-AUC: 0.5000
    Confusion matrix:
    [[359429      0]
     [ 12580      0]]
                  precision    recall  f1-score   support

               0       0.97      1.00      0.98    359429
               1       0.00      0.00      0.00     12580

        accuracy                           0.97    372009
       macro avg       0.48      0.50      0.49    372009
    weighted avg       0.93      0.97      0.95    372009


    Обучаю LogisticRegression...

    LogisticRegression
    Accuracy: 0.8700
    ROC-AUC: 0.9272
    Confusion matrix:
    [[313399  46030]
     [  2342  10238]]
                  precision    recall  f1-score   support

               0       0.99      0.87      0.93    359429
               1       0.18      0.81      0.30     12580

        accuracy                           0.87    372009
       macro avg       0.59      0.84      0.61    372009
    weighted avg       0.97      0.87      0.91    372009


    Предсказания сохранены: /Users/pontigor/python_test/mephi_homework_tasks/sessions_tasks/classical_ml/
                            session02/sber_auto_subscription/outputs/predictions.csv
    Метрики сохранены: /Users/pontigor/python_test/mephi_homework_tasks/sessions_tasks/classical_ml/session02/
                       sber_auto_subscription/outputs/metrics.json
    Модель сохранена: /Users/pontigor/python_test/mephi_homework_tasks/sessions_tasks/classical_ml/session02/
                      sber_auto_subscription/models/model.pkl

    Обучение завершено за 49.72 сек.
"""

import json
import pickle
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sessions_tasks.classical_ml.session02.sber_auto_subscription.src.config import (
    METRICS_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    PREDICTIONS_PATH,
    PROCESSED_DATASET_PATH,
    TARGET_COLUMN,
)


def load_dataset() -> pd.DataFrame:
    """
    Загружает подготовленный датасет для обучения модели.

    Возвращает:
        - DataFrame с признаками и целевой переменной

    Исключения:
        - FileNotFoundError -> если подготовленный датасет не найден
    """

    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Подготовленный датасет не найден. "
            "Сначала запустите подготовку данных командой: "
            "poetry run python -m "
            "sessions_tasks.classical_ml.session02.sber_auto_subscription.src.data_preparation"
        )

    dataset = pd.read_csv(PROCESSED_DATASET_PATH)

    return dataset


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Разделяет датасет на признаки и целевую переменную.

    Аргументы:
        - dataset: DataFrame с признаками и target

    Возвращает:
        - X: признаки
        - y: целевая переменная
    """

    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    return X, y


def get_feature_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Определяет числовые и категориальные признаки.

    Аргументы:
        - X: DataFrame с признаками

    Возвращает:
        - numeric_features: список числовых признаков
        - categorical_features: список категориальных признаков
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Создает preprocessing pipeline для числовых и категориальных признаков.

    Для числовых признаков:
        - заполнение пропусков медианой
        - масштабирование

    Для категориальных признаков:
        - заполнение пропусков значением unknown
        - OneHotEncoder с обработкой неизвестных категорий
        - объединение редких категорий через min_frequency

    Аргументы:
        - numeric_features: список числовых признаков
        - categorical_features: список категориальных признаков

    Возвращает:
        - ColumnTransformer для подготовки признаков
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=100)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_baseline_model(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Создает baseline-модель.

    DummyClassifier не ищет закономерности в данных.
    Он нужен только для сравнения с основной моделью.

    Аргументы:
        - preprocessor: объект подготовки признаков

    Возвращает:
        - sklearn Pipeline с DummyClassifier
    """

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )


def build_logistic_regression_model(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Создает основную модель LogisticRegression.

    LogisticRegression выбрана как итоговая модель, так как в ноутбуке
    она показала лучший ROC-AUC и при этом остается интерпретируемой.

    Аргументы:
        - preprocessor: объект подготовки признаков

    Возвращает:
        - sklearn Pipeline с LogisticRegression
    """

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """
    Оценивает качество модели на тестовой выборке.

    Аргументы:
        - model: обученный sklearn Pipeline
        - X_test: тестовые признаки
        - y_test: истинные значения target
        - model_name: название модели

    Возвращает:
        - словарь с метриками, предсказаниями и вероятностями
    """

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion matrix:")
    print(matrix)
    # Вывод без лишнего шума
    print(classification_report(y_test, predictions, zero_division=0))

    return {
        "model_name": model_name,
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(roc_auc), 4),
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": matrix.tolist(),
    }


def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    probabilities,
) -> None:
    """
    Сохраняет предсказания модели на тестовой выборке.

    Аргументы:
        - X_test: тестовые признаки
        - y_test: истинные значения target
        - predictions: предсказанные классы
        - probabilities: вероятности класса 1
    """

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = X_test.copy()
    predictions_df[TARGET_COLUMN] = y_test.values
    predictions_df["prediction"] = predictions
    predictions_df["probability"] = probabilities

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nПредсказания сохранены: {PREDICTIONS_PATH}")


def save_metrics(metrics: dict[str, Any]) -> None:
    """
    Сохраняет метрики модели в JSON-файл.

    Аргументы:
        - metrics: словарь с метриками
    """

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    print(f"Метрики сохранены: {METRICS_PATH}")


def save_model(model: Pipeline) -> None:
    """
    Сохраняет обученную модель в pickle-файл.

    Аргументы:
        - model: обученный sklearn Pipeline
    """

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"Модель сохранена: {MODEL_PATH}")


def main() -> None:
    """
    Точка входа для обучения модели из командной строки.
    """

    start_time = perf_counter()

    print("Загружаю подготовленный датасет...", flush=True)
    dataset = load_dataset()

    print(f"Размер датасета: {dataset.shape}", flush=True)
    print(
        "Распределение target:",
        dataset[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
        flush=True,
    )

    X, y = split_features_target(dataset)

    numeric_features, categorical_features = get_feature_columns(X)

    print(f"Количество числовых признаков: {len(numeric_features)}", flush=True)
    print(f"Количество категориальных признаков: {len(categorical_features)}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"X_train: {X_train.shape}", flush=True)
    print(f"X_test: {X_test.shape}", flush=True)
    print(f"y_train target rate: {y_train.mean():.4%}", flush=True)
    print(f"y_test target rate: {y_test.mean():.4%}", flush=True)

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    baseline_model = build_baseline_model(preprocessor=preprocessor)

    print("\nОбучаю baseline-модель...", flush=True)
    baseline_model.fit(X_train, y_train)

    baseline_metrics = evaluate_model(
        model=baseline_model,
        X_test=X_test,
        y_test=y_test,
        model_name="DummyClassifier",
    )

    logreg_model = build_logistic_regression_model(preprocessor=preprocessor)

    print("\nОбучаю LogisticRegression...", flush=True)
    logreg_model.fit(X_train, y_train)

    logreg_metrics = evaluate_model(
        model=logreg_model,
        X_test=X_test,
        y_test=y_test,
        model_name="LogisticRegression",
    )

    metrics = {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "roc_auc": baseline_metrics["roc_auc"],
            "confusion_matrix": baseline_metrics["confusion_matrix"],
        },
        "logistic_regression": {
            "accuracy": logreg_metrics["accuracy"],
            "roc_auc": logreg_metrics["roc_auc"],
            "confusion_matrix": logreg_metrics["confusion_matrix"],
        },
        "best_model": {
            "name": "LogisticRegression",
            "accuracy": logreg_metrics["accuracy"],
            "roc_auc": logreg_metrics["roc_auc"],
        },
    }

    save_predictions(
        X_test=X_test,
        y_test=y_test,
        predictions=logreg_metrics["predictions"],
        probabilities=logreg_metrics["probabilities"],
    )

    save_metrics(metrics=metrics)
    save_model(model=logreg_model)

    print(
        f"\nОбучение завершено за {perf_counter() - start_time:.2f} сек.",
        flush=True,
    )


if __name__ == "__main__":
    main()
