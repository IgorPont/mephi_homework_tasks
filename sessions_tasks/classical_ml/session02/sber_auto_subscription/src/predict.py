"""
Модуль получения предсказаний для проекта анализа сайта "СберАвтоподписка".

Скрипт загружает сохраненную модель из models/model.pkl,
берет пример визита из подготовленного датасета и возвращает:
    - prediction: предсказанный класс 0 или 1
    - probability: вероятность совершения целевого действия

Важно:
    Финальная модель использует только признаки, доступные на момент начала визита.
    Поведенческие агрегаты по событиям внутри всей сессии в модель не передаются,
    так как они могут приводить к data leakage.

Запуск из корня проекта mephi_homework_tasks:
    poetry run python -m sessions_tasks.classical_ml.session02.sber_auto_subscription.src.predict

Пример вывода:
    {
        "prediction": 0,
        "probability": 0.1234,
        "true_target": 0
    }

    Расшифровка:
        prediction = 0 -> модель считает, что целевого действия не будет
        prediction = 1 -> модель считает, что целевое действие будет
        probability -> вероятность целевого действия
        true_target -> фактическое значение из датасета, добавлено только для проверки
"""

import json
import pickle
from typing import Any

import pandas as pd

from sessions_tasks.classical_ml.session02.sber_auto_subscription.src.config import (
    MODEL_PATH,
    PROCESSED_DATASET_PATH,
    TARGET_COLUMN,
)


def load_model() -> Any:
    """
    Загружает обученную модель из pickle-файла.

    Возвращает:
        - Обученный sklearn Pipeline

    Исключения:
        - FileNotFoundError -> если файл модели не найден
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Файл модели не найден. "
            "Сначала обучите модель командой: "
            "poetry run python -m "
            "sessions_tasks.classical_ml.session02.sber_auto_subscription.src.train_model"
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


def load_sample_visit(row_index: int = 0) -> tuple[pd.DataFrame, int | None]:
    """
    Загружает одну строку из подготовленного датасета для демонстрации предсказания.

    Аргументы:
        - row_index: индекс строки, которую нужно взять из датасета

    Возвращает:
        - sample_features: DataFrame с одной строкой признаков
        - true_target: фактическое значение target, если оно есть
    """

    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Подготовленный датасет не найден. "
            "Сначала запустите подготовку данных командой: "
            "poetry run python -m "
            "sessions_tasks.classical_ml.session02.sber_auto_subscription.src.data_preparation"
        )

    dataset = pd.read_csv(PROCESSED_DATASET_PATH)

    if row_index < 0 or row_index >= len(dataset):
        raise IndexError(
            f"Некорректный row_index={row_index}. "
            f"Допустимый диапазон: от 0 до {len(dataset) - 1}"
        )

    sample = dataset.iloc[[row_index]].copy()

    true_target = None

    if TARGET_COLUMN in sample.columns:
        true_target = int(sample[TARGET_COLUMN].iloc[0])
        sample_features = sample.drop(columns=[TARGET_COLUMN])
    else:
        sample_features = sample

    return sample_features, true_target


def predict_visit(
    model: Any,
    visit_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Получает предсказание модели для одного или нескольких визитов.

    Аргументы:
        - model: обученный sklearn Pipeline
        - visit_data: DataFrame с признаками визита

    Возвращает:
        - Словарь с предсказанным классом и вероятностью класса 1
    """

    prediction = int(model.predict(visit_data)[0])
    probability = float(model.predict_proba(visit_data)[0, 1])

    result = {
        "prediction": prediction,
        "probability": round(probability, 4),
    }

    return result


def main() -> None:
    """
    Точка входа для получения демонстрационного предсказания.
    """

    model = load_model()

    sample_features, true_target = load_sample_visit(row_index=0)

    result = predict_visit(
        model=model,
        visit_data=sample_features,
    )

    if true_target is not None:
        result["true_target"] = true_target

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()
