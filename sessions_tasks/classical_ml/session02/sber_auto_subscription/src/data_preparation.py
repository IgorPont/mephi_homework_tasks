"""
Модуль подготовки итогового датасета для обучения модели.

Скрипт выполняет полный цикл подготовки данных:
    - загружает исходные таблицы ga_sessions и ga_hits
    - формирует целевую переменную target
    - создает признаки на основе визитов и событий пользователей
    - сохраняет итоговый датасет в data/processed/sber_auto_dataset.csv

Запуск из корня проекта mephi_homework_tasks:
    poetry run python -m sessions_tasks.classical_ml.session02.sber_auto_subscription.src.data_preparation

Пример вывода:
    Загружаю исходные данные...
    Исходные данные загружены за 115.61 сек.
    Размер ga_sessions: (1860042, 18)
    Размер ga_hits: (15726470, 11)
    Формирую итоговый датасет...
    Итоговый датасет сформирован за 20.31 сек.
    Размер итогового датасета: (1860042, 34)
    Итоговый датасет сохранен: /Users/pontigor/python_test/mephi_homework_tasks/sessions_tasks/classical_ml/session02/
                               sber_auto_subscription/data/processed/sber_auto_dataset.csv
"""

from time import perf_counter

import pandas as pd

from sessions_tasks.classical_ml.session02.sber_auto_subscription.src.config import (
    PROCESSED_DATASET_PATH,
    PROCESSED_DATA_DIR,
    TARGET_ACTIONS,
    TARGET_COLUMN,
)
from sessions_tasks.classical_ml.session02.sber_auto_subscription.src.data_loader import (
    load_raw_data,
)
from sessions_tasks.classical_ml.session02.sber_auto_subscription.src.features import (
    build_model_dataset,
)


def prepare_dataset() -> pd.DataFrame:
    """
    Подготавливает итоговый датасет для обучения модели.

    Порядок работы:
        1. Загружает исходные данные ga_sessions и ga_hits
        2. Создает целевую переменную на уровне визитов
        3. Создает признаки из таблиц sessions и hits
        4. Возвращает готовый DataFrame

    Возвращает:
        - DataFrame с подготовленными признаками и целевой переменной
    """

    print("Загружаю исходные данные...", flush=True)

    start_time = perf_counter()
    sessions, hits = load_raw_data()

    print(
        f"Исходные данные загружены за {perf_counter() - start_time:.2f} сек.",
        flush=True,
    )
    print(f"Размер ga_sessions: {sessions.shape}", flush=True)
    print(f"Размер ga_hits: {hits.shape}", flush=True)

    print("Формирую итоговый датасет...", flush=True)

    start_time = perf_counter()
    dataset = build_model_dataset(
        sessions=sessions,
        hits=hits,
        target_actions=TARGET_ACTIONS,
        target_column=TARGET_COLUMN,
    )

    print(
        f"Итоговый датасет сформирован за {perf_counter() - start_time:.2f} сек.",
        flush=True,
    )
    print(f"Размер итогового датасета: {dataset.shape}", flush=True)

    return dataset


def save_dataset(dataset: pd.DataFrame) -> None:
    """
    Сохраняет подготовленный датасет в CSV-файл.

    Перед сохранением создает директорию data/processed,
    если она еще не существует.

    Аргументы:
        - dataset: подготовленный DataFrame для обучения модели
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(
        PROCESSED_DATASET_PATH,
        index=False,
    )

    print(f"Итоговый датасет сохранен: {PROCESSED_DATASET_PATH}", flush=True)


def main() -> None:
    """
    Точка входа для запуска подготовки данных из командной строки.
    """

    dataset = prepare_dataset()
    save_dataset(dataset)


if __name__ == "__main__":
    main()
