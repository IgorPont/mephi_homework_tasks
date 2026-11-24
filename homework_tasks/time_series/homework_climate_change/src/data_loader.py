"""
Модуль загрузки исходных климатических данных.

Данные загружаются из CSV-файлов, расположенных
в директории data/raw/.
"""

from pathlib import Path

import pandas as pd

from homework_tasks.time_series.homework_climate_change.src.config import (
    COUNTRY_TEMPERATURES_PATH,
)


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Загружает DataFrame из CSV-файла.

    Аргументы:
        - file_path: Путь к исходному CSV-файлу

    Возвращает:
        - Загруженный DataFrame

    Исключения:
        - FileNotFoundError: Если файл не найден
    """

    if not file_path.is_file():
        raise FileNotFoundError(f"Не найден файл с данными: {file_path}")

    return pd.read_csv(file_path)


def load_country_temperatures() -> pd.DataFrame:
    """
    Загружает исходные данные о среднемесячных температурах по странам.
    """

    return load_dataframe(COUNTRY_TEMPERATURES_PATH)