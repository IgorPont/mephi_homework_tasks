"""
Модуль генерации признаков для прогнозирования среднемесячной температуры.

Содержит функции формирования лаговых и календарных признаков
на основе подготовленного временного ряда.
"""

import pandas as pd

from homework_tasks.time_series.homework_climate_change.src.config import (
    FREQUENCY,
    LAGS,
    TARGET_COLUMN,
    WINTER_MONTHS,
)


def build_temperature_features(
    temperature_series: pd.Series,
    lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    """
    Формирует лаговые и календарные признаки временного ряда.

    Лаговые признаки используют только предшествующие наблюдения.
    Первые строки, для которых отсутствует необходимая история,
    исключаются из результирующего набора данных.

    Аргументы:
        - temperature_series: Подготовленный временной ряд температур
        - lags: Положительные значения лагов в месяцах

    Возвращает:
        - DataFrame с целевой переменной, лаговыми
          и календарными признаками

    Исключения:
        - ValueError: Если параметры генерации признаков
          или структура временного ряда некорректны
    """

    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("Лаги должны быть положительными")

    if len(lags) != len(set(lags)):
        raise ValueError("Значения лагов должны быть уникальными")

    if not isinstance(temperature_series.index, pd.DatetimeIndex):
        raise ValueError("Временной ряд должен иметь индекс DatetimeIndex")

    if not temperature_series.index.is_unique:
        raise ValueError("Даты наблюдений должны быть уникальными")

    if not temperature_series.index.is_monotonic_increasing:
        raise ValueError("Наблюдения должны быть отсортированы по времени")

    if temperature_series.index.freqstr != FREQUENCY:
        raise ValueError(
            f"Временной ряд должен иметь частоту {FREQUENCY}"
        )

    if temperature_series.isna().any():
        raise ValueError("Временной ряд содержит пропущенные значения")

    if len(temperature_series) <= max(lags):
        raise ValueError(
            "Недостаточно наблюдений для формирования лаговых признаков"
        )

    features = temperature_series.to_frame(name=TARGET_COLUMN)

    for lag in lags:
        features[f"lag_{lag}"] = temperature_series.shift(lag)

    features["month"] = features.index.month
    features["is_winter"] = (
        features["month"].isin(WINTER_MONTHS).astype(int)
    )

    # Исключаем начальные наблюдения без необходимой истории
    return features.iloc[max(lags):].copy()
