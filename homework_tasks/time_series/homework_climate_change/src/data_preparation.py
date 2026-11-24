"""
Модуль подготовки данных для прогнозирования среднемесячной температуры.

Содержит функции подготовки временного ряда, обработки пропущенных
значений и формирования выборок для обучения и оценки моделей.
"""

from collections.abc import Iterator

import pandas as pd

from homework_tasks.time_series.homework_climate_change.src.config import (
    COUNTRY,
    DATE_COLUMN,
    FORECAST_HORIZON,
    FREQUENCY,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_WINDOW_SIZE,
    WINDOW_STEP,
)


def prepare_temperature_series(
    country_temperatures: pd.DataFrame,
    country: str = COUNTRY,
) -> pd.Series:
    """
    Формирует временной ряд среднемесячных температур выбранной страны.

    Наблюдения сортируются в хронологическом порядке.
    Пропущенные значения температуры сохраняются для дальнейшего анализа.

    Аргументы:
        - country_temperatures: Исходные климатические данные
        - country: Название страны

    Возвращает:
        - Временной ряд температур с индексом DatetimeIndex

    Исключения:
        - ValueError: Если страна отсутствует в датасете
          или даты наблюдений не являются уникальными
    """

    temperatures = country_temperatures.loc[
        country_temperatures["Country"] == country,
        [DATE_COLUMN, TARGET_COLUMN],
    ].copy()

    if temperatures.empty:
        raise ValueError(f"Страна {country} отсутствует в датасете")

    temperatures[DATE_COLUMN] = pd.to_datetime(
        temperatures[DATE_COLUMN],
        format="%Y-%m-%d",
    )

    if temperatures[DATE_COLUMN].isna().any():
        raise ValueError("Обнаружены пропущенные даты наблюдений")

    if temperatures[DATE_COLUMN].duplicated().any():
        raise ValueError(
            f"Обнаружены повторяющиеся даты наблюдений для страны {country}"
        )

    temperature_series = (
        temperatures.set_index(DATE_COLUMN)
        .sort_index()[TARGET_COLUMN]
    )

    return temperature_series


def split_temperature_series(
    temperature_series: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.Series, pd.Series]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки.

    Разбиение выполняется в хронологическом порядке без перемешивания.
    Первые наблюдения используются для обучения,
    последние для оценки качества прогнозирования.

    Аргументы:
        - temperature_series: Подготовленный временной ряд температур
        - test_size: Доля наблюдений для тестовой выборки

    Возвращает:
        - train_series: Обучающая выборка
        - test_series: Тестовая выборка

    Исключения:
        - ValueError: Если параметры разбиения или структура
          временного ряда некорректны
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться между 0 и 1")

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

    split_index = int(len(temperature_series) * (1 - test_size))

    if not 0 < split_index < len(temperature_series):
        raise ValueError(
            "Недостаточно наблюдений для формирования train и test"
        )

    train_series = temperature_series.iloc[:split_index].copy()
    test_series = temperature_series.iloc[split_index:].copy()

    return train_series, test_series


def generate_rolling_splits(
    temperature_series: pd.Series,
    train_window_size: int = TRAIN_WINDOW_SIZE,
    forecast_horizon: int = FORECAST_HORIZON,
    step: int = WINDOW_STEP,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """
    Формирует последовательные разбиения методом скользящего окна.

    Размер обучающего окна и горизонт прогнозирования остаются
    фиксированными. После каждого разбиения границы окон смещаются
    вперед на заданное количество месяцев.

    Неполные окна не формируются.

    Аргументы:
        - temperature_series: Подготовленный временной ряд
        - train_window_size: Размер обучающего окна в месяцах
        - forecast_horizon: Размер валидационного окна в месяцах
        - step: Шаг смещения окон в месяцах

    Возвращает:
        - Итератор пар обучающих и валидационных временных рядов

    Исключения:
        - ValueError: Если параметры разбиения, структура
          или размер временного ряда некорректны
    """

    if train_window_size < 1:
        raise ValueError("train_window_size должен быть положительным")

    if forecast_horizon < 1:
        raise ValueError("forecast_horizon должен быть положительным")

    if step < 1:
        raise ValueError("step должен быть положительным")

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

    window_size = train_window_size + forecast_horizon

    if len(temperature_series) < window_size:
        raise ValueError(
            "Недостаточно наблюдений для формирования скользящего окна"
        )

    for start_index in range(
        0,
        len(temperature_series) - window_size + 1,
        step,
    ):
        train_end_index = start_index + train_window_size
        validation_end_index = train_end_index + forecast_horizon

        train_window = temperature_series.iloc[
            start_index:train_end_index
        ].copy()

        validation_window = temperature_series.iloc[
            train_end_index:validation_end_index
        ].copy()

        yield train_window, validation_window
