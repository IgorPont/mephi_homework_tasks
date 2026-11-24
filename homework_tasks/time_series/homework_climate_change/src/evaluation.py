"""
Модуль оценки качества прогнозирования среднемесячной температуры.

Содержит функции расчета метрик ошибок прогнозирования.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def calculate_forecast_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Рассчитывает метрики качества прогнозирования временного ряда.

    Аргументы:
        - y_true: Фактические значения температуры
        - y_pred: Прогнозные значения температуры

    Возвращает:
        - Словарь с метриками MAE, RMSE и MAPE

    Исключения:
        - ValueError: Если наблюдения отсутствуют,
          временные индексы не совпадают,
          значения не являются конечными
          или MAPE не определена
    """

    if y_true.empty or y_pred.empty:
        raise ValueError("Фактические и прогнозные значения отсутствуют")

    if not y_true.index.equals(y_pred.index):
        raise ValueError(
            "Временные индексы фактических и прогнозных "
            "значений должны совпадать"
        )

    if not np.isfinite(y_true.to_numpy()).all():
        raise ValueError("Фактические значения должны быть конечными")

    if not np.isfinite(y_pred.to_numpy()).all():
        raise ValueError("Прогнозные значения должны быть конечными")

    if y_true.eq(0).any():
        raise ValueError(
            "MAPE не определена при нулевых фактических значениях"
        )

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(
            mean_absolute_percentage_error(y_true, y_pred) * 100
        ),
    }
