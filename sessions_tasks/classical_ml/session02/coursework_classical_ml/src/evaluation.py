"""
Оценка качества моделей для задач регрессии и классификации.

Модуль содержит функции для расчета основных метрик качества,
которые используются в экспериментах курсовой работы.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression_model(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Рассчитывает метрики качества для задачи регрессии.

    Аргументы:
        - y_true: Истинные значения целевой переменной.
        - y_pred: Предсказанные значения модели.

    Возвращает:
        - Словарь с метриками MAE, RMSE, R2 и MAPE.

    Примечание:
        - MAE показывает среднюю абсолютную ошибку.
        - RMSE сильнее штрафует крупные ошибки.
        - R2 показывает долю объясненной дисперсии.
        - MAPE показывает среднюю относительную ошибку в процентах.
    """

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    mae = mean_absolute_error(y_true_array, y_pred_array)
    rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
    r2 = r2_score(y_true_array, y_pred_array)

    non_zero_mask = y_true_array != 0

    if non_zero_mask.any():
        mape = np.mean(
            np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask]
                   )
        ) * 100
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


def evaluate_classification_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Рассчитывает метрики качества для задачи бинарной классификации.

    Аргументы:
        - y_true: Истинные значения классов.
        - y_pred: Предсказанные классы.
        - y_proba: Вероятности положительного класса, если модель умеет их возвращать.

    Возвращает:
        - Словарь с метриками accuracy, balanced_accuracy, precision, recall, f1 и roc_auc.

    Примечание:
        - Accuracy показывает общую долю верных ответов.
        - Balanced accuracy используется при дисбалансе классов.
        - Precision показывает точность положительных предсказаний.
        - Recall показывает полноту нахождения положительного класса.
        - F1 является балансом между precision и recall.
        - ROC-AUC оценивает качество ранжирования объектов по вероятности класса 1.
    """

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_proba))
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics
