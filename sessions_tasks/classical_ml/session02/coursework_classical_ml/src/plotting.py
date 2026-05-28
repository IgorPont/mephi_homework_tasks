"""
Функции визуализации для курсовой работы.

Модуль содержит переиспользуемые функции для построения графиков
по результатам EDA и модельных экспериментов.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_regression_results(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Сравнение моделей регрессии",
) -> None:
    """
    Строит столбчатую диаграмму качества моделей регрессии.

    Аргументы:
        - results_df: DataFrame с результатами эксперимента
        - metric: Метрика для сравнения моделей
        - title: Заголовок графика
    """

    sorted_results = results_df.sort_values(by=metric, ascending=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=sorted_results,
        x=metric,
        y="model",
    )
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("Модель")
    plt.grid(axis="x", alpha=0.3)
    plt.show()


def plot_classification_results(
    results_df: pd.DataFrame,
    metric: str = "f1",
    title: str = "Сравнение моделей классификации",
) -> None:
    """
    Строит столбчатую диаграмму качества моделей классификации.

    Аргументы:
        - results_df: DataFrame с результатами эксперимента
        - metric: Метрика для сравнения моделей
        - title: Заголовок графика
    """

    sorted_results = results_df.sort_values(by=metric, ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=sorted_results,
        x=metric,
        y="model",
    )
    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("Модель")
    plt.grid(axis="x", alpha=0.3)
    plt.show()


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    title: str = "Фактические значения и предсказания",
) -> None:
    """
    Строит scatter plot фактических и предсказанных значений.

    Аргументы:
        - y_true: Истинные значения
        - y_pred: Предсказанные значения
        - title: Заголовок графика
    """

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)

    min_value = min(min(y_true), min(y_pred))
    max_value = max(max(y_true), max(y_pred))

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.title(title)
    plt.xlabel("Фактическое значение")
    plt.ylabel("Предсказанное значение")
    plt.grid(alpha=0.3)
    plt.show()


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Важность признаков",
) -> pd.DataFrame:
    """
    Строит график важности признаков для моделей, которые поддерживают feature_importances_.

    Аргументы:
        - model: Обученная модель или pipeline
        - feature_names: Список названий признаков
        - top_n: Количество наиболее важных признаков
        - title: Заголовок графика

    Возвращает:
        - DataFrame с важностью признаков
    """

    final_model = model

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        final_model = model.named_steps["model"]

    if not hasattr(final_model, "feature_importances_"):
        raise ValueError("Переданная модель не поддерживает атрибут feature_importances_.")

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": final_model.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    top_importance_df = importance_df.head(top_n)

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_importance_df,
        x="importance",
        y="feature",
    )
    plt.title(title)
    plt.xlabel("Важность")
    plt.ylabel("Признак")
    plt.grid(axis="x", alpha=0.3)
    plt.show()

    return importance_df
