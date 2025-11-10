"""
Задание 4. Классификация методом Наивного Байеса.

Условия задания:
    Вам предоставлены данные о людях с наличием или отсутствием заболевания:
        путь к файлу - mephi_homework_tasks/sessions_tasks/math_analysis/session01/data/dataset_for_task_4_session_1.csv

    Признаки:
        Age_Group - возрастная группа.
        Test - позитивный или негативный тест на заболевание.
        Status - целевая переменная, есть инфекция или нет.
    Необходимо реализовать алгоритм Наивного Байеса для решения задачи классификации.
    Обязательно оцените качество полученного результата по итогу.

Запуск:
    poetry run python -m homework_tasks.sessions_tasks.math_analysis.session01.task04_naive_bayes

Пример вывода после запуска:

    =================== ЗАДАНИЕ 4 - Наивный Байес ===================
    Accuracy: 0.750  |  Baseline: 0.512
    Модель классифицирует правильно примерно в 75.0% случаев
    =================================================================
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score


def main() -> None:
    """
    Шаги решения:
        1) Чтение данных из CSV.
        2) Преобразование категориальных признаков в числовые.
        3) Разделение данных на обучающую и тестовую выборки.
        4) Обучение модели CategoricalNB.
        5) Расчет точности (accuracy) и вывод результата.
    """
    # 1) Формируем путь к CSV относительно текущего файла
    data_path = Path(__file__).with_name("data") / "dataset_for_task_4_session_1.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Не найден файл с данными: {data_path}")

    # 2) Читаем CSV и проверяем, что нужные столбцы на месте
    df = pd.read_csv(data_path)
    X_cols, y_col = ["Age_Group", "Test"], "Status"
    if not set(X_cols + [y_col]).issubset(df.columns):
        raise ValueError(f"Ожидаются столбцы {X_cols + [y_col]}, а получено: {list(df.columns)}")

    # 3) Кодируем категориальные значения в числовые коды для работы модели с категориальными признаками
    for col in X_cols + [y_col]:
        df[col] = pd.Categorical(df[col]).codes

    # Разделяем на признаки (X) и целевую переменную (y).
    X, y = df[X_cols], df[y_col]

    # 4) Разбиваем на обучающую и тестовую выборки (30% тест, фиксируем random_state для повторяемости),
    #    stratify=y - сохраняем пропорции классов в train и test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 5) Обучаем наивный Байес для категориальных признаков
    model = CategoricalNB()
    model.fit(X_train, y_train)

    # 6) Предсказываем на тесте и считаем метрики
    y_pred = model.predict(X_test)
    # Доля верных ответов
    acc = accuracy_score(y_test, y_pred)
    # Доля самого частого класса
    baseline = float(y_test.value_counts(normalize=True).max())

    # 7) Печатаем итог
    print("\n=================== ЗАДАНИЕ 4 - Наивный Байес ===================")
    print(f"Accuracy: {acc:.3f}  |  Baseline: {baseline:.3f}")
    print(f"Модель классифицирует правильно примерно в {acc * 100:.1f}% случаев")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
