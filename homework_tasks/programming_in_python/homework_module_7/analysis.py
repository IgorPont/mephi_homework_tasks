"""
Модуль с решениями задач модуля 7 (Очистка данных)
для датасета diabetes_data.csv.
"""

from pathlib import Path

import pandas as pd
import numpy as np

# --- Константы путей ---------------------------------------------------------

# Директория, в которой лежит текущий файл analysis.py
BASE_DIR: Path = Path(__file__).resolve().parent

# Полный путь к датасету с данными по диабету
DIABETES_FILE: Path = BASE_DIR / "data" / "diabetes_data.csv"


# --- Загрузка данных ---------------------------------------------------------

def load_diabetes_data(path: Path = DIABETES_FILE) -> pd.DataFrame:
    """
    Загрузить таблицу diabetes_data из CSV-файла.

    Parameters
    ----------
    path : Path
        Путь к файлу diabetes_data.csv.

    Returns
    -------
    pd.DataFrame
        Таблица с исходными данными по пациентам.
    """
    # Файл имеет стандартный разделитель ',', кодировку по умолчанию
    df = pd.read_csv(path)
    return df


# --- Задача 8.1: поиск и удаление дубликатов --------------------------------

def task_8_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Найти все полностью повторяющиеся строки в данных и удалить их.

    Для поиска дубликатов используются ВСЕ признаки в таблице,
    т.е. строка считается дубликатом, если совпадают значения
    во всех столбцах.

    Алгоритм действий:
        - найти все повторяющиеся строки в данных;
        - удалить их;
        - вывести / вернуть количество записей,
          оставшихся после удаления дубликатов.

    Parameters
    ----------
    df : pd.DataFrame
        Исходная таблица diabetes_data.

    Returns
    -------
    pd.DataFrame
        Новая таблица без дубликатов.
        Количество строк в этой таблице и есть ответ к задаче 8.1.
    """
    # Число строк до очистки
    initial_rows: int = df.shape[0]

    # Удаляем полностью повторяющиеся строки.
    # keep="first" оставляет первое вхождение, остальные удаляет.
    df_no_duplicates = df.drop_duplicates(keep="first")

    # Число строк после удаления дубликатов
    final_rows: int = df_no_duplicates.shape[0]

    # Сколько строк было удалено
    removed_rows: int = initial_rows - final_rows

    # Логируем для наглядности
    print("[8.1] Количество строк до удаления дубликатов:", initial_rows)
    print("[8.1] Количество строк после удаления дубликатов:", final_rows)
    print("[8.1] Количество удаленных строк-дубликатов:", removed_rows)

    # Для дальнейших задач нам понадобится уже очищенная таблица,
    # поэтому возвращаем именно ее.
    return df_no_duplicates


def task_8_2(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """
    Задание 8.2.
    Найти неинформативные признаки и удалить их.

    Признак считается неинформативным, если:
        - либо 95% значений одинаковые,
        - либо 95% записей уникальны (столбец почти полностью уникальный).

    Parameters
    ----------
    df : pd.DataFrame
        Очищенная от дублей таблица diabetes_data.
    threshold : float
        Порог неинформативности (по условию — 0.95).

    Returns
    -------
    (pd.DataFrame, list[str])
        Очищенная таблица,
        список удаленных признаков.
    """

    columns_to_remove: list[str] = []

    n = df.shape[0]  # количество строк

    for col in df.columns:
        # Доля самого частого значения
        top_freq_ratio = df[col].value_counts(normalize=True).max()

        # Доля уникальных значений
        unique_ratio = df[col].nunique() / n

        # Проверяем условия неинформативности
        if top_freq_ratio >= threshold or unique_ratio >= threshold:
            columns_to_remove.append(col)

    # Логируем
    print("[8.2] Найдены неинформативные признаки:", columns_to_remove)

    # Удаляем признаки
    df_cleaned = df.drop(columns=columns_to_remove)

    return df_cleaned, columns_to_remove


def task_8_3(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Задание 8.3.
    В столбцах Glucose, BloodPressure, SkinThickness, Insulin, BMI
    пропуски зашифрованы нулем. Надо заменить нули на np.nan и
    вычислить долю пропусков в столбце Insulin.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица после удаления дублей и неинформативных признаков.

    Returns
    -------
    (pd.DataFrame, float)
        Обновленная таблица,
        доля пропусков в Insulin (округленная до сотых).
    """
    import numpy as np

    cols_with_fake_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    # Заменяем нули на NaN
    df[cols_with_fake_missing] = df[cols_with_fake_missing].replace(0, np.nan)

    # Доля пропусков в Insulin
    missing_ratio = df["Insulin"].isna().mean()

    # Округляем до сотых по условию задачи
    missing_ratio_rounded = round(missing_ratio, 2)

    print("[8.3] Доля пропусков в столбце Insulin:", missing_ratio_rounded)

    return df, missing_ratio_rounded


def task_8_4(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Задание 8.4.
    Удалить признаки, в которых доля пропусков превышает 30 %.
    Вернуть обновлённый DataFrame и количество оставшихся признаков.

    Важно:
    - работаем с таблицей, уже очищенной от дублей (8.1)
      и неинформативных признаков (8.2)
    - считаем пропуски по столбцам .isna().mean()

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    (pd.DataFrame, int)
        Очищенная таблица,
        количество оставшихся признаков.
    """

    # Доля пропусков по каждому признаку
    missing_fraction = df.isna().mean()

    # Признаки, которые нужно удалить (> 30 % пропусков)
    cols_to_drop = missing_fraction[missing_fraction > 0.30].index.tolist()

    print("[8.4] Признаки с пропусками > 30%:", cols_to_drop)

    # Удаляем признаки
    df_cleaned = df.drop(columns=cols_to_drop)

    # Сколько признаков осталось
    num_features_left = df_cleaned.shape[1]

    print("[8.4] Количество признаков после удаления:", num_features_left)

    return df_cleaned, num_features_left


def task_8_5(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Задание 8.5.
    Удалить только те строки, где количество пропусков > 2.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    (pd.DataFrame, int)
        Очищенная таблица,
        количество строк после удаления.
    """

    # Считаем число пропусков в каждой строке
    row_missing_count = df.isna().sum(axis=1)

    # Маска строк, где <= 2 пропуска
    mask = row_missing_count <= 2

    # Фильтруем таблицу
    df_filtered = df[mask].copy()

    # Новое число строк
    new_rows_count = df_filtered.shape[0]

    print("[8.5] Удалено строк:", df.shape[0] - new_rows_count)
    print("[8.5] Осталось строк:", new_rows_count)

    return df_filtered, new_rows_count


def task_8_6(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Задание 8.6.
    Заменить пропуски на медиану в оставшихся данных.
    Найти среднее значение SkinThickness после заполнения.
    Округлить до десятых.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    (pd.DataFrame, float)
        Очищенная таблица,
        среднее значение SkinThickness (округлённое до 0.1)
    """

    # Заполняем пропуски медианами по каждому столбцу
    df_filled = df.fillna(df.median(numeric_only=True))

    # Среднее значение SkinThickness после заполнения медианой
    mean_skin = round(df_filled["SkinThickness"].mean(), 1)

    print("[8.6] Среднее SkinThickness после заполнения медианой:", mean_skin)

    return df_filled, mean_skin


def task_8_7(df: pd.DataFrame) -> int:
    """
    Задание 8.7.
    Найти количество выбросов в столбце SkinThickness
    по классическому правилу межквартильного размаха (IQR).

    Используется правило:
        выбросы < Q1 - 1.5*IQR
        выбросы > Q3 + 1.5*IQR
    """

    col = "SkinThickness"

    # Вычисляем квартилы
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Логическая маска выбросов
    mask_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

    outliers_count = int(mask_outliers.sum())

    print("[8.7] Количество выбросов в SkinThickness:", outliers_count)

    return outliers_count


def task_8_8(df: pd.DataFrame) -> int:
    """
    Задание 8.8.
    Найти количество выбросов в признаке SkinThickness
    по классическому методу z-отклонений.

    Правило:
        |z| > 3  → выброс
    """

    col = "SkinThickness"

    # Среднее и стандартное отклонение
    mean_val = df[col].mean()
    std_val = df[col].std()

    # Вычисляем z-оценку
    z_scores = (df[col] - mean_val) / std_val

    # Выбросы — где |z| > 3
    mask_outliers = z_scores.abs() > 3
    outliers_count = int(mask_outliers.sum())

    print("[8.8] Количество выбросов по z-отклонению:", outliers_count)

    return outliers_count


def task_8_9(df: pd.DataFrame) -> int:
    """
    Задание 8.9.
    Найти число выбросов в признаке DiabetesPedigreeFunction
    с помощью классического метода межквартильного размаха (IQR),
    затем логарифмировать признак и снова найти число выбросов.

    Вернуть разницу:
        (выбросы в исходном признаке) - (выбросы в логарифме признака).
    """

    col = "DiabetesPedigreeFunction"

    # --- 1. Выбросы в исходном признаке ---
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers_raw = df[(df[col] < lower) | (df[col] > upper)]
    n_raw = len(outliers_raw)

    print(f"[8.9] Выбросов в исходном признаке: {n_raw}")

    # --- 2. Логарифмирование (log без +1, по условию) ---
    df_log = df[col].apply(lambda x: np.log(x) if x > 0 else np.nan).dropna()

    # IQR после логарифмирования
    Q1_log = df_log.quantile(0.25)
    Q3_log = df_log.quantile(0.75)
    IQR_log = Q3_log - Q1_log

    lower_log = Q1_log - 1.5 * IQR_log
    upper_log = Q3_log + 1.5 * IQR_log

    outliers_log = df_log[(df_log < lower_log) | (df_log > upper_log)]
    n_log = len(outliers_log)

    print(f"[8.9] Выбросов после логарифмирования: {n_log}")

    # --- Разница ---
    diff = int(n_raw - n_log)
    print(f"[8.9] Разница (исходные минус логарифмированные): {diff}")

    return diff


# --- Точка входа для ручного запуска модуля ---------------------------------
if __name__ == "__main__":
    # Загружаем исходный датасет
    diabetes_df = load_diabetes_data()

    # === ЗАДАНИЕ 8.1 ===
    diabetes_df = task_8_1(diabetes_df)
    result_8_1: int = diabetes_df.shape[0]
    # Итоговый ответ, который нужно ввести в форму на Stepik
    print("[8.1] Итоговый ответ (число записей в данных после удаления дубликатов):", result_8_1)

    # === ЗАДАНИЕ 8.2 ===
    diabetes_df, removed_cols_8_2 = task_8_2(diabetes_df)
    print("[8.2] Итоговый ответ (неинформативные признаки):", removed_cols_8_2)

    # === ЗАДАНИЕ 8.3 ===
    diabetes_df, insulin_missing_ratio = task_8_3(diabetes_df)
    print("[8.3] Итоговый ответ (доля пропусков Insulin):", insulin_missing_ratio)

    # === ЗАДАНИЕ 8.4 ===
    diabetes_df, num_features_left = task_8_4(diabetes_df)
    print("[8.4] Итоговый ответ (число признаков):", num_features_left)

    # === ЗАДАНИЕ 8.5 ===
    diabetes_df, rows_left = task_8_5(diabetes_df)
    print("[8.5] Итоговый ответ (число строк):", rows_left)

    # === ЗАДАНИЕ 8.6 ===
    diabetes_df, skin_mean = task_8_6(diabetes_df)
    print("[8.6] Итоговый ответ (среднее SkinThickness):", skin_mean)

    # === ЗАДАНИЕ 8.7 ===
    outliers_8_7 = task_8_7(diabetes_df)
    print("[8.7] Итоговый ответ (количество выбросов):", outliers_8_7)

    # === ЗАДАНИЕ 8.8 ===
    outliers_8_8 = task_8_8(diabetes_df)
    print("[8.8] Итоговый ответ (количество выбросов):", outliers_8_8)

    # === ЗАДАНИЕ 8.9 ===
    result_8_9 = task_8_9(diabetes_df)
    print("[8.9] Итоговый ответ (разница выбросов):", result_8_9)
