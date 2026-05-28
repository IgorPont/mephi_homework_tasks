"""
Загрузка исходных данных для курсовой работы.

Модуль отвечает за чтение датасета и базовую проверку его структуры.
Предобработка, очистка и генерация признаков выполняются в отдельных модулях.
"""

from pathlib import Path

import pandas as pd

from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.config import DATA_FILE_PATH


def load_coursework_dataset(file_path: Path = DATA_FILE_PATH) -> pd.DataFrame:
    """
    Загружает датасет курсовой работы из Excel-файла.

    Аргументы:
        - file_path: Путь к Excel-файлу с исходными данными.

    Возвращает:
        - DataFrame с исходными данными.

    Исключения:
        - FileNotFoundError: Если файл с данными не найден.
        - ValueError: Если после загрузки получен пустой DataFrame.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {file_path}")

    dataframe = pd.read_excel(file_path, engine="openpyxl")

    if dataframe.empty:
        raise ValueError(f"Файл загружен, но DataFrame пустой: {file_path}")

    return dataframe


def get_dataset_overview(dataframe: pd.DataFrame) -> dict[str, int]:
    """
    Получает базовую информацию о размере датасета.

    Аргументы:
        - dataframe: Исходный DataFrame.

    Возвращает:
        - Словарь с количеством строк, столбцов, пропущенных значений и дубликатов.

    Пример использования:
        poetry run python -c "from sessions_tasks.classical_ml.session02.coursework_classical_ml.src.data_loader \
        import load_coursework_dataset, get_dataset_overview; df = load_coursework_dataset(); print(df.shape); \
        print(get_dataset_overview(df)); print(df.head())"
    """
    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicated_rows": int(dataframe.duplicated().sum()),
    }
