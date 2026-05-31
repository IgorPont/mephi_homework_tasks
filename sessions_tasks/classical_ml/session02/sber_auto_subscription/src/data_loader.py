"""
Модуль загрузки исходных данных проекта.

Данные представлены в двух форматах:
    - pickle (.pkl) -> основной формат для работы
    - csv (.csv) -> резервный формат

Сначала читаем pickle, так как он быстрее загружается
и лучше сохраняет структуру pandas DataFrame.
В качестве резерва pickle-файлов используем CSV.

Пример загрузки таблиц проекта с отображением времени загрузки:
    poetry run python -u -c "
    from time import perf_counter
    from src.data_loader import load_sessions, load_hits

    start = perf_counter()
    print('Загружаю sessions...', flush=True)
    sessions = load_sessions()
    print('sessions loaded:', sessions.shape, 'time:', round(perf_counter() - start, 2), 'sec', flush=True)

    start = perf_counter()
    print('Загружаю hits...', flush=True)
    hits = load_hits()
    print('hits loaded:', hits.shape, 'time:', round(perf_counter() - start, 2), 'sec', flush=True)"

Пример выполнения команды:
    Загружаю sessions...
    sessions loaded: (1860042, 18) time: 19.35 sec
    Загружаю hits...
    hits loaded: (15726470, 11) time: 98.51 sec
"""

from pathlib import Path

import pandas as pd

from sessions_tasks.classical_ml.session02.sber_auto_subscription.src.config import (
    GA_HITS_CSV_PATH,
    GA_HITS_PKL_PATH,
    GA_SESSIONS_CSV_PATH,
    GA_SESSIONS_PKL_PATH,
)


def load_dataframe(
    pkl_path: Path,
    csv_path: Path,
) -> pd.DataFrame:
    """
    Загружает DataFrame из pickle или CSV.

    Сначала выполняется попытка загрузки из pickle-файла.
    Если pickle-файл отсутствует, данные загружаются из CSV.

    Аргументы:
        - pkl_path: Путь к pickle-файлу
        - csv_path: Путь к CSV-файлу

    Возвращает:
        - Загруженный DataFrame

    Исключения:
        - FileNotFoundError -> если не найден ни pickle-файл, ни CSV-файл
    """

    if pkl_path.exists():
        return pd.read_pickle(pkl_path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "Не найден файл с данными. "
        f"Проверены пути: {pkl_path} и {csv_path}"
    )


def load_sessions() -> pd.DataFrame:
    """
    Загружает данные о визитах пользователей на сайт.

    Возвращает:
        - DataFrame с данными из ga_sessions
    """

    return load_dataframe(
        pkl_path=GA_SESSIONS_PKL_PATH,
        csv_path=GA_SESSIONS_CSV_PATH,
    )


def load_hits() -> pd.DataFrame:
    """
    Загружает данные о событиях пользователей на сайте.

    Возвращает:
        - DataFrame с данными из ga_hits
    """

    return load_dataframe(
        pkl_path=GA_HITS_PKL_PATH,
        csv_path=GA_HITS_CSV_PATH,
    )


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает обе исходные таблицы проекта.

    Возвращает:
        Кортеж из двух DataFrame:
            - sessions: данные о визитах
            - hits: данные о событиях
    """

    sessions = load_sessions()
    hits = load_hits()

    return sessions, hits
