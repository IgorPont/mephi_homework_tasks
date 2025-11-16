"""
Модуль с решениями заданий модуля 5 (Pandas, продвинутый уровень)
для датасета ratings_movies.csv.

Сейчас реализовано решение задачи 8.1:
    - добавить признак year_release;
    - посчитать, для скольких строк год выпуска не указан.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# --- Константы путей ---------------------------------------------------------

# Директория, в которой лежит текущий файл analysis.py
BASE_DIR: Path = Path(__file__).resolve().parent

# Файлы для блока про фильмы
DATA_FILE: Path = BASE_DIR / "data" / "ratings_movies.csv"

# Файлы для блока про интернет-магазин
ORDERS_FILE: Path = BASE_DIR / "data" / "orders.csv"
PRODUCTS_FILE: Path = BASE_DIR / "data" / "products.csv"


# --- Загрузка данных ---------------------------------------------------------

def load_ratings_movies(path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Загрузить таблицу ratings_movies из CSV-файла.

    Parameters
    ----------
    path : Path
        Путь к CSV-файлу с данными.

    Returns
    -------
    pd.DataFrame
        Таблица с исходными данными.
    """
    # read_csv читает CSV в DataFrame; sep по умолчанию ',', нам подходит
    df = pd.read_csv(path)
    return df


def load_orders(path: Path = ORDERS_FILE) -> pd.DataFrame:
    """
    Загрузить таблицу orders из CSV-файла.

    Parameters
    ----------
    path : Path
        Путь к файлу orders.csv.

    Returns
    -------
    pd.DataFrame
        Таблица с данными о заказах.
    """
    # В файле orders.csv разделитель — ';'
    df = pd.read_csv(path, sep=";")
    return df


def load_products(path: Path = PRODUCTS_FILE) -> pd.DataFrame:
    """
    Загрузить таблицу products из CSV-файла.

    Parameters
    ----------
    path : Path
        Путь к файлу products.csv.

    Returns
    -------
    pd.DataFrame
        Таблица с данными о товарах.
    """
    # В файле products.csv разделитель — ';'
    df = pd.read_csv(path, sep=";")
    return df


# --- Вспомогательная функция для извлечения года выпуска ---------------------

def get_year_release(title: str) -> int | None:
    """
    Извлечь год выпуска фильма из строки с названием.

    Ожидаемый формат года – четыре цифры в круглых скобках,
    например: "Toy Story (1995)".

    Если шаблон не найден, возвращается None (в DataFrame это станет NaN).

    Parameters
    ----------
    title : str
        Название фильма (значение столбца `title`).

    Returns
    -------
    int | None
        Год выпуска фильма или None, если год не найден.
    """
    # На всякий случай обрабатываем некорректные типы (NaN и т.п.)
    if not isinstance(title, str):
        return None

    # Ищем все вхождения шаблона "(YYYY)" — четыре цифры в скобках
    candidates = re.findall(r"\((\d{4})\)", title)

    # Если ничего не нашли — возвращаем None
    if not candidates:
        return None

    # Берем последнее найденное значение на случай,
    # если в строке несколько групп цифр
    year_str = candidates[-1]

    # Пробуем привести к целому числу;
    # при ошибке тоже возвращаем None
    try:
        return int(year_str)
    except ValueError:
        return None


def load_orders_products(
        orders_path: Path = ORDERS_FILE,
        products_path: Path = PRODUCTS_FILE,
) -> pd.DataFrame:
    """
    Загрузить таблицы orders и products и объединить их
    в одну таблицу orders_products.

    Входные файлы:
    - orders.csv   — разделитель ';', ключевой столбец "ID товара"
    - products.csv — разделитель ';', ключевой столбец "Product_ID"

    Используем левое соединение, чтобы сохранить все заказы,
    даже если по какому-то заказу нет информации о товаре.
    """
    # Читаем исходные таблицы (разделитель — ';')
    orders = pd.read_csv(orders_path, sep=";")
    products = pd.read_csv(products_path, sep=";")

    # Объединяем: для каждого заказа подтягиваем информацию о товаре
    orders_products = orders.merge(
        products,
        how="left",  # все заказы должны остаться
        left_on="ID товара",  # поле с ID товара в orders.csv
        right_on="Product_ID",  # поле с ID товара в products.csv
    )

    return orders_products


# --- Задача 8.1 --------------------------------------------------------------

def task_8_1(df: pd.DataFrame) -> int:
    """
    Добавить в таблицу признак `year_release` и посчитать,
    для скольких строк год выпуска не указан.

    По условиям задания Stepik ожидает ответ именно
    в виде количества СТРОК, а не количества уникальных фильмов.

    Parameters
    ----------
    df : pd.DataFrame
        Исходная таблица ratings_movies.

    Returns
    -------
    int
        Количество строк, в которых год выпуска отсутствует.
    """
    # 1. Строим новый столбец `year_release`, извлекая год из `title`
    df["year_release"] = df["title"].apply(get_year_release)

    # 2. Считаем количество строк, где год отсутствует (NaN)
    missing_rows_count: int = int(df["year_release"].isna().sum())

    # Дополнительно: считаем количество уникальных фильмов без года,
    # это не требуется заданием, но полезно для самопроверки.
    unique_missing_movies: int = int(
        df.loc[df["year_release"].isna(), "movieId"].nunique()
    )

    # Логируем подробную информацию, чтобы было понятно, что происходит
    print("[8.1] Количество строк с отсутствующим годом выпуска:", missing_rows_count)
    print(
        "[8.1] Количество фильмов (уникальных movieId) без года выпуска:",
        unique_missing_movies,
    )

    # Возвращаем именно число строк — этого требует система проверки
    return missing_rows_count


def worst_film_1999(df):
    """
    Возвращает фильм, выпущенный в 1999 году,
    который получил наименьшую среднюю оценку зрителей.

    df: DataFrame с колонками ["rating", "title", "year_release"]
    """

    # Фильтруем фильмы 1999 года
    df_1999 = df[df["year_release"] == 1999].copy()

    # Группируем по названию и считаем среднюю оценку
    film_mean_rating = df_1999.groupby("title")["rating"].mean()

    # Получаем фильм с минимальной средней оценкой
    worst_film_title = film_mean_rating.idxmin()
    worst_film_rating = film_mean_rating.min()

    return worst_film_title, worst_film_rating


def task_8_3(df: pd.DataFrame) -> str:
    """
    Найти сочетание жанров (genres) фильмов 2010 года,
    которое получило наименьшую среднюю оценку зрителей.

    Возвращает строку в формате EXACT, как в таблице (через |, без пробелов).

    Parameters
    ----------
    df : pd.DataFrame
        Таблица ratings_movies с колонками:
        ["rating", "genres", "year_release"].

    Returns
    -------
    str
        Сочетание жанров с минимальной средней оценкой.
    """

    # Фильмы 2010 года
    df_2010 = df[df["year_release"] == 2010].copy()

    # Группируем по жанрам и считаем средний рейтинг
    mean_by_genres = df_2010.groupby("genres")["rating"].mean()

    # Находим жанры с минимальной средней оценкой
    worst_genres = mean_by_genres.idxmin()
    worst_rating = mean_by_genres.min()

    # Логируем для наглядности
    print("[8.3] Худшее сочетание жанров 2010 года:", worst_genres)
    print("[8.3] Средняя оценка:", worst_rating)

    return worst_genres


def task_8_4(df: pd.DataFrame) -> int:
    """
    Определить, какой пользователь (userId) посмотрел
    наибольшее количество уникальных комбинаций жанров (genres).

    Считаются уникальные строки в колонке `genres` для каждого userId.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица ratings_movies.

    Returns
    -------
    int
        userId пользователя, который посмотрел максимальное число уникальных жанровых сочетаний.
    """

    # Группировка: по каждому userId считаем число уникальных genres
    unique_genres_per_user = df.groupby("userId")["genres"].nunique()

    # Идентификатор пользователя с максимальным количеством уникальных жанров
    top_user = int(unique_genres_per_user.idxmax())
    top_count = int(unique_genres_per_user.max())

    # Логируем дополнительную информацию
    print("[8.4] Максимальное число уникальных сочетаний жанров:", top_count)
    print("[8.4] Пользователь с максимальным разнообразием жанров:", top_user)

    return top_user


def task_8_5(df: pd.DataFrame) -> int:
    """
    Найти пользователя, который выставил НАИМЕНЬШЕЕ количество оценок,
    но при этом имеет НАИБОЛЬШУЮ среднюю оценку среди таких пользователей.

    Логика:
    1) Группируем по userId
    2) Считаем:
       - число оценок (count)
       - среднюю оценку (mean)
    3) Находим минимальное число оценок
    4) Среди пользователей с этим минимальным числом выбираем того,
       у кого средняя оценка максимальна.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    int
        userId пользователя, удовлетворяющего условию.
    """

    # Группировка и расчет параметров
    stats = (
        df.groupby("userId")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "ratings_count", "mean": "ratings_mean"})
    )

    # Минимальное количество выставленных оценок
    min_count = stats["ratings_count"].min()

    # Пользователи, у которых именно такое минимальное число оценок
    subset = stats[stats["ratings_count"] == min_count]

    # Среди них находим пользователя с максимальной средней оценкой
    best_user = int(subset["ratings_mean"].idxmax())
    best_mean = float(subset["ratings_mean"].max())

    # Логируем для удобства
    print("[8.5] Минимальное количество оценок:", min_count)
    print("[8.5] Лучший пользователь среди них:", best_user)
    print("[8.5] Средняя оценка этого пользователя:", best_mean)

    return best_user


def task_8_6(df: pd.DataFrame) -> str:
    """
    Найти сочетание жанров (genres) среди фильмов 2018 года,
    которое имеет:
        • наибольший средний рейтинг (mean rating),
        • и при этом количество оценок > 10.

    Возвращать строку вида 'Action|Adventure' — без пробелов.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        Сочетание жанров, удовлетворяющее условиям.
    """

    # Фильмы 2018 года
    df_2018 = df[df["year_release"] == 2018].copy()

    # Группировка по жанрам с расчетом среднего рейтинга и количества оценок
    genre_stats = (
        df_2018.groupby("genres")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rating_mean", "count": "rating_count"})
    )

    # Оставляем только те жанровые сочетания, где count > 10
    filtered = genre_stats[genre_stats["rating_count"] > 10]

    # Находим строку с максимальным средним рейтингом
    best_genres = filtered["rating_mean"].idxmax()
    best_rating = filtered["rating_mean"].max()

    # Логируем
    print("[8.6] Лучшие жанры 2018 года:", best_genres)
    print("[8.6] Средний рейтинг:", best_rating)

    return best_genres


def task_8_7(df: pd.DataFrame) -> pd.DataFrame:
    """
    Задание 8.7.
    Добавить признак year_rating — год выставления оценки.
    Построить сводную таблицу зависимости среднего рейтинга
    от года выставления оценки и сочетания жанров.

    Дополнительно функция выводит подсказки по вариантам A–D
    (чтобы не считать их руками в Excel/ноутбуке).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Сводная таблица: index = year_rating, columns = genres,
        values = mean(rating).
    """

    # Год выставления оценки берем из столбца date
    # Преобразуем к datetime и сразу извлекаем год
    df["year_rating"] = pd.to_datetime(df["date"]).dt.year

    # Сводная таблица: средний рейтинг по годам и жанрам
    pivot = df.pivot_table(
        index="year_rating",
        columns="genres",
        values="rating",
        aggfunc="mean"
    )

    print("\n[8.7] Сводная таблица (фрагмент):")
    print(pivot.head())

    # --- Подсказки по вариантам ответов A–D -------------------------------

    # A: За весь период сочетание жанров "Action|Adventure"
    #    ни разу не получало среднюю оценку ниже 3
    col_a = "Action|Adventure"
    if col_a in pivot.columns:
        min_action_adv = pivot[col_a].min()
        print(f"[8.7][A] Минимальный средний рейтинг для {col_a}: {min_action_adv:.3f}")
    else:
        print(f"[8.7][A] Колонка {col_a!r} отсутствует в таблице.")

    # B: Наилучшую оценку жанр
    #    "Action|Adventure|Animation|Children|Comedy|IMAX" получил в 2010 году
    col_b = "Action|Adventure|Animation|Children|Comedy|IMAX"
    if col_b in pivot.columns:
        ser_b = pivot[col_b].dropna()
        if not ser_b.empty:
            best_year_b = int(ser_b.idxmax())
            best_rating_b = ser_b.max()
            print(
                f"[8.7][B] Лучший год для {col_b}: {best_year_b}, "
                f"средний рейтинг = {best_rating_b:.3f}"
            )
        else:
            print(f"[8.7][B] Для {col_b} нет значений рейтинга.")
    else:
        print(f"[8.7][B] Колонка {col_b!r} отсутствует в таблице.")

    # C: Среди сочетаний жанров, получивших наивысшую среднюю оценку в 2018 году,
    #    есть сочетание "Animation|Children|Mystery"
    year_c = 2018
    col_c = "Animation|Children|Mystery"
    if year_c in pivot.index:
        row_2018 = pivot.loc[year_c]
        max_2018 = row_2018.max()
        best_genres_2018 = row_2018[row_2018 == max_2018].dropna().index.tolist()
        has_c = col_c in best_genres_2018
        print(f"[8.7][C] Жанры с максимальной средней оценкой в {year_c}:")
        print("        ", best_genres_2018)
        print(f"[8.7][C] Присутствует ли {col_c}? -> {has_c}")
    else:
        print(f"[8.7][C] В сводной таблице нет строки для {year_c} года.")

    # D: Для жанра "Comedy" прослеживается тенденция падения рейтинга
    #    с каждым годом (1996–2018).
    col_d = "Comedy"
    if col_d in pivot.columns:
        ser_d = pivot[col_d].dropna()
        is_decreasing = ser_d.is_monotonic_decreasing
        print(f"[8.7][D] Значения средней оценки для {col_d}:")
        print(ser_d)
        print(f"[8.7][D] Монотонно убывает? -> {is_decreasing}")
    else:
        print(f"[8.7][D] Колонка {col_d!r} отсутствует в таблице.")

    return pivot


def task_8_8(orders_products: pd.DataFrame) -> int:
    """
    Найти идентификатор заказа (Order ID), для которого в объединенной
    таблице orders_products нет информации о товаре.

    Информацию о товаре считаем отсутствующей, если в колонке `Name`
    (название товара из products.csv) стоит NaN.
    """
    # Ищем строки, где название товара не подтянулось —
    # это значит, что Product_ID из products.csv для этого заказа не найден
    mask_missing_product = orders_products["Name"].isna()

    # Берем уникальные идентификаторы заказов, для которых нет товара
    missing_order_ids = orders_products.loc[mask_missing_product, "Order ID"].unique()

    # По условию задачи такой заказ один; приводим к int
    missing_order_id = int(missing_order_ids[0])

    print("[8.8] Идентификатор заказа без информации о товаре:", missing_order_id)
    return missing_order_id


def task_8_9(orders_products: pd.DataFrame) -> str:
    """
    Найти товар, по которому была произведена отмена.

    Отмена фиксируется в колонке 'Отменен' значением 'Да'.
    Возвращает название товара (Name).
    """

    # Фильтруем отмененные заказы
    canceled = orders_products[orders_products["Отменен"] == "Да"]

    # Извлекаем название товара
    product_name = canceled["Name"].iloc[0]

    print("[8.9] Товар, по которому была произведена отмена:", product_name)
    return product_name


def task_8_10(orders_products: pd.DataFrame) -> int:
    """
    Найти покупателя (ID Покупателя), который принес наибольшую суммарную прибыль.

    Прибыль считается ТОЛЬКО из оплаченных заказов:
    прибыль = Количество * Price
    """
    # Берем только оплаченные заказы
    paid = orders_products[orders_products["Оплачен"] == "Да"].copy()

    # Вычисляем прибыль
    paid["profit"] = paid["Количество"] * paid["Price"]

    # Группируем по ID Покупателя и считаем суммарную прибыль
    profit_by_customer = paid.groupby("ID Покупателя")["profit"].sum()

    # Идентификатор покупателя с максимальной прибылью
    best_customer = int(profit_by_customer.idxmax())
    best_profit = float(profit_by_customer.max())

    print("[8.10] Покупатель с максимальной прибылью:", best_customer)
    print("[8.10] Его прибыль:", best_profit)

    return best_customer


# --- Точка входа для ручного запуска модуля ---------------------------------
if __name__ == "__main__":
    # Загружаем таблицу
    ratings_movies_df = load_ratings_movies()

    # === ЗАДАНИЕ 8.1 ===
    result_8_1 = task_8_1(ratings_movies_df)
    # Итоговый ответ, который нужно ввести в форму на Stepik
    print("[8.1] Итоговый ответ (число строк без года выпуска):", result_8_1)

    # === ЗАДАНИЕ 8.2 ===
    film_1999_title, film_1999_rating = worst_film_1999(ratings_movies_df)
    print(f"[8.2] Худший фильм 1999 года: {film_1999_title}")
    print(f"[8.2] Средняя оценка: {film_1999_rating}")

    # === ЗАДАНИЕ 8.3 ===
    result_8_3 = task_8_3(ratings_movies_df)
    print("[8.3] Итоговый ответ (жанры):", result_8_3)

    # === ЗАДАНИЕ 8.4 ===
    result_8_4 = task_8_4(ratings_movies_df)
    print("[8.4] Итоговый ответ (userId):", result_8_4)

    # === ЗАДАНИЕ 8.5 ===
    result_8_5 = task_8_5(ratings_movies_df)
    print("[8.5] Итоговый ответ (userId):", result_8_5)

    # === ЗАДАНИЕ 8.6 ===
    result_8_6 = task_8_6(ratings_movies_df)
    print("[8.6] Итоговый ответ (genres):", result_8_6)

    # === ЗАДАНИЕ 8.7 ===
    pivot_8_7 = task_8_7(ratings_movies_df)
    # Если нужно, можно сохранить сводную таблицу в CSV/Excel:
    # pivot_8_7.to_csv(BASE_DIR / "pivot_8_7.csv")

    # === ЗАДАНИЕ 8.8 ===
    orders_products_df = load_orders_products()
    result_8_8 = task_8_8(orders_products_df)
    print("[8.8] Итоговый ответ (Order ID):", result_8_8)

    # === ЗАДАНИЕ 8.9 ===
    result_8_9 = task_8_9(orders_products_df)
    print("[8.9] Итоговый ответ (Name):", result_8_9)

    # === ЗАДАНИЕ 8.10 ===
    result_8_10 = task_8_10(orders_products_df)
    print("[8.10] Итоговый ответ (ID Покупателя):", result_8_10)
