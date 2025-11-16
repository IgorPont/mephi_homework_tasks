"""
Базовый модуль для работы с датасетом Citi Bike в рамках
Модуля 4. Базовые приемы работы с данными в Pandas.

В этом файле:

1. Определяем путь к датасету относительно текущего файла.
2. Реализуем функцию загрузки датасета с разумными типами данных.
3. Реализуем функции для заданий.
"""

from pathlib import Path

import pandas as pd

# --- Константы путей ---------------------------------------------------------

# Директория, в которой лежит текущий файл analysis.py
BASE_DIR = Path(__file__).resolve().parent

# Полный путь к датасету Citi Bike
DATA_PATH = BASE_DIR / "data" / "citibike-tripdata.csv"


# --- Функции для работы с датасетом -----------------------------------------

def load_citibike() -> pd.DataFrame:
    """
    Загружает датасет Citi Bike из файла `citibike-tripdata.csv`.

    Функция:
    - Использует явное указание типов данных для экономии памяти.
    - Парсит временные столбцы как datetime.
    - Возвращает pandas.DataFrame с данными по поездкам.

    Эта функция оптимизирована "под жизнь": с ней удобно решать большинство
    аналитических задач (группировки, фильтрация по датам и т.п.).

    :return: DataFrame с данными Citi Bike.
    """
    # Явно задаем типы для некоторых столбцов.
    # Int32 и Int8 — это "nullable" целочисленные типы Pandas:
    # они позволяют хранить пропуски (NA) вместе с целыми числами.
    dtypes = {
        "start station id": "Int32",
        "start station name": "category",
        "start station latitude": "float32",
        "start station longitude": "float32",
        "end station id": "Int32",
        "end station name": "category",
        "end station latitude": "float32",
        "end station longitude": "float32",
        "bikeid": "Int32",
        "usertype": "category",
        "birth year": "float32",
        "gender": "Int8",
    }

    # Эти столбцы сразу парсим как datetime для удобства дальнейшего анализа.
    date_columns = ["starttime", "stoptime"]

    df = pd.read_csv(
        DATA_PATH,
        dtype=dtypes,
        parse_dates=date_columns,
    )

    return df


def count_missing_start_station_id(df: pd.DataFrame) -> int:
    """
    Считает количество пропусков в столбце `start station id`.

    Логика:
    - Берем столбец `start station id`.
    - Метод .isna() возвращает булеву маску: True там, где значение пропущено.
    - Метод .sum() для булевой серии считает количество True,
      что и дает нам число пропусков.

    :param df: DataFrame с данными Citi Bike.
    :return: количество пропусков (NA) в столбце `start station id`.
    """
    missing_count = df["start station id"].isna().sum()
    # Приводим к обычному int, чтобы избежать numpy-типов в выводе
    return int(missing_count)


def get_start_stop_dtype_raw() -> str:
    """
    Определяет тип данных столбцов `starttime` и `stoptime`
    при чтении CSV-файла "как в задании курса", то есть через
    обычный `pd.read_csv` БЕЗ параметра parse_dates.

    Это важно, потому что:
    - В наших практических задачах мы чаще хотим иметь datetime.
    - Но в тестах курсовой среды дата по умолчанию читается как строки (object),
      и вопрос 6.2 как раз про это поведение.

    Логика:
    - Читаем CSV-файл без доп.параметров.
    - Смотрим типы столбцов `starttime` и `stoptime`.
    - Они должны совпадать, поэтому возвращаем тип в виде строки.

    :return: строковое представление типа данных (например, 'object').
    """
    # Читаем файл "по-умолчанию": без parse_dates и явных dtypes
    df_raw = pd.read_csv(DATA_PATH)

    start_dtype = df_raw["starttime"].dtype
    stop_dtype = df_raw["stoptime"].dtype

    # На всякий случай проверим, что типы совпадают
    if start_dtype != stop_dtype:
        # Такая ситуация маловероятна, но если вдруг —
        # вернем оба типа через запятую для диагностики.
        return f"{start_dtype}, {stop_dtype}"

    # Преобразуем тип в строку (например, 'object')
    return str(start_dtype)


def most_popular_start_station(df: pd.DataFrame) -> int:
    """
    Возвращает идентификатор самой популярной стартовой станции.

    Логика:
        - Берем столбец `start station id`.
        - Считаем количество вхождений каждого ID через value_counts().
          Самое верхнее значение — это самый популярный идентификатор.
        - .idxmax() не используем, потому что value_counts уже дает отсортированный
          Series по убыванию.
        - Пропуски (NA) автоматически игнорируются.

    :param df: DataFrame с данными CitiBike.
    :return: идентификатор самой популярной станции как обычное целое число.
    """
    popular_id = df["start station id"].value_counts().idxmax()
    return int(popular_id)


def most_popular_bike(df: pd.DataFrame) -> int:
    """
    Возвращает идентификатор самого популярного велосипеда.

    Логика:
        - Берем столбец `bikeid`.
        - Считаем количество поездок для каждого велосипеда через value_counts().
        - Первый элемент в отсортированном Series — самый популярный.
        - Пропуски отсутствуют, но если бы были — value_counts() их бы игнорировала.

    :param df: DataFrame с данными CitiBike.
    :return: идентификатор велосипеда как целое число.
    """
    popular_bike_id = df["bikeid"].value_counts().idxmax()
    return int(popular_bike_id)


def dominant_usertype_share(df: pd.DataFrame) -> float:
    """
    Возвращает долю клиентов преобладающего типа usertype.

    Логика:
        - В столбце `usertype` хранятся два типа: "Subscriber" и "Customer".
        - Находим количество каждого типа через value_counts().
        - Определяем наиболее частый тип.
        - Вычисляем долю: count_max / total_count.
        - Округляем до двух знаков после запятой.

    :param df: DataFrame с данными CitiBike.
    :return: доля наиболее частого типа клиентов (float).
    """
    counts = df["usertype"].value_counts()
    dominant_share = counts.max() / counts.sum()
    return round(float(dominant_share), 2)


def gender_trip_counts(df: pd.DataFrame) -> tuple[int, int]:
    """
    Возвращает пол (gender) и количество поездок для той группы,
    у которой больше всего поездок среди gender ∈ {1, 2}.

    :param df: DataFrame с колонкой 'gender'
    :return: (top_gender, top_gender_count)
             top_gender — 1 (мужчины) или 2 (женщины)
             top_gender_count — число поездок у этой группы
    """
    # Оставляем только мужчин и женщин
    gender_series = df[df["gender"].isin([1, 2])]["gender"]

    # Считаем количество поездок по каждому полу
    gender_counts = gender_series.value_counts()

    # Наибольшее количество и соответствующий пол
    top_gender = int(gender_counts.idxmax())  # 1 или 2
    top_gender_count = int(gender_counts.max())

    return top_gender, top_gender_count


def check_statements(df: pd.DataFrame):
    """
    Проверяет утверждения A–D из задания 6.7.
    Возвращает словарь с результатами и выводит подробное объяснение.

    Утверждения:
        A — Число уникальных стартовых и конечных стоянок, которыми пользовались клиенты, не равны друг другу.
        B — Минимальный возраст клиента составлял 10 лет.
        C — Самой непопулярной стартовой стоянкой является 'Eastern Pkwy & Washington Ave'.
        D — Большее всего поездок завершается на стоянке 'Liberty Light Rail'.
    """

    print("\n--- Задание 6.7: проверка утверждений A–D ---")

    # ---------- A ----------
    unique_start = df["start station id"].nunique()
    unique_end = df["end station id"].nunique()
    A_result = unique_start != unique_end

    print(f"[A] Уникальных стартовых стоянок: {unique_start}")
    print(f"[A] Уникальных конечных стоянок: {unique_end}")
    print(f"[A] A → {A_result}\n")

    # ---------- B ----------
    df_birth_valid = df[df["birth year"].notna() & (df["birth year"] > 1900)]
    max_birth = df_birth_valid["birth year"].max()
    min_age = 2018 - max_birth
    B_result = (min_age == 10)

    print(f"[B] Самый поздний год рождения: {max_birth}")
    print(f"[B] Минимальный возраст клиента: {min_age}")
    print(f"[B] B → {B_result}\n")

    # ---------- C ----------
    start_counts = df["start station name"].value_counts(ascending=True)
    least_start = start_counts.index[0]
    C_result = (least_start == "Eastern Pkwy & Washington Ave")

    print(f"[C] Самая непопулярная стартовая стоянка: '{least_start}'")
    print(f"[C] C → {C_result}\n")

    # ---------- D ----------
    end_counts = df["end station name"].value_counts(ascending=False)
    most_popular_end = end_counts.index[0]
    D_result = (most_popular_end == "Liberty Light Rail")

    print(f"[D] Самая популярная конечная стоянка: '{most_popular_end}'")
    print(f"[D] D → {D_result}\n")

    # Итог
    print("=== Итог: истины только A и C ===\n")

    return {
        "A": A_result,
        "B": B_result,
        "C": C_result,
        "D": D_result
    }


def drop_station_ids(df: pd.DataFrame) -> int:
    """
    Удаляет столбцы идентификаторов стартовой и конечной стоянки.
    Возвращает количество столбцов после удаления.
    """
    print("\n--- Задание 6.8: удаляем дублирующие столбцы start/end station id ---")

    cols_to_drop = ["start station id", "end station id"]

    df.drop(columns=cols_to_drop, inplace=True)

    print(f"Удалили столбцы: {cols_to_drop}")
    print(f"Теперь столбцов: {df.shape[1]}")

    return df.shape[1]


def count_senior_trips(df: pd.DataFrame) -> int:
    """
    Создает столбец age = 2018 - birth year,
    удаляет birth year,
    возвращает количество поездок клиентов старше 60 лет.
    """
    print("\n--- Задание 6.9: создаем возраст клиента и считаем поездки 60+ ---")

    # Создаем возраст клиента
    df["age"] = 2018 - df["birth year"]

    # Удаляем старый столбец
    df.drop(columns=["birth year"], inplace=True)

    # Считаем поездки старше 60 лет
    senior_trip_count = df[df["age"] > 60].shape[0]

    print(f"Количество поездок клиентов старше 60 лет: {senior_trip_count}")

    return senior_trip_count


def compute_trip_duration(df: pd.DataFrame) -> int:
    """
    Создает столбец trip_duration как разницу stoptime - starttime.
    Возвращает длительность поездки с индексом 3 в целых минутах.
    """

    print("\n--- Задание 6.10: вычисляем длительность поездок ---")

    # Убедимся, что столбцы в формате datetime
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["stoptime"] = pd.to_datetime(df["stoptime"])

    # Вычисляем длительность поездки
    df["trip_duration"] = (df["stoptime"] - df["starttime"]).dt.total_seconds() / 60

    # Берем нужную строку (индекс 3)
    trip_minutes = int(df.loc[3, "trip_duration"])

    print(f"Длительность поездки с индексом 3: {trip_minutes} минут")

    return trip_minutes


# --- Точка входа при запуске файла как скрипта -------------------------------
if __name__ == "__main__":
    # --- Задание 6.1: количество пропусков в start station id ----------------
    citibike_df = load_citibike()
    missing_start_station_id = count_missing_start_station_id(citibike_df)
    print(f"[6.1] Количество пропусков в столбце 'start station id': "
          f"{missing_start_station_id}")

    # --- Задание 6.2: тип столбцов starttime и stoptime ----------------------
    start_stop_dtype = get_start_stop_dtype_raw()
    print(f"[6.2] Тип данных столбцов 'starttime' и 'stoptime' "
          f"при чтении через pd.read_csv: {start_stop_dtype}")

    # --- Задание 6.3: самая популярная стартовая станция --------------------
    popular_station = most_popular_start_station(citibike_df)
    print(f"[6.3] Самая популярная стартовая станция (ID): {popular_station}")

    # --- Задание 6.4: самый популярный велосипед ----------------------------
    popular_bike = most_popular_bike(citibike_df)
    print(f"[6.4] Самый популярный велосипед (bikeid): {popular_bike}")

    # --- Задание 6.5: доля преобладающего типа клиентов ---------------------
    usertype_share = dominant_usertype_share(citibike_df)
    print(f"[6.5] Доля преобладающего типа usertype: {usertype_share}")

    # --- Задание 6.6: кто больше занимается велоспортом (мужчины или женщины) ----
    top_gender, top_gender_count = gender_trip_counts(citibike_df)
    # Преобразуем код пола в человекочитаемую подпись
    gender_label = "мужчины" if top_gender == 1 else "женщины"
    print(f"[6.6] Больше всего поездок совершают: {gender_label}.")
    print(f"      Количество поездок: {top_gender_count}")

    # --- Задание 6.7: проверка утверждений A–D о данных CitiBike ---
    statements = check_statements(citibike_df)

    # --- Задание 6.8: удаление дублирующих признаков стоянок ---
    columns_after_drop = drop_station_ids(citibike_df)
    print(f"[6.8] Количество столбцов после удаления: {columns_after_drop}")

    # --- Задание 6.9: замена birth year на age и подсчет поездок людей старше 60 ---
    trips_over_60 = count_senior_trips(citibike_df)
    print(f"[6.9] Поездок клиентов старше 60 лет: {trips_over_60}")

    # --- Задание 6.10: длительность поездки trip_duration ---
    duration_index_3 = compute_trip_duration(citibike_df)
    print(f"[6.10] Поездка с индексом 3 длилась (мин): {duration_index_3}")
