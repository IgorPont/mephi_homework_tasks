"""
Модуль генерации признаков для проекта анализа сайта "СберАвтоподписка".

В этом файле собраны функции, которые преобразуют исходные данные
о визитах и событиях пользователей в признаки для модели машинного обучения.

Основная идея:
    - признаки на уровне визита берем из ga_sessions
    - поведенческие признаки агрегируем из ga_hits по session_id
    - итоговый датасет формируется на уровне одного визита
"""

import numpy as np
import pandas as pd

ORGANIC_MEDIUMS = ["organic", "referral", "(none)"]

SOCIAL_SOURCES = [
    "QxAxdyPLuQMEcrdZWdWb",
    "MvfHsxITijuriZxsqZqt",
    "ISrKoXQCxqqYvAZICvjs",
    "IZEXUFLARCUMynmHNBGo",
    "PlbkrSYoHuZBWfYjYnfw",
    "gVRrcxiDQubJiljoTbGm",
]

BEHAVIOR_ACTION_GROUPS = {
    "has_view_card": [
        "view_card",
        "view_new_card",
        "view_used_card",
        "go_to_car_card",
        "greenday_go_to_car_card",
    ],
    "has_search": [
        "search_form_search_btn",
        "search_form_region",
        "search_form_cost_from",
        "search_form_cost_to",
        "search_form_mark_select",
        "search_form_model_select",
        "sap_search_form_cost_from",
        "sap_search_form_cost_to",
    ],
    "has_pagination": [
        "pagination_click",
    ],
    "has_quiz": [
        "quiz_show",
        "quiz_start",
    ],
    "has_subscription_interest": [
        "auto_subscription_click",
        "click_auto_subscription",
        "click_on_subscription",
        "sub_landing",
        "sub_car_page",
        "sub_banner_click",
        "sub_view_cars_click",
        "sub_view_faq_click",
        "sub_offer_click",
    ],
    "has_phone_interaction": [
        "sub_call_number_click",
        "tap_on_phone_495",
        "tap_on_phone_800",
        "mobile call",
        "click_on_phone",
    ],
    "has_form_interaction": [
        "showed_form_request_call",
        "form_request_call_sent",
        "phone_entered",
        "phone_entered_on_form_request_call",
        "name_entered",
        "name_entered_on_form_request_call",
        "show_form_lets_get_acquainted",
    ],
}


def prepare_sessions_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Создает признаки на основе таблицы визитов ga_sessions.

    Что создается:
        - календарные признаки из visit_date и visit_time
        - бинарные признаки типа трафика
        - признак первого визита пользователя
        - ограниченная версия visit_number

    Аргументы:
        - sessions: DataFrame с исходными визитами пользователей

    Возвращает:
        - DataFrame с базовыми признаками на уровне session_id
    """

    sessions_features = sessions.copy()

    sessions_features["visit_date"] = pd.to_datetime(
        sessions_features["visit_date"],
        errors="coerce",
    )

    sessions_features["visit_time"] = pd.to_datetime(
        sessions_features["visit_time"],
        format="%H:%M:%S",
        errors="coerce",
    )

    sessions_features["visit_month"] = sessions_features["visit_date"].dt.month
    sessions_features["visit_day"] = sessions_features["visit_date"].dt.day
    sessions_features["visit_weekday"] = sessions_features["visit_date"].dt.weekday
    sessions_features["visit_hour"] = sessions_features["visit_time"].dt.hour

    sessions_features["is_weekend"] = (
        sessions_features["visit_weekday"]
        .isin([5, 6])
        .astype(int)
    )

    sessions_features["is_organic_traffic"] = (
        sessions_features["utm_medium"]
        .isin(ORGANIC_MEDIUMS)
        .astype(int)
    )

    sessions_features["is_paid_traffic"] = (
        ~sessions_features["utm_medium"]
        .isin(ORGANIC_MEDIUMS)
    ).astype(int)

    sessions_features["is_social_traffic"] = (
        sessions_features["utm_source"]
        .isin(SOCIAL_SOURCES)
        .astype(int)
    )

    sessions_features["is_first_visit"] = (
        sessions_features["visit_number"]
        .eq(1)
        .astype(int)
    )

    sessions_features["visit_number_clipped"] = (
        sessions_features["visit_number"]
        .clip(upper=10)
    )

    columns_to_drop = [
        "session_id",
        "client_id",
        "visit_date",
        "visit_time",
    ]

    existing_columns_to_drop = [
        column
        for column in columns_to_drop
        if column in sessions_features.columns
    ]

    return sessions_features.drop(columns=existing_columns_to_drop)


def create_hits_session_features(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Создает агрегированные признаки по событиям пользователей.

    Все признаки считаются на уровне session_id.

    Что создается:
        - количество событий в визите
        - максимальный номер события
        - количество уникальных страниц
        - количество уникальных event_action
        - количество уникальных event_category

    Аргументы:
        - hits: DataFrame с событиями пользователей

    Возвращает:
        - DataFrame с агрегированными признаками по session_id
    """

    hits_features = (
        hits
        .groupby("session_id", as_index=False)
        .agg(
            hit_count=("hit_number", "count"),
            max_hit_number=("hit_number", "max"),
            unique_page_count=("hit_page_path", "nunique"),
            unique_event_action_count=("event_action", "nunique"),
            unique_event_category_count=("event_category", "nunique"),
        )
    )

    return hits_features


def create_behavior_features(
    hits: pd.DataFrame,
    action_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Создает бинарные поведенческие признаки на уровне визита.

    Для каждой группы действий проверяется, было ли в рамках session_id
    хотя бы одно событие из заданного списка.

    Аргументы:
        - hits: DataFrame с событиями пользователей
        - action_groups (словарь вида):
            {
                "название_признака": ["event_action_1", "event_action_2"]
            }

    Возвращает:
        - DataFrame с session_id и бинарными поведенческими признаками
    """

    if action_groups is None:
        action_groups = BEHAVIOR_ACTION_GROUPS

    behavior_features = hits[["session_id"]].drop_duplicates().copy()

    for feature_name, actions in action_groups.items():
        sessions_with_action = (
            hits.loc[
                hits["event_action"].isin(actions),
                "session_id",
            ]
            .drop_duplicates()
        )

        behavior_features[feature_name] = (
            behavior_features["session_id"]
            .isin(sessions_with_action)
            .astype(int)
        )

    return behavior_features


def create_target(
    sessions: pd.DataFrame,
    hits: pd.DataFrame,
    target_actions: list[str],
    target_column: str = "target",
) -> pd.DataFrame:
    """
    Формирует целевую переменную на уровне визита.

    Визит получает target = 1, если в рамках session_id
    было совершено хотя бы одно действие из target_actions.

    Аргументы:
        - sessions: DataFrame с визитами пользователей
        - hits: DataFrame с событиями пользователей
        - target_actions: список целевых event_action
        - target_column: название целевой переменной

    Возвращает:
        - Копия sessions с добавленной колонкой target
    """

    sessions_with_target = sessions.copy()

    target_sessions = (
        hits.loc[
            hits["event_action"].isin(target_actions),
            "session_id",
        ]
        .drop_duplicates()
    )

    sessions_with_target[target_column] = (
        sessions_with_target["session_id"]
        .isin(target_sessions)
        .astype(int)
    )

    return sessions_with_target


def build_model_dataset(
    sessions: pd.DataFrame,
    hits: pd.DataFrame,
    target_actions: list[str],
    target_column: str = "target",
) -> pd.DataFrame:
    """
    Собирает итоговый датасет для обучения модели.

    Порядок действий:
        1. Формирует target на уровне визитов
        2. Создает признаки из ga_sessions
        3. Создает агрегированные признаки из ga_hits
        4. Создает бинарные поведенческие признаки из ga_hits
        5. Объединяет все признаки в один DataFrame
        6. Удаляет технические поля, которые не нужны модели

    Аргументы:
        - sessions: DataFrame с визитами пользователей
        - hits: DataFrame с событиями пользователей
        - target_actions: список целевых действий
        - target_column: название целевой переменной

    Возвращает:
        - Итоговый DataFrame для обучения модели
    """

    sessions_with_target = create_target(
        sessions=sessions,
        hits=hits,
        target_actions=target_actions,
        target_column=target_column,
    )

    session_ids = sessions_with_target[["session_id"]].copy()

    sessions_features = prepare_sessions_features(sessions_with_target)

    hits_session_features = create_hits_session_features(hits)
    behavior_features = create_behavior_features(hits)

    model_dataset = (
        session_ids
        .join(sessions_features)
        .merge(
            hits_session_features,
            on="session_id",
            how="left",
        )
        .merge(
            behavior_features,
            on="session_id",
            how="left",
        )
    )

    numeric_fill_columns = [
        "hit_count",
        "max_hit_number",
        "unique_page_count",
        "unique_event_action_count",
        "unique_event_category_count",
        *BEHAVIOR_ACTION_GROUPS.keys(),
    ]

    existing_numeric_fill_columns = [
        column
        for column in numeric_fill_columns
        if column in model_dataset.columns
    ]

    model_dataset[existing_numeric_fill_columns] = (
        model_dataset[existing_numeric_fill_columns]
        .fillna(0)
    )

    # Признаки, которые удалем перед обучением
    columns_to_drop = [
        # Технический идентификатор визита
        "session_id",

        # Почти полностью пустой признак,
        # по результатам EDA в нем около 99% пропусков
        "device_model",

        # Вспомогательный признак,
        # в скриптовой версии используются is_organic_traffic и is_paid_traffic
        "traffic_type",

        # Потенциальная утечка целевой переменной,
        # признаки отражают позднюю стадию заполнения формы или ввода телефона
        "has_form_interaction",
        "has_phone_interaction",
    ]

    existing_columns_to_drop = [
        column
        for column in columns_to_drop
        if column in model_dataset.columns
    ]

    final_dataset = model_dataset.drop(columns=existing_columns_to_drop)

    return final_dataset
