"""
Схемы и валидация входных данных для API.

Сервис ожидает JSON с признаками клиента кредитной карты.
Названия признаков должны совпадать с названиями столбцов,
которые использовались при обучении модели.
"""

# Список признаков, которые модель ожидает на вход
# Столбец ID и целевая переменная default.payment.next.month не используются.
FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def validate_features(data: dict) -> dict:
    """
    Проверяет входной JSON и возвращает признаки в правильном порядке.

    Аргументы:
        - data: JSON-словарь с признаками клиента

    Возвращает:
        - dict: валидированный словарь с признаками

    Исключения:
        - ValueError: если отсутствуют обязательные признаки
            или значения невозможно привести к числам.
    """

    if not isinstance(data, dict):
        raise ValueError("Тело запроса должно быть JSON-объектом")

    missing_features = [
        feature for feature in FEATURE_COLUMNS
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            "Отсутствуют обязательные признаки: "
            + ", ".join(missing_features)
        )

    validated_data = {}

    for feature in FEATURE_COLUMNS:
        value = data[feature]

        try:
            validated_data[feature] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Признак {feature} должен быть числовым значением"
            ) from exc

    return validated_data
