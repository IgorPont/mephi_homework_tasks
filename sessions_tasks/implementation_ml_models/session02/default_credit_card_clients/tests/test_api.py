"""
Тесты Flask API для сервиса прогнозирования дефолта.

Проверяем:
    1. Работоспособность health-check
    2. Корректный ответ predict-эндпоинта
    3. Ошибку при неполном JSON-запросе

Запуск из корня проекта default_credit_card_clients:
    pytest tests/

Запуск из корня репозитория:
    poetry run pytest sessions_tasks/implementation_ml_models/session02/default_credit_card_clients/tests/

Пример успешного результата:
    .....                                                                                                                                                [100%]
    5 passed in 2.16s
"""

from http import HTTPStatus

from app.api import app

VALID_CLIENT_PAYLOAD = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}


def test_health_endpoint_returns_healthy_status() -> None:
    """
    Проверяет, что /health возвращает статус healthy
    """

    client = app.test_client()

    response = client.get("/health")
    response_data = response.get_json()

    assert response.status_code == HTTPStatus.OK
    assert response_data["status"] == "healthy"
    assert response_data["service"] == "credit-default-prediction-api"
    assert response_data["default_model_version"] == "v2"


def test_predict_endpoint_returns_prediction() -> None:
    """
    Проверяет, что /predict возвращает прогноз и вероятность
    """

    client = app.test_client()

    response = client.post("/predict", json=VALID_CLIENT_PAYLOAD)
    response_data = response.get_json()

    assert response.status_code == HTTPStatus.OK
    assert response_data["model_version"] == "v2"
    assert response_data["prediction"] in [0, 1]
    assert 0 <= response_data["probability"] <= 1
    assert response_data["risk_label"] in ["default", "no_default"]


def test_predict_v1_endpoint_returns_v1_model_version() -> None:
    """
    Проверяет, что /predict/v1 использует модель v1
    """

    client = app.test_client()

    response = client.post("/predict/v1", json=VALID_CLIENT_PAYLOAD)
    response_data = response.get_json()

    assert response.status_code == HTTPStatus.OK
    assert response_data["model_version"] == "v1"


def test_predict_v2_endpoint_returns_v2_model_version() -> None:
    """
    Проверяет, что /predict/v2 использует модель v2
    """

    client = app.test_client()

    response = client.post("/predict/v2", json=VALID_CLIENT_PAYLOAD)
    response_data = response.get_json()

    assert response.status_code == HTTPStatus.OK
    assert response_data["model_version"] == "v2"


def test_predict_endpoint_returns_error_for_missing_features() -> None:
    """
    Проверяет, что API возвращает ошибку при отсутствии обязательных признаков
    """

    client = app.test_client()

    response = client.post(
        "/predict",
        json={
            "LIMIT_BAL": 20000,
            "SEX": 2,
        },
    )
    response_data = response.get_json()

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "error" in response_data
    assert "Отсутствуют обязательные признаки" in response_data["error"]
