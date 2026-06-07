"""
Flask API для сервиса прогнозирования дефолта по кредитной карте.

Эндпоинты:
    - GET /health -> проверка работоспособности сервиса
    - POST /predict -> прогноз моделью по умолчанию
    - POST /predict/v1  -> прогноз моделью v1
    - POST /predict/v2  -> прогноз моделью v2

Запуск из папки default_credit_card_clients:
    python app/api.py

Запуск из корня репозитория:
    poetry run python sessions_tasks/implementation_ml_models/session02/default_credit_card_clients/app/api.py
"""

import logging
from datetime import datetime, timezone
from http import HTTPStatus

from flask import Flask, jsonify, request

from app.config import (
    API_HOST, API_PORT,
    DEFAULT_MODEL_VERSION
)
from app.model_handler import ModelHandler
from app.schemas import validate_features

# Базовая настройка логирования
# В prod такие JSON-логи отправляем в ELK, Loki или др систему мониторинга
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загружаем модели один раз при старте приложения,
# чтобы не читать joblib-файлы при каждом запросе
model_handler = ModelHandler()


@app.get("/health")
def health():
    """
    Проверяет работоспособность сервиса.

    Возвращает:
        - tuple: JSON-ответ и HTTP-статус
    """

    return jsonify(
        {
            "status": "healthy",
            "service": "credit-default-prediction-api",
            "default_model_version": DEFAULT_MODEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ), HTTPStatus.OK


@app.post("/predict")
def predict_default():
    """
    Выполняет прогноз дефолта моделью по умолчанию.

    Возвращает:
        - tuple: JSON-ответ с прогнозом и HTTP-статус
    """

    return _predict(model_version=DEFAULT_MODEL_VERSION)


@app.post("/predict/v1")
def predict_v1():
    """
    Выполняет прогноз дефолта моделью v1.

    Используется как контрольная версия модели для A/B-теста.

    Возвращает:
        - tuple: JSON-ответ с прогнозом и HTTP-статус
    """

    return _predict(model_version="v1")


@app.post("/predict/v2")
def predict_v2():
    """
    Выполняет прогноз дефолта моделью v2.

    Используется как новая тестовая версия модели для A/B-теста.

    Возвращает:
        - tuple: JSON-ответ с прогнозом и HTTP-статус.
    """

    return _predict(model_version="v2")


def _predict(model_version: str):
    """
    Общая функция предсказания для всех predict-эндпоинтов.

    Аргументы:
        - model_version: версия модели для инференса

    Возвращает:
        - tuple: JSON-ответ и HTTP-статус
    """

    try:
        request_data = request.get_json(silent=True)

        if request_data is None:
            return jsonify(
                {
                    "error": "Некорректный запрос, ожидается JSON-объект с признаками клиента"
                }
            ), HTTPStatus.BAD_REQUEST

        features = validate_features(request_data)
        result = model_handler.predict(
            features=features,
            model_version=model_version,
        )

        response = {
            "prediction": result.prediction,
            "probability": round(result.probability, 6),
            "model_version": result.model_version,
            "risk_label": "default" if result.prediction == 1 else "no_default",
        }

        logger.info(
            {
                "event": "prediction_completed",
                "model_version": result.model_version,
                "prediction": result.prediction,
                "probability": result.probability,
            }
        )

        return jsonify(response), HTTPStatus.OK

    except ValueError as exc:
        logger.warning(
            {
                "event": "validation_error",
                "model_version": model_version,
                "error": str(exc),
            }
        )

        return jsonify(
            {
                "error": str(exc),
            }
        ), HTTPStatus.BAD_REQUEST

    except Exception as exc:
        logger.exception(
            {
                "event": "unexpected_error",
                "model_version": model_version,
                "error": str(exc),
            }
        )

        return jsonify(
            {
                "error": "Внутренняя ошибка сервиса."
            }
        ), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=False,
    )
