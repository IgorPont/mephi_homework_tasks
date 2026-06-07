"""
Слой работы с ML-моделями.

Модуль отвечает за:
    1. Загрузку сохраненных joblib-моделей
    2. Выбор версии модели
    3. Выполнение предсказания
"""

from dataclasses import dataclass

import joblib
import pandas as pd

from app.config import (
    AVAILABLE_MODEL_VERSIONS,
    DEFAULT_MODEL_VERSION
)
from app.schemas import FEATURE_COLUMNS


@dataclass
class PredictionResult:
    """
    Результат предсказания модели.

    Атрибуты:
        - prediction: класс прогноза (1 -> ожидается дефолт, 0 -> дефолт не ожидается)
        - probability: вероятность дефолта
        - model_version: версия использованной модели
    """

    prediction: int
    probability: float
    model_version: str


class ModelHandler:
    """
    Класс для загрузки моделей и выполнения инференса
    """

    def __init__(self) -> None:
        """
        Инициализирует обработчик моделей и загружает все доступные версии
        """
        self.models = self._load_models()

    def _load_models(self) -> dict:
        """
        Загружает все доступные модели из файлов.

        Возвращает:
            - dict: словарь вида {"v1": model_v1, "v2": model_v2}.

        Исключения:
            FileNotFoundError: если файл модели не найден
        """

        loaded_models = {}

        for version, model_path in AVAILABLE_MODEL_VERSIONS.items():
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Файл модели для версии {version} не найден: {model_path}"
                )

            loaded_models[version] = joblib.load(model_path)

        return loaded_models

    def predict(
        self,
        features: dict,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> PredictionResult:
        """
        Выполняет прогноз дефолта для одного клиента.

        Аргументы:
            - features: словарь с признаками клиента
            - model_version: версия модели: v1 или v2

        Возвращает:
            - PredictionResult: результат предсказания

        Исключения:
            - ValueError: если запрошена неизвестная версия модели
        """

        if model_version not in self.models:
            raise ValueError(
                f"Неизвестная версия модели: {model_version}. "
                f"Доступные версии: {', '.join(self.models.keys())}"
            )

        model = self.models[model_version]

        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS,
        )

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            model_version=model_version,
        )
