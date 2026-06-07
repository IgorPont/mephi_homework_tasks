"""
Конфигурация ML-сервиса.

В этом модуле хранятся основные пути к моделям и настройки API.
"""

from pathlib import Path

# Корневая директория проекта:
# default_credit_card_clients/
BASE_DIR = Path(__file__).resolve().parent.parent

# Директория с обученными моделями
MODELS_DIR = BASE_DIR / "models"

# Пути к моделям для A/B-тестирования
MODEL_V1_PATH = MODELS_DIR / "credit_default_model_v1.joblib"
MODEL_V2_PATH = MODELS_DIR / "credit_default_model_v2.joblib"

# Версия модели по умолчанию
# Так как RandomForest показал качество выше, используем v2 как основную модель
DEFAULT_MODEL_VERSION = "v2"

# Доступные версии моделей
AVAILABLE_MODEL_VERSIONS = {
    "v1": MODEL_V1_PATH,
    "v2": MODEL_V2_PATH,
}

# Настройки Flask-приложения
API_HOST = "0.0.0.0"
API_PORT = 5001
