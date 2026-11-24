"""
Конфигурация проекта прогнозирования среднемесячной температуры.

Модуль содержит пути к исходным и обработанным данным,
а также параметры подготовки временного ряда и моделирования.

Примечание:
    - RANDOM_STATE не нужен, так как данные не перемешиваются.
    - WINDOW_STEP в задании явно не определен, установлен равным 12 месяцам.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# === Директории проекта ===

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# === Исходные датасеты ===

COUNTRY_TEMPERATURES_PATH = (
    RAW_DATA_DIR / "GlobalLandTemperaturesByCountry.csv"
)

# === Настройки временного ряда ===

COUNTRY = "Belarus"

DATE_COLUMN = "dt"
TARGET_COLUMN = "AverageTemperature"

# Месячная частота с привязкой к началу месяца
FREQUENCY = "MS"

# === Настройки Holdout-разбиения ===

TEST_SIZE = 0.2

# === Настройки Rolling Window ===

# Размер обучающего окна в месяцах (10 лет)
TRAIN_WINDOW_SIZE = 120

# Горизонт прогнозирования в месяцах
FORECAST_HORIZON = 12

# Шаг смещения скользящего окна в месяцах
WINDOW_STEP = 12

# === Настройки генерации признаков ===

LAGS = (1, 12)

WINTER_MONTHS = (12, 1, 2)
