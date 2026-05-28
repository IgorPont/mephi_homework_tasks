"""
Конфигурация путей и основных параметров курсовой работы.

Модуль содержит единые настройки, которые используются в EDA,
экспериментах по регрессии и экспериментах по классификации.
"""

from pathlib import Path

# Корневая директория курсовой работы:
# sessions_tasks/classical_ml/session02/coursework_classical_ml/
COURSEWORK_DIR = Path(__file__).resolve().parents[1]

# Основные директории проекта
DATA_DIR = COURSEWORK_DIR / "data"
NOTEBOOKS_DIR = COURSEWORK_DIR / "notebooks"
REPORTS_DIR = COURSEWORK_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
OUTPUTS_DIR = COURSEWORK_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Исходный файл с данными
DATA_FILE_NAME = "Данные_для_курсовои_Классическое_МО.xlsx"
DATA_FILE_PATH = DATA_DIR / DATA_FILE_NAME

# Целевые признаки
TARGET_IC50 = "IC50, mM"
TARGET_CC50 = "CC50, mM"
TARGET_SI = "SI"

REGRESSION_TARGETS = [
    TARGET_IC50,
    TARGET_CC50,
    TARGET_SI,
]

# Фиксируем random_state для воспроизводимости экспериментов
RANDOM_STATE = 42

# Размер тестовой выборки
TEST_SIZE = 0.2

# Количество фолдов для кросс-валидации (число равных частей, на которые разбивается исходный набор)
CV_FOLDS = 5

# Служебные столбцы, которые не должны использоваться как признаки
ID_COLUMNS = [
    "Unnamed: 0",
]

# Все столбцы, которые не должны попадать в матрицу признаков
NON_FEATURE_COLUMNS = ID_COLUMNS + REGRESSION_TARGETS

# Названия задач регрессии
REGRESSION_TASK_NAMES = {
    TARGET_IC50: "regression_ic50",
    TARGET_CC50: "regression_cc50",
    TARGET_SI: "regression_si",
}

# Названия задач классификации
CLASSIFICATION_TASK_NAMES = {
    "ic50_above_median": "classification_ic50_above_median",
    "cc50_above_median": "classification_cc50_above_median",
    "si_above_median": "classification_si_above_median",
    "si_above_8": "classification_si_above_8",
}
