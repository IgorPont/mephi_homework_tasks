"""
Конфигурация проекта книжной рекомендательной системы.

Модуль содержит пути к исходным и обработанным данным,
моделям и результатам экспериментов.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# === Директории проекта ===

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# === Исходные датасеты ===

RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
BOOKS_PATH = RAW_DATA_DIR / "books.csv"
TAGS_PATH = RAW_DATA_DIR / "tags.csv"
BOOK_TAGS_PATH = RAW_DATA_DIR / "book_tags.csv"
TO_READ_PATH = RAW_DATA_DIR / "to_read.csv"

# Подготовленные данные для обучения и оценки моделей
TRAIN_RATINGS_PATH = PROCESSED_DATA_DIR / "train_ratings.csv"
TEST_RATINGS_PATH = PROCESSED_DATA_DIR / "test_ratings.csv"

# === Настройки экспериментов ===

# Фиксируем случайность
# для воспроизводимости результатов
RANDOM_STATE = 42
# Количество книг
# в итоговой выдаче рекомендаций
TOP_K = 5
# Минимальная оценка,
# при которой книга считается релевантной
RELEVANCE_THRESHOLD = 4
# Доля взаимодействий
# для проверки качества моделей
TEST_SIZE = 0.2

# === Настройки Content-Based рекомендаций ===

# Количество наиболее значимых тегов для каждой книги
MAX_TAGS_PER_BOOK = 20
