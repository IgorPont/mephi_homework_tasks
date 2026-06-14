"""
Конфигурация проекта анализа сайта "СберАвтоподписка".

В этом модуле хранятся пути к файлам, директориям и основные константы,
которые используются в скриптах подготовки данных, обучения модели
и получения предсказаний.
"""

from pathlib import Path

# Корневая директория задачи:
# sessions_tasks/classical_ml/session02/sber_auto_subscription
BASE_DIR = Path(__file__).resolve().parents[1]

# Директории с данными
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Директории для моделей и результатов
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Исходные данные в формате pickle
GA_SESSIONS_PKL_PATH = RAW_DATA_DIR / "ga_sessions.pkl"
GA_HITS_PKL_PATH = RAW_DATA_DIR / "ga_hits.pkl"

# Исходные данные в формате CSV
GA_SESSIONS_CSV_PATH = RAW_DATA_DIR / "ga_sessions.csv"
GA_HITS_CSV_PATH = RAW_DATA_DIR / "ga_hits.csv"

# Подготовленный датасет
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "sber_auto_dataset.csv"

# Сохраненная модель и результаты
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"

# Название целевой переменной
TARGET_COLUMN = "target"

# Список целевых действий.
# Визит считается конверсионным, если в рамках session_id
# было совершено хотя бы одно действие из этого списка
TARGET_ACTIONS = [
    # Успешные заявки и отправки форм по подписке
    "sub_submit_success",
    "sub_car_claim_submit_click",
    "sub_car_request_submit_click",
    "sub_custom_question_submit_click",

    # Заказ звонка / переход к звонку
    "sub_callback_submit_click",
    "sub_call_number_click",
    "callback requested",
    "form_request_call_sent",
    "click_on_request_call",

    # Открытие диалога / чат
    "sub_open_dialog_click",
    "chat requested",
    "chat established",
    "client initiate chat",
    "user gave contacts during chat",
    "start_chat",

    # GreenDay заявки / звонки / диалоги
    "greenday_sub_submit_success",
    "greenday_sub_callback_submit_click",
    "greenday_sub_call_number_click",
    "greenday_sub_open_dialog_click",

    # Общие успешные заявки
    "request_success",
]

# Признаки, которые не используются в финальной модели из-за риска data leakage (утечки данных).
# Они рассчитываются по событиям внутри всей сессии и могут содержать информацию
# о действиях, произошедших после целевого события
LEAKAGE_RISK_FEATURES = [
    "hit_count",
    "max_hit_number",
    "unique_page_count",
    "unique_event_action_count",
    "unique_event_category_count",
    "has_view_card",
    "has_search",
    "has_pagination",
    "has_quiz",
    "has_subscription_interest",
    "has_phone_interaction",
    "has_form_interaction",
]
