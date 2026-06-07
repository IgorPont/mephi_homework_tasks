"""
Модуль обучения моделей для прогнозирования дефолта по кредитной карте.

Логика:
    1. Загружает датасет UCI Credit Card
    2. Разделяет данные на train/test
    3. Обучает две модели:
        - v1: LogisticRegression
        - v2: RandomForestClassifier
    4. Сохраняет модели в формате joblib
    5. Выводит метрики качества

Запуск из корня проекта default_credit_card_clients:
    python models/train_model.py

Запуск из корня репозитория:
    poetry run python sessions_tasks/implementation_ml_models/session02/default_credit_card_clients/models/train_model.py

Пример вывода:
    Обучение модели v1: LogisticRegression...
    Обучение модели v2: RandomForestClassifier...
    ================================================================================
    Метрики модели: v1 LogisticRegression
    ================================================================================
    Accuracy: 0.6797
    Precision: 0.3672
    Recall: 0.6202
    F1-score: 0.4613
    ROC-AUC: 0.7081

    Classification report:
                  precision    recall  f1-score   support

               0       0.87      0.70      0.77      4673
               1       0.37      0.62      0.46      1327

        accuracy                           0.68      6000
       macro avg       0.62      0.66      0.62      6000
    weighted avg       0.76      0.68      0.70      6000

    ================================================================================
    Метрики модели: v2 RandomForestClassifier
    ================================================================================
    Accuracy: 0.7872
    Precision: 0.5168
    Recall: 0.5795
    F1-score: 0.5464
    ROC-AUC: 0.7750

    Classification report:
                  precision    recall  f1-score   support

               0       0.88      0.85      0.86      4673
               1       0.52      0.58      0.55      1327

        accuracy                           0.79      6000
       macro avg       0.70      0.71      0.70      6000
    weighted avg       0.80      0.79      0.79      6000

    ================================================================================
    Модели успешно сохранены:
    - /Users/pontigor/python_test/mephi_homework_tasks/sessions_tasks/implementation_ml_models/session02/default_credit_card_clients/models/credit_default_model_v1.joblib
    - /Users/pontigor/python_test/mephi_homework_tasks/sessions_tasks/implementation_ml_models/session02/default_credit_card_clients/models/credit_default_model_v2.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Корневая директория текущего проекта:
# default_credit_card_clients/
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути к данным и моделям
DATA_PATH = BASE_DIR / "data" / "UCI_Credit_Card.csv"
MODEL_V1_PATH = BASE_DIR / "models" / "credit_default_model_v1.joblib"
MODEL_V2_PATH = BASE_DIR / "models" / "credit_default_model_v2.joblib"

# Целевая переменная
TARGET_COLUMN = "default.payment.next.month"

# Служебный столбец, который не нужен для обучения
ID_COLUMN = "ID"

# Фиксируем random_state для воспроизводимости
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """
    Загружает исходный датасет.

    Возвращает:
        pd.DataFrame: таблица с данными клиентов
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Разделяет датасет на признаки и целевую переменную

    Аргументы:
        data: исходный датасет

    Возвращает:
        tuple[pd.DataFrame, pd.Series]: признаки X и целевая переменная y
    """

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"В датасете отсутствует целевой столбец: {TARGET_COLUMN}")

    # ID не берем
    columns_to_drop = [TARGET_COLUMN]

    if ID_COLUMN in data.columns:
        columns_to_drop.append(ID_COLUMN)

    x = data.drop(columns=columns_to_drop)
    y = data[TARGET_COLUMN]

    return x, y


def build_logistic_regression_model(feature_columns: list[str]) -> Pipeline:
    """
    Создает pipeline с LogisticRegression.

    Для логистической регрессии масштабирование признаков важно,
    поэтому используем StandardScaler.

    Аргументы:
        feature_columns: список признаков

    Возвращает:
        Pipeline: sklearn pipeline для обучения и инференса
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_columns),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ],
    )


def build_random_forest_model(feature_columns: list[str]) -> Pipeline:
    """
    Создает pipeline с RandomForestClassifier.

    RandomForest хорошо подходит для табличных данных и не требует
    обязательного масштабирования числовых признаков.

    Аргументы:
        feature_columns: список признаков

    Возвращает:
        Pipeline: sklearn pipeline для обучения и инференса
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("classifier", model),
        ],
    )


def print_metrics(
    model_name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Выводит основные метрики качества модели.

    Аргументы:
        - model_name: название модели
        - model: обученная модель
        - x_test: тестовые признаки
        - y_test: истинные значения целевой переменной
    """

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    print("=" * 80)
    print(f"Метрики модели: {model_name}")
    print("=" * 80)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))


def train_and_save_models() -> None:
    """
    Обучает две версии модели и сохраняет их в директорию models/
    """

    data = load_dataset()
    x, y = split_features_and_target(data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    feature_columns = list(x_train.columns)

    model_v1 = build_logistic_regression_model(feature_columns)
    model_v2 = build_random_forest_model(feature_columns)

    print("Обучение модели v1: LogisticRegression...")
    model_v1.fit(x_train, y_train)

    print("Обучение модели v2: RandomForestClassifier...")
    model_v2.fit(x_train, y_train)

    print_metrics(
        model_name="v1 LogisticRegression",
        model=model_v1,
        x_test=x_test,
        y_test=y_test,
    )

    print_metrics(
        model_name="v2 RandomForestClassifier",
        model=model_v2,
        x_test=x_test,
        y_test=y_test,
    )

    joblib.dump(model_v1, MODEL_V1_PATH)
    joblib.dump(model_v2, MODEL_V2_PATH)

    print("=" * 80)
    print("Модели успешно сохранены:")
    print(f"- {MODEL_V1_PATH}")
    print(f"- {MODEL_V2_PATH}")


if __name__ == "__main__":
    train_and_save_models()
