# Default of Credit Card Clients

Итоговый проект по дисциплине **"Внедрение моделей машинного обучения"**

Проект представляет собой контейнеризированный ML API-сервис для прогнозирования дефолта клиента по кредитной карте на
следующий месяц.

## 1. Описание задачи

Цель проекта -> реализовать полный минимальный цикл внедрения модели машинного обучения:

- загрузка и анализ датасета
- обучение моделей бинарной классификации
- сохранение обученных моделей
- разработка API для инференса
- контейнеризация сервиса
- тестирование
- описание архитектуры, мониторинга и A/B-тестирования

Используется датасет **Default of Credit Card Clients Dataset**, который можно скачать по ссылке:

```text
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
```

Целевая переменная:

```text
default.payment.next.month
```

Значения целевой переменной:

- `0` -> дефолт не ожидается
- `1` -> ожидается дефолт

## 2. Структура проекта

```text
default_credit_card_clients/
├── app/
│   ├── __init__.py                         # Делает папку app Python-пакетом
│   ├── api.py                              # Flask API -> эндпоинты /health, /predict, /predict/v1, /predict/v2
│   ├── config.py                           # Настройки сервиса -> пути к моделям, порт, версия модели по умолчанию
│   ├── model_handler.py                    # Загрузка joblib-моделей и выполнение предсказаний
│   └── schemas.py                          # Список признаков и валидация входного JSON-запроса
│
├── data/
│   └── UCI_Credit_Card.csv                 # Исходный датасет клиентов кредитных карт
│
├── docs/
│   ├── AB_TEST_PLAN.md                     # План A/B-тестирования моделей v1 и v2
│   └── ARCHITECTURE.md                     # Описание архитектуры ML-сервиса и production-подходов
│
├── models/
│   ├── train_model.py                      # Скрипт обучения моделей и сохранения артефактов
│   ├── credit_default_model_v1.joblib      # Сохраненная модель v1 -> LogisticRegression
│   └── credit_default_model_v2.joblib      # Сохраненная модель v2 -> RandomForestClassifier
│
├── tests/
│   └── test_api.py                         # Pytest-тесты для проверки API
│
├── Dockerfile                              # Инструкция сборки Docker-образа сервиса
├── docker-compose.yml                      # Запуск API-сервиса через Docker Compose
├── requirements.txt                        # Python-зависимости проекта
└── README.md                               # Основное описание проекта, запуск, API и результаты
```

## 3. Используемые технологии

- Python 3.12
- Flask
- Gunicorn
- pandas
- numpy
- scikit-learn
- joblib
- pytest
- Docker
- Docker Compose

## 4. Порядок запуска проекта

Ниже приведен порядок действий для проверки проекта после скачивания папки `default_credit_card_clients`.

### 4.1. Перейти в папку проекта

Если скачена только папка с проектом:

```bash
cd default_credit_card_clients
```

Если проект находится внутри общего репозитория:

```bash
cd sessions_tasks/implementation_ml_models/session02/default_credit_card_clients
```

### 4.2. Создать и активировать виртуальное окружение

Для macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Для Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4.3. Установить зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4. Проверить наличие датасета

В папке `data/` должен находиться файл:

```text
data/UCI_Credit_Card.csv
```

Если обученные модели уже есть в папке `models/`, повторное обучение выполнять необязательно.

### 4.5. При необходимости обучить модели

```bash
python models/train_model.py
```

После выполнения команды в папке `models/` должны появиться файлы:

```text
models/credit_default_model_v1.joblib
models/credit_default_model_v2.joblib
```

В проекте используются две модели:

- `v1` -> LogisticRegression
- `v2` -> RandomForestClassifier

По умолчанию API использует модель `v2`, так как она показала лучшее качество на тестовой выборке.

### 4.6. Запустить API локально

```bash
python app/api.py
```

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:5001
```

### 4.7. Проверить работоспособность сервиса

В новом окне терминала выполнить:

```bash
curl http://127.0.0.1:5001/health
```

Пример ответа:

```json
{
  "default_model_version": "v2",
  "service": "credit-default-prediction-api",
  "status": "healthy",
  "timestamp": "2026-06-06T14:43:56.924465+00:00"
}
```

### 4.8. Выполнить прогноз через API

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "PAY_AMT6": 0
  }'
```

Пример ответа:

```json
{
  "model_version": "v2",
  "prediction": 1,
  "probability": 0.827695,
  "risk_label": "default"
}
```

Где:

- `prediction = 0` -> дефолт не ожидается
- `prediction = 1` -> ожидается дефолт
- `probability` -> вероятность дефолта
- `model_version` -> версия использованной модели
- `risk_label` -> текстовая интерпретация прогноза

### 4.9. Проверить отдельные версии моделей

Контрольная модель `v1`:

```bash
curl -X POST http://127.0.0.1:5001/predict/v1 \
  -H "Content-Type: application/json" \
  -d '{
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
    "PAY_AMT6": 0
  }'
```

Тестовая модель `v2`:

```bash
curl -X POST http://127.0.0.1:5001/predict/v2 \
  -H "Content-Type: application/json" \
  -d '{
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
    "PAY_AMT6": 0
  }'
```

### 4.10. Запустить тесты

```bash
pytest tests/
```

Ожидаемый результат:

```text
5 passed
```

### 4.11. Запуск через Docker

Собрать и запустить контейнер:

```bash
docker compose up --build
```

После запуска API будет доступен по адресу:

```text
http://127.0.0.1:5001
```

Проверка health-check:

```bash
curl http://127.0.0.1:5001/health
```

Остановить контейнер:

```bash
docker compose down
```

### 4.12. Краткий сценарий полной проверки

```bash
cd default_credit_card_clients

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python models/train_model.py
python app/api.py
```

В новом терминале:

```bash
curl http://127.0.0.1:5001/health
```

Затем выполнить `POST /predict` с JSON-признаками клиента.

## 5. Модели

В проекте обучены две версии модели.

### v1 -> LogisticRegression

Базовая модель для сравнения.

Метрики на тестовой выборке:

| Метрика   | Значение |
|-----------|---------:|
| Accuracy  |   0.6797 |
| Precision |   0.3672 |
| Recall    |   0.6202 |
| F1-score  |   0.4613 |
| ROC-AUC   |   0.7081 |

### v2 -> RandomForestClassifier

Основная модель сервиса.

Метрики на тестовой выборке:

| Метрика   | Значение |
|-----------|---------:|
| Accuracy  |   0.7872 |
| Precision |   0.5168 |
| Recall    |   0.5795 |
| F1-score  |   0.5464 |
| ROC-AUC   |   0.7750 |

Модель `v2` выбрана моделью по умолчанию, так как она показала более высокие значения `F1-score` и `ROC-AUC`.

## 6. Обучение моделей

Для обучения моделей используется скрипт:

```text
models/train_model.py
```

Запуск из папки проекта:

```bash
python models/train_model.py
```

Или через Poetry из корня репозитория:

```bash
poetry run python sessions_tasks/implementation_ml_models/session02/default_credit_card_clients/models/train_model.py
```

После запуска будут созданы файлы:

```text
models/credit_default_model_v1.joblib
models/credit_default_model_v2.joblib
```

## 7. Локальный запуск API

Из папки проекта:

```bash
python app/api.py
```

Сервис запускается на порту `5001`.

Проверка работоспособности:

```bash
curl http://127.0.0.1:5001/health
```

Пример ответа:

```json
{
  "default_model_version": "v2",
  "service": "credit-default-prediction-api",
  "status": "healthy",
  "timestamp": "2026-06-06T14:43:56.924465+00:00"
}
```

## 8. API

### GET /health

Проверка состояния сервиса.

Пример запроса:

```bash
curl http://127.0.0.1:5001/health
```

### POST /predict

Прогноз моделью по умолчанию.

Пример запроса:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "PAY_AMT6": 0
  }'
```

Пример ответа:

```json
{
  "model_version": "v2",
  "prediction": 1,
  "probability": 0.827695,
  "risk_label": "default"
}
```

### POST /predict/v1

Прогноз моделью `v1`, используется как контрольная версия для A/B-тестирования.

### POST /predict/v2

Прогноз моделью `v2`, используется как новая версия модели-кандидата.

## 9. Docker-запуск

Перед сборкой контейнеров необходимо установить Docker.

Инструкция по установке:

```text
https://docs.docker.com/engine/install/
```

Сборка и запуск контейнера:

```bash
docker compose up --build
```

После запуска сервис доступен по адресу:

```text
http://127.0.0.1:5001
```

Проверка:

```bash
curl http://127.0.0.1:5001/health
```

Остановка контейнера:

```bash
docker compose down
```

## 10. Тестирование

Запуск тестов из папки проекта:

```bash
pytest tests/
```

Фактический результат:

```text
5 passed in 2.16s
```

Тесты проверяют работу:

- `/health`
- `/predict`
- `/predict/v1`
- `/predict/v2`
- ошибку при неполном JSON-запросе

## 11. Архитектура

Проект реализован как монолитный ML API-сервис.

Основные компоненты:

- `Flask API` -> принимает HTTP-запросы
- `schemas.py` -> валидирует входные признаки
- `ModelHandler` -> загружает модели и выполняет инференс
- `joblib` -> хранит обученные ML-модели
- `Gunicorn` -> запускает приложение в контейнере
- `Docker` -> обеспечивает воспроизводимое окружение

Подробное описание архитектуры находится в файле проекта:

```text
docs/ARCHITECTURE.md
```

## 12. Логирование и мониторинг

В сервисе настроено базовое логирование событий инференса.

Логируются:

- факт успешного предсказания
- версия модели
- прогноз
- вероятность дефолта
- ошибки валидации
- внутренние ошибки сервиса

В production-среде можно дополнительно отслеживать:

- количество запросов
- среднее время ответа
- долю ошибок
- распределение вероятностей дефолта
- drift признаков
- drift предсказаний
- качество модели после появления фактических дефолтов

## 13. DVC и MLflow

В текущей реализации DVC и MLflow описаны как часть возможного production-процесса.

DVC может использоваться для версионирования:

- исходного датасета
- train/test выборок
- подготовленных признаков
- артефактов моделей

MLflow может использоваться для:

- логирования параметров экспериментов
- логирования метрик
- сравнения моделей
- регистрации лучшей модели
- управления жизненным циклом моделей

## 14. A/B-тестирование

В проекте предусмотрены две версии модели:

- `v1` -> контрольная модель
- `v2` -> тестовая модель

Для демонстрации A/B-подхода реализованы отдельные эндпоинты:

```text
POST /predict/v1
POST /predict/v2
```

Подробный план A/B-тестирования находится в файле проекта:

```text
docs/AB_TEST_PLAN.md
```

## 15. Бизнес-метрики

Для оценки эффекта модели в бизнесе можно использовать:

- default rate (уровень дефолта) среди одобренных клиентов
- approval rate (уровень одобрения)
- финансовые потери от дефолтов
- упущенную прибыль из-за отказа хорошим клиентам
- profit per approved application (прибыль на одобренную заявку)
- долю ручных проверок
- качество кредитного портфеля

Основная бизнес-цель модели -> снизить финансовые потери от дефолтов без чрезмерного отказа платежеспособным клиентам.

## 16. Итог

В проекте реализован полный минимальный pipeline внедрения ML-модели:

- обучены две модели бинарной классификации
- модели сохранены в формате `joblib`
- реализован Flask API для инференса
- добавлены эндпоинты для A/B-тестирования
- сервис упакован в Docker
- добавлен Docker Compose
- написаны тесты API
- подготовлена документация по архитектуре и A/B-тестированию
