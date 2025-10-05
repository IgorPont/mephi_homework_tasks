# mephi_homework_tasks

Решения домашних и сессионных заданий НИЯУ МИФИ по прикладному анализу данных и машинному обучению.

## 📦 Установка окружения

```bash
# Установить зависимости
poetry install

# Если нужны Jupyter-ноутбуки
poetry install -E notebooks
```

## 🧪 Запуск тестов

```bash
poetry run pytest
```

## 🎨 Форматирование и линтинг

```bash
# Проверка Ruff (линтер + isort)
poetry run ruff check .

# Автоисправления Ruff
poetry run ruff check . --fix

# Форматирование Black
poetry run black .
```

## ⏳ Пример быстрой проверки

```bash
poetry run ruff check .
poetry run black .
poetry run pytest -q
```

## 📓 Jupyter

Чтобы использовать окружение в Jupyter Notebook или Lab:

```bash
poetry run python -m ipykernel install --user --name mephi-homework-tasks
```

После этого в Jupyter можно выбрать ядро `mephi-homework-tasks`.

---

✍️ Автор: IgorPont
