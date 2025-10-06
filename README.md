# 🧮 mephi_homework_tasks

📊 **Решения домашних и сессионных заданий НИЯУ МИФИ по дисциплинам  
*"Прикладной анализ данных"* и *"Машинное обучение"* на языке Python**.

---

## 📂 Структура проекта

```
homework_tasks/    # Домашние задания (по модулям: алгебра, статистика, ML и др.)
sessions_tasks/    # Сессионные задания (по темам и дисциплинам)
tests/             # Тесты Pytest, сгруппированные по пакетам
pyproject.toml     # Конфигурация Poetry, Ruff, Black, Pytest
poetry.lock
README.md
```

---

## ⚙️ Установка окружения

```bash
# Установить зависимости проекта
poetry install

# Если нужны Jupyter-ноутбуки
poetry install -E notebooks
```

---

## 🧪 Запуск тестов

```bash
# Запуск всех тестов
poetry run pytest

# Тихий режим (только статусы)
poetry run pytest -q

# Запуск только тестов по заданию
poetry run pytest tests/sessions_tasks/programming_in_python/session01/test_task06_schrodinger_eraser.py -v

# Проверить один тест внутри модуля
poetry run pytest tests/sessions_tasks/programming_in_python/session01/test_task06_schrodinger_eraser.py::test_erase_in_file_io -v
```

---

## 🎨 Линтинг и автоформатирование

```bash
# Проверка Ruff (линтер + isort)
poetry run ruff check .

# Автоисправления Ruff
poetry run ruff check . --fix

# Форматирование Black
poetry run black .
```

---

## 🚀 Пример быстрой проверки проекта

```bash
poetry run ruff check .
poetry run black .
poetry run pytest -q
```

---

## 📓 Работа в Jupyter

Чтобы добавить окружение Poetry в Jupyter:

```bash
poetry run python -m ipykernel install --user --name homework-tasks
```

После этого в Jupyter Notebook или Lab можно выбрать ядро **`homework-tasks`**.

---

✍️ **Автор:** *IgorPont*  
📧 *pontigor11@gmail.com*  
🛠 *Python 3.13 • Poetry • Pytest • Ruff • Black*
