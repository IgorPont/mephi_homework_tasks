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

# Если нужны Jupyter-ноутбуки (группа extras notebooks)
poetry install -E notebooks
```

---

## 🧠 Работа с Jupyter Notebook / Lab

### 📦 Первый запуск (только один раз)

```bash
poetry install -E notebooks

# Зарегистрировать ядро Jupyter для этого проекта
poetry run python -m ipykernel install --user   --name mephi-homework-tasks-py313   --display-name "Python 3.13 (mephi-homework-tasks)"
```

### ▶️ Запуск Jupyter

```bash
poetry run jupyter lab
```

или

```bash
poetry run jupyter notebook
```

После запуска интерфейс откроется в браузере.  
В правом верхнем углу выберите ядро **Python 3.13 (mephi-homework-tasks)**.

### 🛑 Остановка Jupyter

- В терминале, где запущен сервер -> `Ctrl + C`, затем `y` для подтверждения.
- Или через интерфейс: **File -> Shut Down**.

### 🔁 Перезапуск

```bash
poetry run jupyter lab
```

### 🧹 Проверка и управление ядрами

```bash
# Показать список всех доступных ядер
jupyter kernelspec list

# Удалить (если пересоздали окружение)
jupyter kernelspec remove mephi-homework-tasks-py313
```

### 💡 Пример Makefile для быстрого старта

```make
.PHONY: notebooks
notebooks:
	poetry install -E notebooks
	poetry run python -m ipykernel install --user --name mephi-homework-tasks-py313 --display-name "Python 3.13 (mephi-homework-tasks)"
	poetry run jupyter lab
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

✍️ **Автор:** *IgorPont*
🛠 *Python 3.13 • Poetry • Pytest • Ruff • Black • Jupyter*
