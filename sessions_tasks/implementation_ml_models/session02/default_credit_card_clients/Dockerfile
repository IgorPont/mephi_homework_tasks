FROM python:3.12-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения, модели и данные
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Порт Flask/Gunicorn-сервиса
EXPOSE 5001

# Запускаем API через gunicorn
# Один контейнер с монолитным ML API
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app.api:app"]