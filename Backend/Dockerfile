FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DJANGO_SETTINGS_MODULE=Backend.settings

EXPOSE 8000

CMD ["uvicorn", "Backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
