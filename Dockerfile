# Użyj lekkiego obrazu z Pythonem
FROM python:3.9-slim

# Ustaw katalog roboczy
WORKDIR /app

RUN apt update && apt install -y iputils-ping curl

# Skopiuj wymagania i zainstaluj
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj cały projekt
COPY . .

#COPY hf-cache /root/.cache/huggingface

# Exponuj port (FastAPI domyślnie działa na 8000)
EXPOSE 8000

# Komenda uruchamiająca aplikację
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
