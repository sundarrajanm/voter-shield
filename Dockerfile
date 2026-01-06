# ---- Base runtime ----
FROM python:3.12.6-slim-bookworm

# ---- Environment hygiene ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---- System dependencies (minimal OCR-safe set) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tam \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY . .

# ---- Default command (1 booth run) ----
CMD ["python", "main.py"]
