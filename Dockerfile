FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIDETECTOR_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml readme.md ./
COPY src ./src
COPY aidetector ./aidetector
COPY models ./models
COPY start.sh ./start.sh

RUN pip install --upgrade pip && pip install -e ".[api,image]"

EXPOSE 8000
CMD ["./start.sh"]
