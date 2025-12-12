FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# Install build tools (kept for simplicity in single-stage image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["python", "mcp_server.py"]
