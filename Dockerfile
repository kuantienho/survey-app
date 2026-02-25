FROM python:3.11-slim

# 系統層套件（避免 numpy / torch / faiss 安裝炸掉）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先裝 Python 套件（快取比較有效）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 安裝 spaCy 模型（你有用到）
RUN python -m spacy download en_core_web_sm

# 複製整個專案
COPY . /app

ENV PORT=10000

# SQLite + gunicorn（單 worker，避免鎖表）
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:10000", "app:app"]