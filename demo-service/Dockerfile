FROM python:3.11-slim

# 빌드 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# requirements 설치
COPY requirements.txt .

RUN pip install --no-cache-dir numpy==1.24.4
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# uvicorn 실행 (reload 옵션 추가) ⭐
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
