FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

RUN pip install --upgrade pip

# Install par couches pour profiter du cache Docker en cas de coupure réseau.
# Si une couche échoue, `docker-compose build` reprendra à partir d'elle.
RUN pip install fastapi==0.111.0 "uvicorn[standard]==0.29.0"
RUN pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2
RUN pip install yfinance==0.2.40
RUN pip install tensorflow-cpu==2.16.1
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.0+cpu
RUN pip install stable-baselines3==2.4.0 gymnasium==0.29.1

COPY requirements.txt .

COPY api.py env.py train.py ./
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
