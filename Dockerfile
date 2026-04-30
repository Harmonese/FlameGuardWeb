FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/logs \
    && addgroup --system flameguard \
    && adduser --system --ingroup flameguard flameguard \
    && chown -R flameguard:flameguard /app

USER flameguard

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=3).read()" || exit 1

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "90", "--bind", "0.0.0.0:5000", "app:app"]
