FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# main.py re-exports app.main:app — uvicorn entrypoint stays compatible
# with the legacy `python main.py` invocation. /health endpoint and
# startup warmup live in app/main.py (P0 #8).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
