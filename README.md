# C-quality-service

LLM-judge сервис для качества C (согласованность ТЗ ↔ ПМИ / ПЗ).
Принимает требования из ТЗ и фрагменты из целевых документов (ПМИ, ПЗ),
прогоняет их через retrieval + reranker + LLM-judge и возвращает покрытие
с пометками COVERED / PARTIAL / MISSING / CONFLICT.

## Запуск с нуля

### 1. Зависимости системы

- **Python 3.11**
- **Git**
- **Ollama** (https://ollama.com) — для локального LLM-судьи

### 2. Клон и установка Python-пакетов

```bash
git clone https://github.com/Marilka888/C-quality-service.git
cd C-quality-service
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download ru_core_news_md
```

### 3. Скачать LLM-модель для Ollama

```bash
ollama pull qwen2.5:3b
```

Можно использовать другую модель (`llama3`, `mistral`, и т.д.) — главное
прописать её имя в env-переменной `CQUALITY_LLM_MODEL_NAME` ниже.

### 4. Weights модели классификатора требований

Weights `model.safetensors` **не лежат в git** (слишком большие). Есть
два варианта:

**(а) Перенести с другого компа**, где сервис уже работает:

```
model/model.safetensors
model/req_classifier/model.safetensors
```

Положить в те же относительные пути.

**(б) Скачать base-модель автоматически.** Sentence-transformers
подтянет базовую `XLMRoberta` при первом запуске. Файлы конфигурации
уже в репе (`model/config.json`, `tokenizer.json`, …).

### 5. Переменные окружения

PowerShell:
```powershell
$env:CQUALITY_LLM_MODEL_NAME = "qwen2.5:3b"
$env:CQUALITY_REQ_CONCURRENCY = "4"
$env:CQUALITY_JUDGE_CONCURRENCY = "2"
```

bash:
```bash
export CQUALITY_LLM_MODEL_NAME="qwen2.5:3b"
export CQUALITY_REQ_CONCURRENCY=4
export CQUALITY_JUDGE_CONCURRENCY=2
```

### 6. Запустить сервис

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

Сервис будет доступен на `http://localhost:8004`.
Health-check: `GET /health`.

### 7. Подключить к оркестратору docback

В env-переменных `docback` (или в `docker-compose.yml`) пропиши URL:
```
CQUALITY_SERVICE_URL=http://localhost:8004
```
(имя переменной зависит от твоей конфигурации docback).

## Запуск в Docker

```bash
docker build -t c-quality-service .
docker run -p 8004:8000 -e CQUALITY_LLM_MODEL_NAME=qwen2.5:3b c-quality-service
```

Ollama должен быть доступен с контейнера — либо запусти его на хосте и
прокинь `host.docker.internal` в env, либо разверни Ollama в том же
docker-compose.
