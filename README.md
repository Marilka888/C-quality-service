# c-quality-service

Микросервис проверки согласованности требований ТЗ с целевыми документами (ПМИ, ПЗ).
Реализует pipeline: type-aware retrieval → reranker → LLM-judge → rule-based verifier → агрегация.

## Роль в pipeline

```
docback → POST /coverage/analyze (tz + pmi/pz artifacts) → c-quality-service
                                                                  ↓
                                                    CoverageAnalysisResponse
                                                    └── requirement_results[]
                                                        ├── status: COVERED | PARTIAL | MISSING | CONFLICT | UNKNOWN
                                                        ├── low_confidence
                                                        ├── grounding_failed
                                                        └── evidence_trace
```

## Статусы покрытия

| Статус | Значение |
|---|---|
| `COVERED` | Требование явно покрыто в целевом документе |
| `PARTIAL` | Покрыто частично |
| `MISSING` | Соответствие не найдено |
| `CONFLICT` | Найдено противоречие между ТЗ и целевым документом |
| `UNKNOWN` | LLM-судья недоступен; вердикт не может быть вынесен |

`UNKNOWN` не учитывается в критическом счётчике — это признак инфраструктурного сбоя, а не дефекта документа.

## Стек

- Python 3.11
- FastAPI + Uvicorn
- sentence-transformers / XLMRoberta (эмбеддинги + reranker)
- spaCy `ru_core_news_md` (лемматизация)
- litellm / Ollama (LLM-судья)
- SQLite (кэш суждений LLM)

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download ru_core_news_md
```

### 2. LLM-модель (Ollama)

```bash
ollama pull qwen2.5:3b   # ~2 GB, достаточно для демо
# или qwen2.5:7b для лучшего качества
```

Поддерживаются любые Ollama-модели — задаётся через `CQUALITY_LLM_MODEL_NAME`.

### 3. Веса классификатора требований

Файлы `model/model.safetensors` и `model/req_classifier/model.safetensors` **не хранятся в git** (бинарные, ~1.7 ГБ суммарно). Конфигурационные файлы (`config.json`, `tokenizer.json` и др.) уже в репозитории.

**Варианты:**

- Скопировать веса с другой машины в `model/` и `model/req_classifier/`
- Sentence-transformers автоматически скачает base-модель XLMRoberta при первом запуске (конфигурация уже присутствует)

### 4. Запуск

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

Health-check: `GET http://localhost:8004/health`

### 5. Подключение к docback

В переменных окружения docback:

```env
CHECK_C_BASE_URL=http://localhost:8004
```

## Docker

```bash
docker build -t c-quality-service .
docker run -p 8004:8000 \
  -v /path/to/models:/app/model \
  -e CQUALITY_LLM_MODEL_NAME=qwen2.5:3b \
  -e CQUALITY_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  c-quality-service
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CQUALITY_LLM_MODEL_NAME` | `qwen2.5:3b` | Модель Ollama для LLM-судьи |
| `CQUALITY_OLLAMA_BASE_URL` | `http://localhost:11434` | URL Ollama-сервера |
| `CQUALITY_LLM_ENABLED` | `true` | Включить LLM-judge (false = только rule-based) |
| `CQUALITY_REQ_CONCURRENCY` | `4` | Параллелизм обработки требований |
| `CQUALITY_JUDGE_CONCURRENCY` | `2` | Параллелизм LLM-запросов |
| `CQUALITY_MIN_RETRIEVAL_SCORE` | `0.05` | Минимальный retrieval-score для допуска к LLM |
| `CQUALITY_EVIDENCE_FLOOR` | `0.30` | Минимальный evidence-score |
| `CQUALITY_USE_RERANKER` | `true` | Включить BGE reranker |

## API

### `POST /coverage/analyze`

Основной endpoint — полный pipeline.

```jsonc
{
  "tz_artifact": { "document_id": "...", "doc_role": "tz", "sections": [...], "requirement_candidates": [...] },
  "target_artifacts": [
    { "document_id": "...", "doc_role": "pmi", "sections": [...], "fragments": [...] }
  ],
  "config": {
    "llm": { "enabled": true, "backend": "ollama" }
  }
}
```

**Ответ:**

```jsonc
{
  "requirement_results": [
    {
      "req_id": "r_001",
      "text": "Система должна...",
      "status": "COVERED",
      "low_confidence": false,
      "grounding_failed": false,
      "status_subcode": null,
      "evidence_trace": "..."
    }
  ],
  "summary": {
    "total": 10,
    "covered": 7,
    "partial": 1,
    "missing": 1,
    "conflict": 1,
    "unknown": 0
  }
}
```

### `GET /health`

```json
{ "status": "ok" }
```

## Тесты

```bash
python -m pytest tests/ -v
```

## Структура проекта

```
app/
├── api/                   # FastAPI router, request/response schemas
├── application/use_cases/ # pipeline use cases (retrieval, judge, aggregate)
├── core/                  # config, logging, lemmatizer, text utils
├── domain/                # entities, enums, value objects
├── infrastructure/
│   ├── embeddings/        # e5 multilingual embeddings
│   ├── llm/               # LLM judge implementations (Ollama, disabled, etc.)
│   ├── reranker/          # BGE reranker
│   └── rules/             # deterministic conflict detector
├── retrieval/             # hybrid retriever (lexical + semantic)
└── scoring/               # aggregation + metrics
model/                     # конфиги sentence-transformer (веса не в git)
data/packages/             # тестовые пакеты для калибровки
scripts/                   # калибровка, диагностика, датасет-билдер
tests/                     # unit и integration тесты
```

## Ограничения

- LLM-судья калиброван под `qwen2.5:7b`; на других моделях вердикты могут смещаться.
- Retrieval ориентирован на русскоязычные документы; смешанные или англоязычные входы деградируют до лексического поиска.
- Строки с `low_confidence=true` или `grounding_failed=true` следует считать предварительными.
