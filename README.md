# Writing Buddy MVP

A minimal paper-writing assistant web app that suggests multiple English next sentences from existing context.

## Stack
- FastAPI
- Jinja2 templates
- Local Ollama API
- Vanilla JS + localStorage

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Ollama is running locally and a model is available (default in code: `qwen3:8b`).
   You can override defaults with env vars:
   ```bash
   export OLLAMA_URL=http://127.0.0.1:11434/api/generate
   export OLLAMA_MODEL=qwen3:8b
   ```
3. Start server:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Open: `http://127.0.0.1:8000`

## Troubleshooting Ollama
If you see `Ollama service unavailable`, check:
```bash
curl http://127.0.0.1:11434/api/tags
ollama list
```
- If connection fails, Ollama daemon is not reachable from this shell.
- If model not found, run:
```bash
ollama pull qwen3:8b
```

## API
- `POST /api/recommend-next-sentences`

Request:
```json
{
  "context_text": "Your English paragraph",
  "candidate_count": 5
}
```

Response:
```json
{
  "sentences": ["..."],
  "meta": {
    "model": "qwen3:8b",
    "requested": 5,
    "returned": 5,
    "filtered": 1,
    "terms": ["..."]
  }
}
```
