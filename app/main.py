import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request


BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
MIN_SENTENCE_LEN = 25
MAX_SENTENCE_LEN = 280


app = FastAPI(title="Writing Buddy MVP", version="0.1.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class RecommendRequest(BaseModel):
    context_text: str = Field(..., min_length=30)
    candidate_count: int = Field(default=5, ge=3, le=8)


class RecommendResponse(BaseModel):
    sentences: List[str]
    meta: dict


def extract_terms(text: str, max_terms: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\\-]{4,}", text.lower())
    stop = {
        "which",
        "therefore",
        "however",
        "because",
        "between",
        "within",
        "their",
        "these",
        "those",
        "whereas",
        "demonstrates",
        "analysis",
        "results",
        "method",
        "methods",
        "paper",
        "study",
    }
    counts = Counter(w for w in words if w not in stop)
    return [w for w, _ in counts.most_common(max_terms)]


def build_prompt(context_text: str, candidate_count: int, terms: List[str]) -> str:
    term_constraints = ", ".join(terms) if terms else "none"
    return (
        "You assist English academic writing.\n"
        "Generate candidate next sentences ONLY, not instructions.\n"
        f"Return exactly {candidate_count} distinct items.\n"
        "Each item must be one complete academic sentence.\n"
        "No meta text, no explanation, no markdown, no numbering.\n"
        f"Preserve continuity and terminology when relevant: {term_constraints}\n\n"
        "Return strict JSON object:\n"
        '{"sentences": ["...", "..."]}\n\n'
        f"Context paragraph:\n{context_text}\n"
    )


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.6,
            "num_predict": 220,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=35) as resp:
            raw = resp.read().decode("utf-8")
            body = json.loads(raw)
            return body.get("response", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        detail = raw or str(exc)
        status = exc.code if 400 <= exc.code < 600 else 502
        if exc.code == 404 and "model" in detail.lower():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{OLLAMA_MODEL}' not found in Ollama. Run: ollama pull {OLLAMA_MODEL}",
            ) from exc
        raise HTTPException(status_code=status, detail=f"Ollama HTTP error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama connection failed at {OLLAMA_URL}. "
                "Ensure Ollama is running and reachable from this app process."
            ),
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Invalid response from Ollama.") from exc


def parse_candidates(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("sentences"), list):
            return [str(x).strip() for x in parsed["sentences"] if str(x).strip()]
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and isinstance(parsed.get("sentences"), list):
                return [str(x).strip() for x in parsed["sentences"] if str(x).strip()]
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    lines = [ln.strip(" -0123456789.\t") for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def normalize_sentence(s: str) -> str:
    cleaned = re.sub(r"\s+", " ", s).strip().strip('"').strip("'")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def is_prompt_leak(sentence: str) -> bool:
    low = sentence.lower()
    blocked = (
        "json array",
        "return only",
        "bullet points",
        "numbering",
        "task:",
        "requirements:",
        "context paragraph",
    )
    return any(token in low for token in blocked)


def rank_and_filter(candidates: List[str], terms: List[str], count: int) -> tuple[List[str], int]:
    unique: List[str] = []
    seen = set()
    for c in candidates:
        n = normalize_sentence(c)
        key = n.lower()
        if key and key not in seen:
            unique.append(n)
            seen.add(key)

    valid = [
        s
        for s in unique
        if MIN_SENTENCE_LEN <= len(s) <= MAX_SENTENCE_LEN and not is_prompt_leak(s)
    ]
    dropped = max(0, len(unique) - len(valid))

    def score(sentence: str) -> int:
        low = sentence.lower()
        return sum(1 for t in terms if t in low)

    ranked = sorted(valid, key=score, reverse=True)
    return ranked[:count], dropped


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model": OLLAMA_MODEL})


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/recommend-next-sentences", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    context_text = req.context_text.strip()[-1200:]
    if len(context_text) < 30:
        raise HTTPException(status_code=400, detail="Please provide more context text.")

    terms = extract_terms(context_text)
    prompt = build_prompt(context_text=context_text, candidate_count=req.candidate_count, terms=terms)
    raw = call_ollama(prompt)
    parsed = parse_candidates(raw)
    sentences, dropped = rank_and_filter(parsed, terms=terms, count=req.candidate_count)
    if len(sentences) < 3:
        retry_prompt = (
            prompt
            + "\nRegenerate. Do not repeat any previous candidates. "
            + "Return only JSON object with key `sentences`."
        )
        retry_raw = call_ollama(retry_prompt)
        retry_parsed = parse_candidates(retry_raw)
        merged = parsed + retry_parsed
        sentences, retry_dropped = rank_and_filter(merged, terms=terms, count=req.candidate_count)
        dropped += retry_dropped

    if not sentences:
        raise HTTPException(status_code=422, detail="No valid suggestions generated. Please try again.")

    return RecommendResponse(
        sentences=sentences,
        meta={
            "model": OLLAMA_MODEL,
            "requested": req.candidate_count,
            "returned": len(sentences),
            "filtered": dropped,
            "terms": terms,
        },
    )
