from functools import lru_cache
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from explain import load_model as load_explainer_model
from explain import predict_probabilities, split_sentences
from predict_distilbert import load_model as load_predictor_model
from predict_distilbert import predict


app = FastAPI(title="Fake News Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ArticleRequest(BaseModel):
    title: str = Field(default="", description="Optional article title")
    text: str = Field(..., min_length=1, description="Article body text")
    max_length: int = Field(default=256, ge=32, le=512, description="Tokenizer max length")


class ExplainRequest(ArticleRequest):
    top_k: int = Field(default=5, ge=1, le=20, description="Number of influential sentences to return")
    max_sentences: int = Field(default=12, ge=1, le=50, description="Maximum sentences to analyze")


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PUBLIC_DIR = BASE_DIR / "public"
UI_DIR = BASE_DIR / "ui"
UI_INDEX = PUBLIC_DIR / "index.html"


def merge_article_text(title: str, text: str) -> str:
    parts = [title.strip(), text.strip()]
    combined = "\n\n".join(part for part in parts if part)
    if not combined:
        raise HTTPException(status_code=400, detail="Provide article text.")
    return combined


@lru_cache(maxsize=1)
def get_prediction_assets():
    if not MODEL_DIR.exists():
        raise HTTPException(status_code=500, detail="Model directory not found. Train the model first.")
    return load_predictor_model(MODEL_DIR)


@lru_cache(maxsize=1)
def get_explanation_assets():
    if not MODEL_DIR.exists():
        raise HTTPException(status_code=500, detail="Model directory not found. Train the model first.")
    return load_explainer_model(MODEL_DIR)


@app.get("/health")
def health_check():
    model_ready = MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()
    return {"status": "ok", "model_ready": model_ready}


@app.get("/")
def serve_dashboard():
    dashboard_path = UI_INDEX if UI_INDEX.exists() else UI_DIR / "index.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(dashboard_path)


@app.get("/metadata")
def model_metadata():
    metadata_path = MODEL_DIR / "training_metadata.json"
    if not metadata_path.exists():
        return {"available": False}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid metadata file: {exc}") from exc

    return {"available": True, "metadata": metadata}


@app.post("/predict")
def predict_article(request: ArticleRequest):
    text = merge_article_text(request.title, request.text)
    tokenizer, model = get_prediction_assets()
    return predict(text, tokenizer, model, request.max_length)


@app.post("/explain")
def explain_article(request: ExplainRequest):
    text = merge_article_text(request.title, request.text)
    tokenizer, model = get_explanation_assets()

    id_to_label, baseline_probs = predict_probabilities(text, tokenizer, model, request.max_length)
    predicted_id = int(baseline_probs.argmax().item())
    predicted_label = id_to_label[predicted_id]
    baseline_confidence = float(baseline_probs[predicted_id].item())

    sentences = split_sentences(text, request.max_sentences)
    impacts: list[dict[str, float | str]] = []

    for index, sentence in enumerate(sentences):
        reduced_sentences = [value for i, value in enumerate(sentences) if i != index]
        reduced_text = " ".join(reduced_sentences).strip()
        if not reduced_text:
            continue

        _, reduced_probs = predict_probabilities(reduced_text, tokenizer, model, request.max_length)
        reduced_confidence = float(reduced_probs[predicted_id].item())
        impacts.append(
            {
                "sentence": sentence,
                "confidence_drop": baseline_confidence - reduced_confidence,
            }
        )

    impacts.sort(key=lambda item: item["confidence_drop"], reverse=True)

    return {
        "label": predicted_label,
        "confidence": baseline_confidence,
        "top_sentences": impacts[: request.top_k],
        "analyzed_sentences": len(sentences),
        "truncated_for_explanation": len(split_sentences(text, 10_000)) > len(sentences),
    }
