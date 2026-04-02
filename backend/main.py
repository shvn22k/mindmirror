"""
Inference API (FastAPI).

Run from project root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Env:
    MODELS_DIR          default models/trained
    REGRESSION_MODEL    default xgboost_regression.joblib
    CLASSIFICATION_MODEL default xgboost_classification.joblib
    INFERENCE_FPS       default 25
    INFERENCE_SAMPLE_EVERY default 3
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import CognitiveLoadPredictor, predict_uploaded_video

_predictor: CognitiveLoadPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = CognitiveLoadPredictor.from_env()
    yield
    _predictor = None


app = FastAPI(
    title="Cognitive load inference",
    description="Upload a face video; returns regression score and LOW/MEDIUM/HIGH classification.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    ok = _predictor is not None
    return {"status": "ok" if ok else "unavailable", "models_loaded": ok}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Video file (e.g. mp4, avi)")):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = ".mp4"
    lower = file.filename.lower()
    for ext in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
        if lower.endswith(ext):
            suffix = ext
            break

    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e}") from e

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        return predict_uploaded_video(data, suffix=suffix, predictor=_predictor)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
