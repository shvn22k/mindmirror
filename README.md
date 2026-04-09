# Mind Mirror

**Mind Mirror** estimates **cognitive load** from ordinary webcam-style face video. The system turns each short clip into facial landmarks, derives interpretable behavior features (eyes, gaze, head, mouth), aggregates them to one vector per clip, and runs trained models to produce a **continuous score** (regression) and a **coarse band** (LOW / MEDIUM / HIGH classification). This repository is the full pipeline—from dataset preparation through training to a small HTTP API and browser test UI—not a paper-style artifact dump.

---

## Cognitive load estimation (in this product)

**Cognitive load** describes how much mental effort someone is investing in a task. In human factors it is often discussed alongside dimensions such as those in the **NASA-TLX** (e.g. mental demand, temporal demand, effort). Mind Mirror does not run a full questionnaire; it learns a mapping from **observable face behavior over time** to a **scalar target** that reflects **perceived mental demand** as annotated in the training data.

Outputs are **estimates**, not clinical or workplace certifications. They are useful for prototyping analytics, feedback loops, and research-style exploration when paired with appropriate ethics, consent, and validation for your use case.

---

## Data foundation: AVCAffe

We build and evaluate on **AVCAffe** (*Audio-Visual Cognitive load and Affect for remote work*): a large multimodal corpus of remote-work-style recordings with continuous ratings on several constructs, including dimensions aligned with workload and affect.

**Citation (please use in any derivative work or publication):**

> Pritam Sarkar, Aaron Posen, Ali Etemad. *AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work.*  
> Proceedings of the **AAAI** Conference on Artificial Intelligence, 37(1), 76–85, 2023.  
> DOI: [10.1609/aaai.v37i1.25078](https://doi.org/10.1609/aaai.v37i1.25078) — Preprint: [arXiv:2205.06887](https://arxiv.org/abs/2205.06887)  
> Project / download pointers: [pritamsarkar.com/AVCAffe](https://pritamsarkar.com/AVCAffe/) and the **Queen’s Borealis Dataverse** entry (dataset DOI **10.5683/SP3/PSWY62**). Always confirm the **current license** on the official download page before use.

**How the data is obtained and used here**

- AVCAffe is distributed by the authors / institutional repository under **research-oriented terms** (check the current license on the official download page before redistributing or using commercially).
- We use **video clips** and **tabular ratings** supplied with the corpus. Audio and other rating dimensions exist in the full dataset; **this codebase trains on the mental-demand scale exposed as `mental_demand.txt`** (comma-separated `participant_task, value` rows), joined to clips by the **`{participant}_{task}`** key (see `cognitive_load/dataset.py`, `cognitive_load/load_labels.py`).
- **Clip naming** is assumed to follow the pattern `aiim001_task_3_clip_012.avi` so that all clips from the same participant–task session share one label row, which matches how labels are keyed in the provided files.

**Why we picked this data and target**

- **Ecological validity:** Recordings resemble **video calls / remote work**, which matches how Mind Mirror is meant to be used (single face, frontal-ish webcam).
- **Dense supervision:** Continuous scores give a **regression** target; we also derive **ordered classes** for product-friendly buckets (see below).
- **Scale:** Enough clips to train **gradient-boosted trees** and optional **TabNet** without hand-engineering only a handful of examples.

We intentionally **do not** claim to use every modality or label in AVCAffe—only the slice needed for a **video→mental-demand** model that powers the product prototype.

---

## Design intent: what the model is allowed to see

| Choice | Rationale |
|--------|-----------|
| **Face video only** | Aligns with deployment (webcam), avoids requiring extra sensors. |
| **Landmarks, not raw pixels** | Stable, lightweight representation; MediaPipe gives **478** points per frame in normalized image space. |
| **Temporal subsampling (`sample_every=3`)** | Cuts compute while preserving slow dynamics (blinks, head motion, gaze drift). Default **25 FPS** nominal video → **effective FPS = 25/3** for sequence statistics. |
| **Clip-level aggregation** | The label is **session/task-level** in the dataset join we use; the model is trained on **one feature row per clip**, not per-frame labels. Inference mirrors that: a **short segment** of video → one prediction. |
| **Behavioral features** | EAR/MAR, gaze proxies, head pose proxies correlate with fatigue, attention, and motor activity—plausible indicators under load, not definitive biomarkers. |

---

## System architecture (textual diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MIND MIRROR (PRODUCT)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  Batch pipeline  │       │  Inference API   │       │  MindMirror UI   │
│  (offline train) │       │  FastAPI + model │       │  Vite + React    │
└────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
         │                          │                          │
         │                          │    multipart video       │
         │                          │◄─────────────────────────┤
         │                          │         JSON scores      │
         │                          ├─────────────────────────►│
         ▼                          ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         SHARED CORE (Python packages)                       │
│  cognitive_load/  features/  training/  backend/pipeline.py               │
└────────────────────────────────────────────────────────────────────────────┘

                         BATCH PIPELINE (DETAIL)

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ AVCAffe      │     │ Landmark     │     │ Per-frame    │     │ Clip CSV     │
  │ videos +     │────►│ extraction   │────►│ features     │────►│ aggregation  │
  │ labels file  │     │ (MediaPipe)  │     │ (pandas)     │     │ clip_features│
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                         │
                                                                         ▼
                                                                ┌──────────────┐
                                                                │ Train / eval │
                                                                │ + preprocess │
                                                                │   .joblib    │
                                                                └──────────────┘

                         RUNTIME INFERENCE (DETAIL)

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ Upload /     │     │ Same         │     │ Same         │     │ Scaler +     │
  │ webcam chunk │────►│ landmark +   │────►│ clip         │────►│ XGBoost      │
  │ (video file) │     │ frame feats  │     │ aggregate    │     │ (reg + cls)  │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## End-to-end pipeline (deep dive)

Below is the **actual** processing order implemented in this repo, with file pointers.

### 1. Ingest and align labels

- **Inputs:** Root folder of videos (e.g. `data/processed/avcaffe/...`) laid out as `participant/task/*.avi`, plus `labels/mental_demand.txt`.
- **Logic:** `AVCaffeDataset` walks the tree, maps each file to `participant_task` via `get_video_label`, and **yields only clips that have a label** (`cognitive_load/dataset.py`).
- **Output (conceptual):** A list of `(video_path, mental_demand_score)` pairs.

**Pipeline visualization**

```
mental_demand.txt          video tree
     │                          │
     │    key: aiim001_task_1    │  …/aiim001/task_1/*_clip_*.avi
     └────────────┬─────────────┘
                  ▼
           join on participant_task
                  ▼
           labeled clip list
```

---

### 2. Landmark extraction

- **Script:** `scripts/run_extraction.py`
- **Engine:** Google **MediaPipe Face Landmarker** (task file under `models/mediapipe/`; auto-download on first use if missing).
- **Mode:** `RunningMode.IMAGE` per decoded frame; **one face**; 478 `(x, y, z)` landmarks per frame (normalized coordinates).
- **Sampling:** `read_video_frames(..., sample_every=3)` so every third frame is processed—consistent for training and inference (`cognitive_load/landmark_extractor.py`, `cognitive_load/video_utils.py`).
- **Output:** NumPy arrays saved per clip (e.g. `*.npy`), shape `(T, 478, 3)`.

```
video ──► OpenCV decode ──► every 3rd frame ──► FaceLandmarker ──► (T,478,3) .npy
```

---

### 3. Per-frame feature engineering

- **Script:** `scripts/run_feature_extraction.py` (uses `features/feature_extractor.py`).
- **Per frame**, from landmarks:

| Module | Quantities | Role |
|--------|------------|------|
| **Eyes** (`eye_features.py`) | EAR left/right/mean (6-point eye ratio), `is_blink` | Blinking frequency and closure relate to attention and fatigue literature. |
| **Head** (`head_features.py`) | Nose-based position `(head_x,y,z)`, heuristic **pitch/yaw/roll** | Gross pose and movement; not full solvePnP solve, but stable relative signal. |
| **Gaze** (`gaze_features.py`) | Iris center vs eye corners → **gaze_x, gaze_y** (normalized) | Proxy for where attention is directed in the frame. |
| **Mouth** (`mouth_features.py`) | MAR (mouth aspect ratio), `is_yawn` | Wide-open mouth episodes (yawns) as a slow physiological cue. |

- **Output:** Parquet (or similar) **one row per sampled frame** per clip, with `clip_id`, timestamps, and columns above.

---

### 4. Clip-level aggregation (the ML input row)

- **Script:** `scripts/aggregate_features.py`
- **Core:** `features/clip_aggregate.py` — collapses the frame table to **one dictionary / CSV row per clip**.

**Effective time base:** `effective_fps = fps / sample_every` (default **25/3**). All rate-based stats (blinks/min, yawns/min) use this.

**Families of clip features (conceptual groups):**

```
EAR sequence ──────► mean, std, min, max
                  └─► blink_count, blink_rate_per_min, avg/max blink duration (ms)

Head positions ───► movement total, variance, range in x/y
Head pose cols ───► mean/std of pitch, yaw, roll

Gaze (x,y) seq ───► dispersion, std_x, std_y
                  └─► fixation_count, avg fixation duration, fixation_ratio
                  └─► mean/std of gaze_x, gaze_y

MAR sequence ─────► mean, std, max
                  └─► yawn_count, yawn_rate_per_min, yawn duration stats

Metadata (excluded from X): clip_id, n_frames, duration_seconds, label
```

- **Join:** Aggregation can attach **`label`** from `mental_demand.txt` when `clip_id` parses to a known key (`aggregate_features.py`).
- **Output:** `clip_features.csv` — the **master training table** (`training/data_prep.load_features`).

---

### 5. Training data preparation

- **Module:** `training/data_prep.py`
- **Cleaning:** Rows without `label` dropped; feature matrix `np.nan_to_num`.
- **Splits:** Stratified by **discretized** class (see below): 70% / 15% / 15% train / val / test (via two-stage `train_test_split`, `random_state=42`).
- **Scaling:** `StandardScaler` **fit on train only**, applied to val/test; same scaler serialized to `preprocess.joblib` for inference.
- **Regression target:** Raw `label` (continuous mental demand in dataset encoding, observed range ~0–21 in our runs).
- **Classification target (3-way):** Derived thresholds on the same continuous score:

  | Class | Rule (default 3-class) |
  |-------|-------------------------|
  | LOW | score ≤ 7 |
  | MEDIUM | 8–14 |
  | HIGH | ≥ 15 |

  (Configurable 5-class variant exists in code for experiments.)

---

### 6. Model training and ensembling

- **Script:** `scripts/train_model.py`
- **Base learners** (`training/models.py`): **XGBoost**, **LightGBM**, **CatBoost**, optional **PyTorch TabNet** (`--skip-tabnet` for faster runs).
- **Tasks:** Both **regression** (MAE, RMSE, R²) and **classification** (accuracy, macro-F1).
- **Ensemble:** `training/ensemble.py` — **weighted average** of base models; weights chosen by a **validation-set grid search** (`optimize_ensemble_weights`) minimizing MAE (regression) or maximizing macro-F1 (classification).
- **Artifacts:** Model files under `models/trained/` (`.joblib`, `.cbm`, TabNet dirs), plus **`preprocess.joblib`** (feature column order, class names, scaler). CatBoost logs may go under `artifacts/catboost/`.
- **Reporting:** Figures and `results/training_report.md` generated when you run training (that folder is gitignored except `.gitkeep`; regenerate locally).

Example headline numbers from a full run (including TabNet + ensemble) are reflected in `results/training_report.md` in a trained workspace—ensemble slightly edges single models on both tasks.

---

### 7. Inference service (Mind Mirror runtime)

- **App:** `backend/main.py` — **FastAPI**, CORS enabled for browser clients.
- **Pipeline:** `backend/pipeline.py` — `CognitiveLoadPredictor`:
  1. Write upload to temp path (correct extension: `.mp4`, `.webm`, …).
  2. **Same** landmark path and `sample_every` as training defaults.
  3. Frame features → `aggregate_clip_features`.
  4. Vector aligned to `feature_names`; **transform** with saved scaler.
  5. **Regression** + **classification** (default: `xgboost_regression.joblib` / `xgboost_classification.joblib`; override via env).

**Endpoints**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Process up, models loaded |
| POST | `/predict` | Multipart field `file` = short video |

---

### 8. Frontend (MindMirror)

- **Path:** `frontend/` — Vite + React + TypeScript (live analysis, dashboard, marketing pages).
- **API wiring:** In development, Vite proxies **`/inference`** → FastAPI (default `http://127.0.0.1:8000`). The analysis flow calls **`POST /inference/predict`** (same as **`POST /predict`** on the Python app). Override with **`VITE_INFERENCE_API_URL`** when the browser must talk to the API directly (ensure CORS).
- **Legacy harness:** `legacy-frontend/index.html` — static upload / MediaRecorder test page (optional QA).

---

## Repository layout (quick map)

| Path | Responsibility |
|------|----------------|
| `cognitive_load/` | Dataset walking, labels, video I/O, MediaPipe wrapper |
| `features/` | Landmark indices, per-frame metrics, clip aggregation |
| `training/` | Data prep, model wrappers, metrics, ensemble |
| `scripts/` | CLI: extract → frame features → aggregate → train |
| `backend/` | HTTP API + predictor wiring |
| `frontend/` | MindMirror web app (Vite + React) |
| `legacy-frontend/` | Static HTML test harness (optional) |
| `data/` | See `data/README.md`; `processed/` is local only |
| `models/trained/` | Your checkpoints + `preprocess.joblib` (local only) |

---

## Setup and run (summary)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Place or generate `data/processed/...` per `data/README.md`.
2. Run extraction → feature extraction → aggregation → `train_model.py`.
3. `uvicorn backend.main:app --port 8000`
4. In another terminal: `cd frontend`, `npm install`, `npm run dev` (default [http://localhost:5173](http://localhost:5173)). Ensure the FastAPI server is running so `/inference` can be proxied to it.
5. Copy `frontend/env.example` to `frontend/.env` if you need custom ports or **`VITE_INFERENCE_API_URL`**.

---

## References

1. Sarkar, P.; Posen, A.; Etemad, A. **AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work.** *AAAI* 2023. [https://doi.org/10.1609/aaai.v37i1.25078](https://doi.org/10.1609/aaai.v37i1.25078) — arXiv: [2205.06887](https://arxiv.org/abs/2205.06887). Dataset hosting: Borealis Dataverse DOI [10.5683/SP3/PSWY62](https://doi.org/10.5683/SP3/PSWY62).  
2. Google **MediaPipe** Face Landmarker — [https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)  
3. Hart, S. G.; Staveland, L. E. **Development of NASA-TLX (Task Load Index).** In *Human Mental Workload* (1988). (Conceptual background for *mental demand* as a construct.)

---

*Mind Mirror — estimate cognitive load from face video using AVCAffe-style supervision and a transparent behavioral feature pipeline.*
