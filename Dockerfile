# Backend inference service only. Conda env via micromamba (conda-forge).
#   docker build -t mindmirror-api .
#
# Trained weights: models/trained/ is gitignored — copy/mount at deploy or use a persistent disk.

FROM mambaorg/micromamba:2-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-inference.yml /tmp/environment-inference.yml
RUN micromamba env create -f /tmp/environment-inference.yml -y \
    && micromamba clean -a -y

ENV PYTHONPATH=/app \
    PATH="/opt/conda/envs/inference/bin:$PATH"

COPY --chown=$MAMBA_USER:$MAMBA_USER backend ./backend
COPY --chown=$MAMBA_USER:$MAMBA_USER cognitive_load ./cognitive_load
COPY --chown=$MAMBA_USER:$MAMBA_USER features ./features
COPY --chown=$MAMBA_USER:$MAMBA_USER models ./models

RUN mkdir -p models/mediapipe \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/mediapipe/face_landmarker.task')"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
