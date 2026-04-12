# Backend inference API. Pip + Debian slim (reliable on Render; avoids conda solver issues).
#   docker build -t mindmirror-api .
#
# Refresh locked deps from conda env `cogload` (optional):
#   conda activate cogload
#   pip freeze > requirements-docker.txt
#   (then trim training-only packages if the image/build is too heavy)
#
# Trained weights: models/trained/ is gitignored — mount a disk + MODELS_DIR or bake in CI.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# MediaPipe tasks load libGLESv2.so.2 / EGL — slim images omit these unless installed explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    libegl1-mesa \
    libgles2-mesa \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY backend ./backend
COPY cognitive_load ./cognitive_load
COPY features ./features
COPY models ./models

RUN mkdir -p models/mediapipe \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/mediapipe/face_landmarker.task')"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
