# syntax=docker/dockerfile:1
# RAGBench in one container: the Replay UI (static files) and the live API (bring-your-own-key, SSE),
# both served by FastAPI from the same origin.
#
#   docker build -t ragbench .
#   docker run --rm -p 8000:8000 ragbench        # http://localhost:8000
#
# On Render (Docker runtime), set once the service URL is known:
#   RAGBENCH_CORS_ORIGINS=["https://<service>.onrender.com"]
#   RAGBENCH_REQUIRE_HTTPS=1
#   RAGBENCH_TRUST_PROXY_HEADERS=1
# Do not set GENERATOR_API_KEY or JUDGE_API_KEY: live runs only ever use the visitor's own key.

# ---- 1. frontend ------------------------------------------------------------------------------
FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
# An empty API URL makes the UI call /api/... on its own origin.
ENV VITE_API_URL=""
RUN npm run build

# ---- 2. runtime -------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf
WORKDIR /app

# CPU-only torch first. sentence-transformers depends on torch, and the default Linux wheel from
# PyPI is a multi-GB CUDA build. By hand:
#   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

# Bake the embedding model into the image, so startup needs no download (the runtime is offline).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

# Editable install from the source tree: repo_root() resolves configs/, data/ and evaluation/
# relative to src/, which a regular wheel install into site-packages would break.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps -e . \
    && python -c "import ragbench.api.main, ragbench.retrieval.store, ragbench.retrieval.embedder"

COPY configs/ configs/
COPY data/questions.yaml data/
COPY data/corpus/manifest.yaml data/corpus/
COPY evaluation/results/ evaluation/results/
# Pre-built index (see deploy/chroma/): building it needs a headless browser for 3 of the 7 sources.
COPY deploy/chroma/ data/chroma/
COPY --from=ui /ui/dist ui/dist

# Chroma opens its SQLite file read-write, so the app user owns the index and nothing else.
RUN useradd --create-home --uid 1000 app && chown -R app:app data/chroma
USER app

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec ragbench serve --host 0.0.0.0 --port \"$PORT\""]
