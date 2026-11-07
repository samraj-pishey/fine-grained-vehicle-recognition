# Production image: Streamlit app + ONNX Runtime CPU inference. No training packages, no dataset, no checkpoints.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ORT_INTRA_OP_THREADS=2 \
    ORT_INTER_OP_THREADS=1 \
    MAX_CONCURRENT_INFERENCES=1 \
    INFERENCE_TIMEOUT_S=60 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/home/app/srv/src

# Run as an unprivileged user (uid 1000)
RUN useradd -m -u 1000 app
WORKDIR /home/app/srv

COPY --chown=app:app requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=app:app app.py pyproject.toml LICENSE THIRD_PARTY_NOTICES.md ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app configs ./configs
COPY --chown=app:app src ./src
COPY --chown=app:app reports/results ./reports/results
# Deployment bundle (manifest + ONNX models + class mapping) - only permitted, checksum-pinned artefacts.
# If absent, the app fetches the pinned bundle once at startup (configs/deployment.yaml).
COPY --chown=app:app artifacts/deployment ./artifacts/deployment

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
