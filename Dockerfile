# Minimal reproducible environment for evaluating a trained checkpoint.
#
# Data and checkpoints are deliberately NOT copied into the image: both are
# gitignored and neither belongs in a container layer. Mount them at run time.
#
# Build:
#   docker build -t team-sigmoid .
#
# Run (paths are the Jupyter workspace layout):
#   docker run --rm \
#     -v /sdb-disk/notebooks/team12/team-Sigmoid/data:/app/data:ro \
#     -v /sdb-disk/notebooks/team12/team-Sigmoid/checkpoints:/app/checkpoints:ro \
#     team-sigmoid \
#     --checkpoint checkpoints/h32_dropout/best.pt \
#     --hidden_size 32 --dropout 0.3
#
# Note: requirements.txt pins the CUDA 12.8 torch wheels, so this image is
# large (~3 GB) even though it runs on CPU. That is deliberate - the pins are
# what makes the run reproducible.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/

# evaluate.py does `from model import TemporalRiskModel`, a sibling import that
# only resolves when src/model is itself on the path.
ENV PYTHONPATH=/app/src/model:/app/src/eval

ENTRYPOINT ["python", "src/model/evaluate.py"]
CMD ["--help"]
