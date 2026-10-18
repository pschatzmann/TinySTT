# Training / export environment for TinySTT (CPU by default).
#   docker build -t tinystt-research research
#   docker run --rm -v $PWD:/work -w /work tinystt-research \
#     python research/train.py --train data/train-clean-100.tsv --dev data/dev-clean.tsv --out runs/base
# GPU: build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
# and run with --gpus all.
FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
