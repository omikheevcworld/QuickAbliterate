# Abliteration pipeline image: NGC PyTorch (aarch64, CUDA 13, sm_121) + pinned deps.
# Build once: docker build -t quick-abliterate:26.09 .   (a few minutes; pure disk/CPU)
# Scripts are mounted at runtime; only the Python dependencies are baked in.
FROM nvcr.io/nvidia/pytorch:26.09-py3

COPY scripts/requirements.txt /tmp/requirements.txt
RUN pip install --no-input -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
