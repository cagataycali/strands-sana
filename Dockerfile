# strands-sana — CUDA-ready inference container
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip

# Install strands-sana + GPU torch
RUN pip install "torch>=2.4" --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install strands-sana

WORKDIR /workspace
VOLUME ["/data"]

ENTRYPOINT ["python", "-c"]
CMD ["from strands_sana import sana_load_model; print(sana_load_model.original_func(model='list'))"]
