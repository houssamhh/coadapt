# ============================================================
# ACSOS 2026 Artifact — CoAdapt Pipeline
# ============================================================
# Two conda environments:
#   - opencood        (Python 3.7, CUDA 11.7, OpenCOOD inference)
#   - decision_making (Python 3.10, LLM selection)
#
# Base image: Debian 12 (Bookworm)
# Note: conda envs pin their own torch/CUDA versions via the yml files.
#
# Build:
#   docker build -t coadapt .
#
# Run (GPU required):
#   docker run --gpus all --rm \
#     -v /path/to/dataset:/workspace/data \
#     -v /path/to/models:/workspace/trained-models \
#     -v /path/to/results:/workspace/results \
#     coadapt \
#     --dataset_root /workspace/data/test \
#     --llm_model google/gemma-3-27b-it \
#     --model_intermediate trained-models/Models/pointpillar_attentive_fusion \
#     --model_early        trained-models/Models/pixor_early_fusion \
#     --model_late         trained-models/Models/pointpillar_late_fusion \
#     --output_dir /workspace/results
# ============================================================

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# ── system deps ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git \
        gcc g++ build-essential \
        libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── miniforge (conda-forge only, no ToS, libmamba solver built-in) ───────────
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# ── copy repo (code only — data/models/results are mounted at runtime) ───────
WORKDIR /workspace
COPY coadapt/       ./coadapt/
COPY lidar_processing/       ./lidar_processing/
COPY evaluation/    ./evaluation/
COPY opv2v_data_dumping ./opv2v_data_dumping/
COPY opencood_environment.yml .
COPY llm_environment.yml      .
COPY trained-models/ ./trained-models/

# ── opencood environment ──────────────────────────────────────
# Create env manually — the yml targets a CUDA 11.7 host and cannot be used
# directly in this base image (spconv-cu117/cumm-cu117 require CUDA 11.7 libs,
# timm==0.9.12 breaks on Python 3.7).
RUN conda create -n opencood python=3.7 -y

RUN conda run -n opencood conda install -y \
        pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 \
        cudatoolkit=10.2 -c pytorch

RUN conda run -n opencood pip install --no-cache-dir spconv-cu113

RUN conda run -n opencood pip install --no-cache-dir \
        cython==3.0.12 pyyaml==6.0.1 tqdm==4.65.2 shapely==2.0.0 \
        "timm==0.6.13" \
        opencoodx==0.1.19 easydict==1.13 einops==0.6.1 \
        tensorboardx==2.6.2.2 pypcd==0.1.1 pyquaternion==0.9.9 \
        msgpack==1.0.5 addict==2.4.0 \
        open3d==0.17.0 opencv-python-headless \
        numpy==1.21.6 scipy==1.7.3 scikit-learn==1.0.2 \
        matplotlib==3.4.2 ninja==1.11.1.4 setuptools==60.2.0

# ── build Cython extensions in-place ─────────────────────────
# ENV PYTHONPATH=/workspace:$PYTHONPATH
# RUN conda run -n opencood python opencood/utils/setup.py build_ext --inplace

# ── decision_making environment ───────────────────────────────
RUN conda env create -f llm_environment.yml

# ── clean conda cache to keep image lean ─────────────────────
RUN conda clean -afy

CMD ["/bin/bash"]
