# CoTracker3 inference environment.
# Host driver 550.x supports up to CUDA 12.4, so we use the cu124 PyTorch image.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/workspace/.torch_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 git wget \
    && rm -rf /var/lib/apt/lists/*

# matplotlib<3.9: cotracker/utils/visualizer.py uses cm.get_cmap, removed in 3.9.
RUN pip install --no-cache-dir \
        "imageio[ffmpeg]" imageio-ffmpeg av \
        "matplotlib<3.9" flow_vis tqdm \
        opencv-python-headless mediapy einops \
        fastapi "uvicorn[standard]" python-multipart

# Qwen2.5-VL for the batch's VLM quality check (scripts/vlm_check.py). torch is pinned so pip never swaps the image's CUDA 12.4 build, and
# transformers 5 needs torch >= 2.5.
RUN pip install --no-cache-dir "torch==2.4.1" \
        "transformers>=4.49,<5" accelerate "qwen-vl-utils>=0.0.8"

# The repo itself is bind-mounted at /workspace at runtime (see docker/run.sh),
# so code edits on the host are picked up without rebuilding.
ENV PYTHONPATH=/workspace
WORKDIR /workspace

CMD ["bash"]
