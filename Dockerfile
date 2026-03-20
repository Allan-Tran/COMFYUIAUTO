FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFYUI_ROOT=/comfyui \
    COMFYUI_DIR=/comfyui/ComfyUI \
    COMFYUI_PORT=8188

RUN apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    ffmpeg \
    git \
    git-lfs \
    ca-certificates \
    curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git

RUN git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git /comfyui/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git /comfyui/ComfyUI/custom_nodes/ComfyUI-GGUF && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone --depth=1 https://github.com/matan1905/ComfyUI-Serving-Toolkit.git /comfyui/ComfyUI/custom_nodes/ComfyUI-Serving-Toolkit && \
    git clone --depth=1 https://github.com/DoctorDiffusion/ComfyUI-MediaMixer.git /comfyui/ComfyUI/custom_nodes/ComfyUI-MediaMixer && \
    git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /comfyui/ComfyUI/requirements.txt runpod requests python-telegram-bot

RUN set -eux; \
    for req in \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-Serving-Toolkit/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-MediaMixer/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; do \
      if [ -f "$req" ]; then python -m pip install -r "$req"; fi; \
    done

RUN mkdir -p /comfyui/ComfyUI/models/checkpoints \
    /comfyui/ComfyUI/models/vae \
    /comfyui/ComfyUI/models/loras \
    /comfyui/ComfyUI/models/controlnet \
    /comfyui/ComfyUI/models/clip \
    /comfyui/ComfyUI/models/clip_vision \
    /comfyui/ComfyUI/models/unet \
    /comfyui/ComfyUI/models/diffusion_models \
    /comfyui/ComfyUI/models/text_encoders \
    /comfyui/ComfyUI/models/upscale_models \
    /comfyui/ComfyUI/output && \
    ln -sfn /comfyui/ComfyUI/output /comfyui/output

COPY rp_handler.py /comfyui/rp_handler.py

CMD ["python", "-u", "rp_handler.py"]
