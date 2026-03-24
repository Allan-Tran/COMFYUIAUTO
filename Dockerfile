ARG BASE_IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204
FROM ${BASE_IMAGE}

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
    git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use.git /comfyui/ComfyUI/custom_nodes/ComfyUI-Easy-Use && \
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
    /comfyui/ComfyUI/custom_nodes/ComfyUI-Easy-Use/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-Serving-Toolkit/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-MediaMixer/requirements.txt \
      /comfyui/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; do \
      if [ -f "$req" ]; then python -m pip install -r "$req"; fi; \
    done

# ComfyUI requirements can pull newer torch/cu13 builds. Pin back to cu128 for driver compatibility.
RUN python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 && \
        python -m pip install --upgrade "chardet<6"

RUN python -m pip install --upgrade sageattention

# Verify Torch has custom_op and is pinned to CUDA 12.8 build.
# Note: torch.cuda.is_available() can be false during docker build when no GPU is mounted.
RUN python -c "import torch; v=torch.version.cuda; assert hasattr(torch.library, 'custom_op'), 'torch.library.custom_op missing'; assert v is not None, 'Torch was built without CUDA support'; assert str(v).startswith('12.8'), f'Expected CUDA 12.8 build, got {v}'; assert torch.__version__.startswith('2.9.1'), f'Expected torch 2.9.1, got {torch.__version__}'"

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
    /comfyui/ComfyUI/output \
    /comfyui/ComfyUI/input && \
    ln -sfn /comfyui/ComfyUI/output /comfyui/output

COPY blank_test.png /comfyui/ComfyUI/input/blank_test.png
COPY rp_handler.py /comfyui/rp_handler.py

CMD ["python", "-u", "rp_handler.py"]
