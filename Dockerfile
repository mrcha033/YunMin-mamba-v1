########################  Build-time arguments  ########################
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.2.2
ARG CU_TAG=cu118
ARG PYVER=cp310
ARG MAMBA_WHL=mamba_ssm-2.2.4+cu11torch2.2cxx11abiFALSE-${PYVER}-${PYVER}-linux_x86_64.whl

########################  Base image  ##################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG TORCH_VERSION CU_TAG PYVER MAMBA_WHL

########################  Env & flags  #################################
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TOKENIZERS_PARALLELISM=false \
    MAX_JOBS=4 \
    MAMBA_SKIP_CUDA_BUILD=TRUE
# ---------------------------------------------------------------------

########################  System packages  #############################
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3 python3-pip python-is-python3 \
        git wget curl unzip vim cmake ninja-build pkg-config \
        libgl1 libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

########################  Python deps  #################################
WORKDIR /app
COPY requirements.txt .

# 1) pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip 'setuptools<70' wheel packaging numpy

# 2) PyTorch & 친구들 – 공식 CU121 인덱스
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}+${CU_TAG}" \
        "torchvision==0.17.2+${CU_TAG}" "torchaudio==2.2.2+${CU_TAG}" \
        --index-url https://download.pytorch.org/whl/${CU_TAG}

# 3) 프로젝트 의존성 + mamba-ssm 휠
ENV MAMBA_SKIP_CUDA_BUILD=TRUE
RUN pip install --no-cache-dir -r requirements.txt
RUN wget -q https://github.com/state-spaces/mamba/releases/download/v2.2.4/${MAMBA_WHL} \
        && pip install --no-cache-dir ${MAMBA_WHL} \
        && rm ${MAMBA_WHL} 
RUN pip install --no-cache-dir "transformers==4.42.4"

# 4) 앱 소스 복사
COPY . .

########################  진단 스크립트  ###############################
RUN python - <<'PY'
import torch, os
print("✅ torch:", torch.__version__, "| CUDA OK:", torch.cuda.is_available())
from mamba_ssm import MambaLMHeadModel
print("✅ mamba-ssm import 완료")
PY

########################  기본 엔트리포인트  ###########################
CMD ["python", "train_mamba.py"]
