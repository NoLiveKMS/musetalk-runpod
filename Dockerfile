# MuseTalk Production Dockerfile for RunPod Serverless
# High-quality real-time avatar video generation
# Version: 1.1 - Fixed mmpose dependencies

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    build-essential \
    python3-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone MuseTalk repository
RUN git clone https://github.com/TMElyralab/MuseTalk.git /workspace/MuseTalk

WORKDIR /workspace/MuseTalk

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy scipy cython && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod boto3 requests huggingface_hub

# Install OpenMMLab packages for MuseTalk (MMPose dependency)
RUN pip install --no-cache-dir mmengine && \
    pip install --no-cache-dir --no-build-isolation mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html && \
    pip install --no-cache-dir --no-build-isolation chumpy && \
    pip install --no-cache-dir mmdet>=3.2.0 && \
    pip install --no-cache-dir mmpose>=1.2.0

# Download model weights from HuggingFace and cache other model dependencies during build
RUN python3 -c " \
from huggingface_hub import hf_hub_download, snapshot_download; \
import os; \
os.makedirs('./models/musetalkV15', exist_ok=True); \
os.makedirs('./models/dwpose', exist_ok=True); \
os.makedirs('./models/face-parse-bisent', exist_ok=True); \
os.makedirs('./models/sd-vae', exist_ok=True); \
os.makedirs('./models/whisper', exist_ok=True); \
hf_hub_download(repo_id='TMElyralab/MuseTalk', filename='musetalkV15/unet.pth', local_dir='./models'); \
hf_hub_download(repo_id='TMElyralab/MuseTalk', filename='musetalkV15/musetalk.json', local_dir='./models'); \
hf_hub_download(repo_id='yzd-v/DWPose', filename='dw-ll_ucoco_384.pth', local_dir='./models/dwpose'); \
hf_hub_download(repo_id='yzd-v/DWPose', filename='rtmdet_m_8xb64_coco-lvis.pth', local_dir='./models/dwpose'); \
hf_hub_download(repo_id='ManyOtherFunctions/face-parse-bisent', filename='79999_iter.pth', local_dir='./models/face-parse-bisent'); \
hf_hub_download(repo_id='ManyOtherFunctions/face-parse-bisent', filename='resnet18-5c106cde.pth', local_dir='./models/face-parse-bisent'); \
snapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir='./models/sd-vae'); \
snapshot_download(repo_id='openai/whisper-tiny', local_dir='./models/whisper'); \
" || echo "Model download failed during build, fallback will handle it"

# Pre-download face-detection s3fd checkpoint to prevent runtime download latency
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth || \
    echo "S3FD checkpoint download failed"

# Copy handler
COPY handler.py /workspace/MuseTalk/handler.py

# Set Python path
ENV PYTHONPATH="/workspace/MuseTalk"

# Run handler
CMD ["python", "-u", "/workspace/MuseTalk/handler.py"]
