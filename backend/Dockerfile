FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# 🔥 Fix treo timezone
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cache layer tốt nhất)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Thiết lập thư mục cache cho HuggingFace
ENV HF_HOME=/shared_storage/hf_cache

EXPOSE 8000
EXPOSE 8001

# Command is overridden in docker-compose.yml based on service (train vs infer)
CMD ["echo", "Please specify command in docker-compose.yml"]