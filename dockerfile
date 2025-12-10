# base Image
# using Ubuntu 22.04 (LTS) — stable and compatible with Python 3.10
FROM ubuntu:22.04

# disable all interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ensure Python prints appear immediately (no buffering)
ENV PYTHONUNBUFFERED=1

# build-time argument to select hardware target:
# - cpu-x86 (default)
# - cuda-x86
# - jetson
# - pi
ARG TARGET=cpu-x86


# install Python 3.10 from deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    python3-pip python3-setuptools python3-wheel


# set Python 3.10 as default python / python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


# install system dependencies:
# - build-essential, cmake → required for native modules
# - OpenCV dependencies → codecs, image formats, V4L2 camera support
# - GTK3 + X11 libs → GUI preview support
# - libgpiod → GPIO access for Linux hardware
# - libcurl → HTTP requests in C++ portion
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget pkg-config \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev x11-apps libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    libgomp1 libglib2.0-0 libgpiod-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# application directory
WORKDIR /app

# upgrade Python packaging tools
RUN python3.10 -m pip install --upgrade pip wheel setuptools

# install base Python packages:
# - numpy pinned for dlib compatibility
# - dlib-bin → avoids compiling heavy C++ dlib manually
# - opencv-python → camera + image processing
# - face-recognition → face embeddings + matching
# - fastapi + uvicorn → API server
RUN python3.10 -m pip install numpy==1.24.3 cmake dlib-bin opencv-python face-recognition fastapi uvicorn python-multipart

# install GPU PyTorch for x86 CUDA (cu118)
RUN if [ "$TARGET" = "cuda-x86" ]; then \
      python3.10 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118 ; \
    fi

# install Jetson ARM PyTorch (L4T wheels, not from PyPI)
# jetson wheels MUST match JetPack version (here: JP 5.x → v512)
RUN if [ "$TARGET" = "jetson" ]; then \
      python3.10 -m pip install \
        https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0-cp310-cp310-linux_aarch64.whl \
        https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torchvision-0.16.0-cp310-cp310-linux_aarch64.whl \
        torchaudio==2.1.0 ; \
    fi

# CPU-only PyTorch (used for Raspberry Pi & standard CPU systems)
RUN if [ "$TARGET" = "pi" ] || [ "$TARGET" = "cpu-x86" ]; then \
      python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
    fi

COPY . .

RUN mkdir -p /app/logs /app/data /app/backup

CMD ["bash", "-c", "tail -f /dev/null"]