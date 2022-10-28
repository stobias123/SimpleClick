FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# you'll want git for sure my dude
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
	    curl \
        libglib2.0-0 \
        software-properties-common \
        git \
        sqlite3 \
        python3 \
        python3-pip \
        python3-tk \
        libcanberra-gtk-module \
        nano \
        python3-opencv \
        python3-dev

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install matplotlib numpy pandas scipy tqdm pyyaml easydict scikit-image bridson Pillow ninja
RUN pip3 install imgaug mxboard graphviz
RUN pip3 install albumentations --no-deps
# RUN pip3 install opencv-python-headless
RUN pip3 install Cython
RUN pip3 install scikit-learn
RUN pip3 install tensorboard
RUN pip3 install qudida mmcv timm

# install torch and vision
RUN pip3 install cuda-python && \
    pip3 install \
        torch==1.12.1+cu116 \
        torchvision==0.13.1+cu116 \
        -f https://download.pytorch.org/whl/torch_stable.html

# python pip requirements
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt
