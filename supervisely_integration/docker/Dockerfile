FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git
RUN python3 -m pip install --upgrade pip

RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN git clone https://github.com/facebookresearch/co-tracker
WORKDIR co-tracker
RUN pip3 install -e .
WORKDIR /
RUN pip3 install opencv-python einops timm matplotlib moviepy flow_vis

RUN python3 -m pip install supervisely==6.72.148

RUN apt-get update
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y