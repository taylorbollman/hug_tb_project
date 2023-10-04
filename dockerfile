FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install torch torchvision