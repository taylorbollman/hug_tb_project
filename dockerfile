# Use the official CUDA image as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base

# Install Python and other dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install torch torchvision

# Use a smaller image as the final image
FROM ubuntu:20.04
COPY --from=base / /

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . .

# Set the entry point
ENTRYPOINT ["python3", "main.py"]