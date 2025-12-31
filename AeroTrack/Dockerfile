# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime GPU version
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.17.0.tgz && \
    mv onnxruntime-linux-x64-gpu-1.17.0 /opt/onnxruntime && \
    rm onnxruntime-linux-x64-gpu-1.17.0.tgz

# Set working directory
WORKDIR /app

# Set environment variables for ONNX Runtime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

# Copy source files
COPY *.h /app/
COPY *.cpp /app/
COPY CMakeLists.txt /app/

# Copy model file
COPY last.onnx /app/

# Create videos directory and copy test video
RUN mkdir -p /app/videos
COPY videos/test.mp4 /app/videos/

# Build the project with ONNX Runtime paths
RUN mkdir -p build && cd build && \
    cmake -DONNXRUNTIME_ROOT=/opt/onnxruntime .. && \
    cmake --build . --config Release

# Set the entrypoint
WORKDIR /app/build
CMD ["./AeroTrack"]
