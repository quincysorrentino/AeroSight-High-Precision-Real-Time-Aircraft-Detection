
## Build the Docker Image

Using Docker Compose (recommended):

```powershell
docker compose up --build
```

Or manually:

```powershell
docker build -t aerotrack:gpu .
docker run --gpus all -v ${PWD}/videos:/app/videos aerotrack:gpu
```

## Without Docker (Windows Local Build)

If you want to run locally on Windows:

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\AeroTrack.exe
```

## CPU-Only Docker Build

If you don't have an NVIDIA GPU, edit the Dockerfile:

- Change `FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` to `FROM ubuntu:22.04`
- Remove the ONNX Runtime GPU download, use CPU version instead

The code will automatically fall back to CPU if CUDA is not available.
