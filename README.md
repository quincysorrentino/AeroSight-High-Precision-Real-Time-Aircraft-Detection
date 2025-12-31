# AeroSight — High-Precision Real-Time Aircraft Detection

A real-time aircraft detection and tracking system using YOLO11n and C++ for high-accuracy aerial perception.

## Overview

Perception for aerial systems such as, collision avoidance, autonomous navigation, or airspace awareness, remains one of the most challenging domains in computer vision. I built **AeroSight** to deepen my practical experience in real-time object detection, tracking, and perception pipeline design.

My goal was to design a system that closely mirrored how perception systems are built in industry. Including model training, running real time inference with hardware limitations, and fusing the results with classical state estimation.

This project implements the perception and tracking components commonly used in Sense-and-Avoid systems:

- **Model Fine-Tuning**: Custom fine-tuning of the YOLO11n model on domain-specific aerial imagery to improve detection performance and robustness.
- **Detection**: Real-time aircraft detection using optimized YOLO11n inference deployed via ONNX Runtime.
- **Tracking**: Persistent track ID assignment with aggressive low-confidence detection matching to maintain stable tracks through uncertainty and partial occlusions.
- **Systems**: A modular C++ framework designed for low-latency execution and deployment on edge hardware.

## Gallery

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/f1b949b0-ea76-4cd9-bb21-5e6bdc734941" style="width:100%;" />
      <br/>
      <em>Image shows Kalman-Only Constant-Velocity Tracker (KO-CV), indicated by red bounding box</em>
    </td>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/decd934e-b119-4eed-869c-84a3952d4833" style="width:100%;" />
      <br/>
      <em>Image shows multiclass (aircraft type) detection capabilities</em>
    </td>
  </tr>
</table>

https://github.com/user-attachments/assets/6b24323e-63bc-4192-aacc-7d9f81f6dd45

Video showing the detector being turned off after three seconds, allowing KO-CV to take over and predict motion. The detector is then re-enabled, and the track is successfully regained.

## Tech Stack

| Technology                | Role                                                                    |
| ------------------------- | ----------------------------------------------------------------------- |
| **ONNX Runtime (CUDA)**   | High-performance GPU-accelerated inference within the C++ pipeline      |
| **OpenCV**                | Image preprocessing, visualization, and video I/O in the runtime system |
| **PyTorch / Ultralytics** | Fine-tuning the YOLOv11n model and exporting trained weights            |
| **Python**                | Dataset preprocessing, training orchestration, evaluation, and tooling  |

## Perception Pipeline

#### Object Detection (YOLO11n)

The system utilizes a finetuned **YOLO11n** model, trained (using an Nvidia A100 GPU instance through Google Colab) on a specialized military aircraft dataset from Kaggle.

- **Preprocessing:** Frames are resized to $640 \times 640$ using **letterboxing** to preserve aspect ratios, followed by channel-swapping (BGR to RGB) and pixel normalization.
- **Post-Processing:** Implements **Non-Maximum Suppression (NMS)** in C++ to prune redundant bounding boxes. Detections are filtered based on a configurable confidence threshold ($T_{conf} > 0.30$).

#### Persistent Tracking System

To maintain stable track IDs across frames despite model uncertainty or classification changes, AeroSight implements an aggressive detection-to-track association strategy:

- **Track Matching:** Uses combined IOU and distance metrics with strong preference for same-aircraft-type matches. Accepts very low confidence detections (≥30%) to prevent track loss during momentary uncertainty.
- **Track Merging:** Automatically consolidates duplicate tracks that are spatially close (<200px) or overlapping (IOU >0.3), keeping the highest confidence ID.
- **Persistence:** Tracks remain active for 6 seconds (180 frames @ 30fps) without detection, allowing recovery from temporary occlusions or model failures.
- **Class Flexibility:** Track IDs persist even when the model changes its mind about aircraft type (e.g., F-16 → F-18), prioritizing ID stability over classification consistency.

#### C++ Inference Engine

The core logic is implemented in C++ to ensure deterministic performance, maintian minimal overhead, and improve processing speeds.

- **Memory Management:** Utilizes smart pointers and pre-allocated tensors to minimize heap allocations during the inference loop.
- **Modularity:** The `Detector` class is decoupled from the `Tracker` class, allowing for "plug-and-play" swapping of models.

#### Repository Layout

```
├── AeroTrack/                          # C++ tracking and inference system
│   ├── main.cpp                        # Video processing pipeline with persistent tracking
│   ├── detector.cpp / detector.h       # ONNX Runtime + OpenCV YOLO wrapper
│   ├── track.cpp / track.h             # Track management and state
│   ├── kalman_filter.cpp / .h          # Kalman filter implementation
│   ├── hungarian.cpp / hungarian.h     # Hungarian algorithm for assignment
│   ├── utils.cpp / utils.h             # Helper functions (IOU, distance, NMS)
│   ├── last.onnx                       # Trained YOLO11n model weights
│   ├── CMakeLists.txt                  # Build configuration
│   ├── Dockerfile                      # Container image specification
│   ├── docker-compose.yml              # Docker deployment config
│   └── BUILD.md                        # Build instructions
│
├── train_yolov11.ipynb                 # Model training notebook (Google Colab)
├── requirements.txt                    # Python dependencies for training
└── README.md                           # This file
```

### Quickstart

#### Build the Docker Image

```powershell
docker compose up --build
```

#### Without Docker (Windows Local Build)

If you want to run locally on Windows:

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\AeroTrack.exe
```

#### CPU-Only Docker Build

If you don't have an NVIDIA GPU, edit the Dockerfile:

- Change `FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` to `FROM ubuntu:22.04`
- Remove the ONNX Runtime GPU download, use CPU version instead

The code will automatically fall back to CPU if CUDA is not available.

```powershell
cmake --build . --config Release

./build/Release/AeroTrack.exe
```
