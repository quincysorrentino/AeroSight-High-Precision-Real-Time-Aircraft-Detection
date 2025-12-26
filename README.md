# AeroDetect: High-Precision Perception for Aerial Systems

## Overview

## Gallery
<img width="3071" height="1494" alt="image" src="https://github.com/user-attachments/assets/f0ff9e6f-fbcd-46b5-bd35-deb5b627245f" />

## Tech Stack
XXX

## Repository Layout

- `AeroTrack/`
  - `main.cpp` – video inference pipeline and UI overlay logic.
  - `detector.cpp / detector.h` – ONNX Runtime, OpenCV wrapper around the YOLOv11 ONNX model.
  - `tracker.cpp / tracker.h` – tracking hooks, extension point for multi‑frame tracking Kalman filter.
  - `CMakeLists.txt`
- `train_yolov11.ipynb` – notebook for training and exporting the YOLOv11 model using Google Colab.
- `military-aircraft-yolo/` – dataset configuration and labels.
- `requirements.txt`
- `.gitignore`

## Quickstart

```powershell
cmake --build . --config Release

./build/Release/AeroTrack.exe
```
