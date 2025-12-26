# Aircraft_Detect

## AeroTrack: Real‑Time Aircraft Detection & Tracking

End‑to‑end project that takes a military aircraft dataset from training in Python to a C++ inference engine capable of running real‑time detection on video.


## Highlights

- **Fine-tuned YOLOv11n** trained on a labeled military aircraft dataset from Kaggle.
- **C++ inference (AeroTrack)** that ingests a video, runs ONNX inference, and writes an annotated `output.mp4`.
- **Class Identification**: class name (e.g., F15, F16, Su57), confidence %.

Tech stack:

- **Training**: Python, Ultralytics YOLOv11, PyTorch
- **Inference**: C++17, OpenCV, ONNX Runtime


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


## Overview

1. **Model Training**  
    Dataset Download from Kaggle (https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data)
    Use Ultralytics YOLOv11n on the dataset to learn to detect multiple aircraft classes
    Run training in Google Colab with Nvidia A100 GPU instance
    Export the trained model in ONNX format

2. **Model Inference and Kalman Filtering**  
   Each frame is:
   - Resized and letterboxed to 640×640.
   - Converted from BGR to RGB and HWC to CHW
   - Passed through ONNX Runtime to get detection outputs
   - Post‑processed into bounding boxes, class IDs, and confidence scores

3. **Render & Export Video**  
   All run in `main.cpp`:
   - Opens an input video
   - Runs the detector on each frame 
   - Draws green bounding boxes, aircraft names, confidence percentages, and an FPS overlay
   - Writes results to output file

## Quickstart

```powershell
cmake --build . --config Release

./build/Release/AeroTrack.exe
```

## What This Project Demonstrates

- Ability to **design an end‑to‑end ML system**: from dataset and training to deployment.
- Practical experience with **modern YOLO training** and **ONNX export**.
- Comfort with **C++17**, **CMake**, and **performance‑oriented coding** (memory access patterns, multi‑threading, Release builds).
- Integration of **OpenCV** and **ONNX Runtime** into a cohesive inference pipeline.
- Attention to **engineering hygiene**: `.gitignore` for large artifacts, clear separation of training vs. inference code.

If you’re reviewing this as part of a portfolio or resume, the key takeaway is that this repo shows how I approach building ML systems that are not just accurate in a notebook, but also deployable, performant, and maintainable in production‑style environments.
