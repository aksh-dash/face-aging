# Face Analysis System

A minimalist, high-performance real-time face tracking and analysis application built with Streamlit and OpenCV.

This application uses deep learning (pretrained Caffe models) to perform real-time face detection, age estimation, and gender classification. It features a sleek, editorial design and supports both photo uploads and live camera streams.

## Features

- **Two Input Modes:** Seamlessly switch between Upload Photo and Live Camera.
- **Robust Real-Time Tracking:** Uses centroid-based tracking with Exponential Moving Average (EMA) smoothing to eliminate bounding box jitter in live video.
- **Temporal Label Smoothing:** Employs a sliding-window majority vote to stabilize age and gender predictions over time, preventing flickering labels.
- **Editorial UI Design:** A custom minimalist dark theme with clean typography, muted colors, and corner-bracket bounding boxes.
- **Performance Optimized:** prediction throttling limits expensive age/gender classification runs to every 5 frames, ensuring high FPS in live mode.

## Technologies Used

- **Streamlit:** Web interface and state management.
- **streamlit-webrtc:** Low-latency WebRTC video streaming for live camera mode.
- **OpenCV (cv2.dnn):** Core computer vision and neural network inference.
- **Pretrained Models:** 
  - Face Detection: OpenCV SSD (Single Shot Detector) ResNet-10.
  - Age & Gender: Levi & Hassner (CVPR 2015) Caffe models.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aksh-dash/face-aging.git
   cd face-aging
   ```

2. **Install dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Files:**
   The `models/` directory must contain the following pretrained Caffe models:
   - `deploy.prototxt` & `res10_300x300_ssd_iter_140000.caffemodel`
   - `age_deploy.prototxt` & `age_net.caffemodel`
   - `gender_deploy.prototxt` & `gender_net.caffemodel`

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

- Navigate to the local URL provided in the terminal (usually `http://localhost:8501`).
- Choose **UPLOAD PHOTO** to analyze a static image.
- Choose **LIVE CAMERA** to start real-time tracking (requires granting browser camera permissions).

## Disclaimer
This project uses pretrained public models. Predictions are rough estimates and not intended for identification or precise biometric analysis.

---
*Built by Akshat Dange · JSPM RSCOE · 2026*