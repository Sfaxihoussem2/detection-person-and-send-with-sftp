# Real-Time Object Detection with OpenCV and SFTP Upload

This project performs real-time object detection using OpenCV and a pre-trained MobileNet SSD model. It captures video streams from two cameras, detects objects, and saves the annotated images locally. Additionally, it uploads the images to an SFTP server. The code is optimized to run on NVIDIA Jetson devices.

---

## Prerequisites

- **NVIDIA Jetson device** (e.g., Jetson Nano, Jetson Xavier, Jetson AGX Orin)
- **JetPack** (includes CUDA, cuDNN, TensorRT, and other dependencies)
- Python 3.x
- OpenCV (compiled with CUDA support)
- pysftp

---

## Installation on NVIDIA Jetson

### 1. Set Up JetPack
Ensure your NVIDIA Jetson device is running the latest version of JetPack. JetPack includes CUDA, cuDNN, TensorRT, and other necessary libraries for deep learning and computer vision.

- Download and install JetPack from the [NVIDIA Developer website](https://developer.nvidia.com/embedded/jetpack).

### 2. Install OpenCV with CUDA Support
OpenCV on Jetson should be compiled with CUDA support for optimal performance. You can either:
- Use the pre-installed OpenCV that comes with JetPack.
- Compile OpenCV from source with CUDA support (recommended for better performance).

To compile OpenCV from source, follow the instructions in this guide:  
[How to Install OpenCV on Jetson Nano](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html)

### 3. Install Required Python Packages
Install the required Python packages using `pip`:

```bash
pip install pysftp
