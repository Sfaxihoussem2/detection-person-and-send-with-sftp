# Real-Time Object / Person Detection with OpenCV & SFTP Upload

**Computer vision pipeline optimized for NVIDIA Jetson devices: multi-camera detection using MobileNet SSD, local annotation, and automatic upload of results via SFTP.**

---

## Project Overview

This project captures video from one or more cameras, runs real-time object detection with a pre-trained MobileNet SSD model via OpenCV, annotates the frames, saves them locally, and uploads the results to a remote server using SFTP.

It is particularly suited for edge devices such as NVIDIA Jetson Nano / Xavier / Orin where CUDA-accelerated OpenCV can deliver practical frame rates.

---

## Key Features

- Real-time object detection (MobileNet SSD)
- Support for multiple camera streams
- Annotated image saving
- Automatic SFTP upload of detection results
- Optimized for NVIDIA Jetson + JetPack environment
- Useful for surveillance, people counting, or remote monitoring scenarios

---

## Hardware & Software Requirements

**Hardware**
- NVIDIA Jetson device (Nano, Xavier, AGX Orin, etc.)
- One or more cameras (USB / CSI)

**Software**
- JetPack (CUDA, cuDNN, TensorRT)
- Python 3.x
- OpenCV compiled with CUDA support (recommended)
- `pysftp`

---

## Installation (Jetson)

1. Ensure the latest compatible JetPack is installed.
2. Prefer an OpenCV build with CUDA support for better performance.
3. Install Python dependencies:

```bash
pip install pysftp
# plus any additional packages required by your specific script
```

4. Configure camera indexes, model paths, and SFTP credentials in the code.

---

## Typical Workflow

```
Cameras → OpenCV capture → MobileNet SSD inference → Annotate frames
         → Save locally → Upload via SFTP
```

---

## Repository Structure

```
├── README.md
└── detection and send sftp/     # Python scripts and related assets
```

---

## Notes & Best Practices

- Never hard-code production SFTP passwords in source code. Use environment variables or a secrets manager.
- Monitor thermal and power limits on Jetson devices during continuous inference.
- Adjust detection confidence thresholds and resolution according to your accuracy vs. speed needs.

---

## Future Improvements

- Switch to a more modern detector (YOLO variants, TensorRT optimized engines)
- Add tracking (DeepSORT / ByteTrack) for unique person counting
- MQTT or WebSocket notification on detection events
- Local dashboard for live view
- Docker container for reproducible deployment

---

## Author

**Houssem Eddine Sfaxi**  
Embedded Systems & IoT Engineer  
[GitHub](https://github.com/Sfaxihoussem2) · [Portfolio](https://sfaxihoussem2.github.io/My-portfolio/) · [LinkedIn](https://www.linkedin.com/in/houssem-sfaxi-3272a9280)

---

## License

Portfolio and educational project.
