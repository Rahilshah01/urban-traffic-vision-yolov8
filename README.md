# 🚦 Urban Traffic Vision: Real-Time Analytics Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Supervision](https://img.shields.io/badge/Supervision-v0.24+-FF6B35?style=for-the-badge)

**Real-time vehicle detection and classification pipeline for urban traffic footage.**  
Detects cars, buses, trucks, and motorcycles frame-by-frame with confidence filtering and NMS deduplication.

</div>

---

## ⚡ Results at a Glance

| Metric | Detail |
|---|---|
| 🚗 Vehicle Classes | Cars (2), Motorcycles (3), Buses (5), Trucks (7) — COCO class IDs |
| 🎯 Confidence Threshold | 0.70 — filters low-confidence noise (manholes, shadows) |
| 🔲 IOU Threshold | 0.50 — merges duplicate boxes on the same vehicle |
| 🧹 NMS Mode | `agnostic_nms=True` — prevents overlapping labels across classes |
| 🤖 Model | YOLOv8 Nano (`yolov8n.pt`) — optimized for real-time inference speed |
| 📺 Output | Live annotated video window with bounding boxes + confidence labels |

---

## 🧠 What This System Does

Raw traffic footage is noisy — the same vehicle can trigger multiple overlapping detections per frame, and low-contrast objects like manholes or road markings generate false positives. This pipeline handles both problems at the inference level:

```
Raw Video Frame
    │
    ▼
YOLOv8 Nano Inference
├── classes=[2,3,5,7]     → Only detect vehicles, ignore everything else
├── conf=0.70             → Drop any detection below 70% confidence
├── iou=0.50              → Merge boxes overlapping >50% (same vehicle)
└── agnostic_nms=True     → NMS applied across all classes, not per-class
    │
    ▼
Supervision Detections Object
    │
    ▼
BoxAnnotator + LabelAnnotator  →  Annotated Frame (live display)
```

---

## 💻 Core Implementation

```python
results = model(
    frame,
    classes=[2, 3, 5, 7],  # COCO IDs: car, motorcycle, bus, truck
    conf=0.7,               # Ignore low-confidence noise like manholes
    iou=0.5,                # Merge multiple boxes on the same vehicle
    agnostic_nms=True       # Prevent overlapping labels for the same object
)[0]

detections = sv.Detections.from_ultralytics(results)

labels = [
    f"{model.model.names[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]
```

The three inference parameters work together as a detection quality pipeline — each one addresses a different source of noise in real-world traffic footage.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Detection Model | Ultralytics YOLOv8 Nano (`yolov8n.pt`) |
| Video I/O | OpenCV (`cv2.VideoCapture`) |
| Detection Handling | Supervision (`sv.Detections.from_ultralytics`) |
| Annotation | `sv.BoxAnnotator` + `sv.LabelAnnotator` |
| NMS | Built-in via Ultralytics inference params |
| Language | Python 3.10+ |

---

## 🔑 Key Engineering Decisions

- **Why YOLOv8 Nano?** The Nano variant prioritizes inference speed over raw accuracy — the right trade-off for real-time video where throughput matters more than catching every single vehicle.
- **Why `classes=[2, 3, 5, 7]`?** Filtering to vehicle-only COCO class IDs at inference time (not post-processing) is faster and cleaner than running full detection and filtering the results afterward.
- **Why `conf=0.7`?** Through testing on traffic footage, 0.7 was the threshold that eliminated environmental false positives (manholes, road markings, pedestrians) while retaining accurate vehicle detections.
- **Why `agnostic_nms=True`?** Standard per-class NMS would allow a bus and a truck box to overlap on the same vehicle. Agnostic NMS suppresses overlaps regardless of class — critical when a large vehicle triggers multiple class detections simultaneously.
- **Why Supervision over raw OpenCV annotation?** The `sv.Detections` abstraction and annotator classes handle the boilerplate of converting YOLO output to drawable bounding boxes cleanly, keeping the main loop readable.

---

## 📊 Visual Output

### Real-Time Detection
YOLOv8 Nano classifies multiple vehicle types simultaneously with confidence scores displayed per detection.

![Detection Screenshot](images/detection_img.png)

### Traffic Density Analysis
Frame-level detection counts aggregated into a temporal density chart — identifying peak congestion windows.

![Traffic Density](images/traffic_density_smooth.png)

### Analytics Summary
Pipeline-generated summary report with peak traffic count and average vehicle flow metrics.

![Analytics Summary](images/analysis_img.png)

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Rahilshah01/urban-traffic-vision-yolov8.git
cd urban-traffic-vision-yolov8

# 2. Install dependencies
pip install ultralytics supervision opencv-python

# 3. Add footage
# Place your traffic video in data/ as traffic_footage.mp4

# 4. Run
python main.py
# Press 'q' to quit the live window
```

---

## 📁 Repository Structure

```
urban-traffic-vision-yolov8/
├── main.py                  # Detection pipeline
├── yolov8n.pt               # YOLOv8 Nano weights (auto-downloaded on first run)
├── data/
│   └── traffic_footage.mp4  # Input video (add your own)
├── images/                  # Output screenshots
└── README.md
```

> 💡 **`yolov8n.pt` auto-downloads** on first run if not present — no manual download needed.

---

## 🔭 Potential Extensions

- **Vehicle counting line** — Add `sv.LineZone` to count vehicles crossing a defined boundary
- **Temporal analytics** — Log frame-level counts to CSV for density trend analysis
- **Multi-camera** — Swap `VideoCapture("file.mp4")` for `VideoCapture(0)` for live webcam/CCTV feed

---

*Built by [Rahil Shah](https://rahil-shah-portfolio.vercel.app/) · MS Data Science @ Stevens Institute of Technology*
