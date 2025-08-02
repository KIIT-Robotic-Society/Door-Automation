### `LOCAL_README.md`

````markdown
# 🔐 Door Automation – Local Setup Guide

This document provides a step-by-step guide to set up and run the Door Automation system locally.  
The system uses face recognition and liveness (anti-spoofing) detection to control door access.

---

## ⚙️ Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip
- OpenCV compatible webcam
- Optional: NVIDIA GPU + CUDA (for faster inference)

---

## 🧰 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/KIIT-Robotic-Society/Door-Automation.git
cd Door-Automation
````

---

### 2. Create and Activate Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you encounter torch-related issues, install with the official URL:

```bash
# For GPU (replace 'cu118' with your CUDA version if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CPU only
pip install torch torchvision
```

---

## 📂 Project Structure

```
Door-Automation/
├── encodings.py                  # Stores face encodings
├── test.py                       # Runs spoof detection and recognition
├── train.py                      # (If available, for spoof model training)
├── liveness_model/               # Contains pre-trained anti-spoofing models
├── face_data/                    # Directory for saved face images
├── attendance_logs/              # Logs for access (timestamped)
├── assets/
│   └── Attendance.png            # UI background
├── requirements.txt              # Python dependencies
└── LOCAL_README.md               # This setup guide
```

---

## 🚀 How to Use

### ✅ 1. Register a New User

Run the face encoding script to register a new face:

```bash
python encodings.py
```

It captures images and generates embeddings stored in a `.pkl` file.

---

### ✅ 2. Run the System (Recognition + Liveness)

To start the door authentication system:

```bash
python test.py
```

What happens:

* Captures real-time webcam input
* Performs face recognition using DeepFace
* Runs spoof detection using pretrained model
* Logs results with timestamp
* Opens door (mocked) if verification passes

---

## 📒 Logs & Outputs

* Access attempts and results are logged in:
  `attendance_logs/access_log.csv`
---

