# 🔐 Door Automation – Local Setup Guide

This document provides a step-by-step guide to set up and run the Door Automation system locally.  
The system uses face recognition and liveness (anti-spoofing) detection to control door access.

---

## ⚙️ Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- OpenCV-compatible webcam
- (Optional) NVIDIA GPU with CUDA support for faster inference

---

## 🧰 Installation Steps

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/KIIT-Robotic-Society/Door-Automation.git
cd Door-Automation
```

---

### 🔹 Step 2: Create and Activate Virtual Environment

#### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### For Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 🔹 Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you face issues with torch installation, install manually using the official PyTorch URL:

```bash
# For GPU (change cu118 to your CUDA version if different)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CPU only
pip install torch torchvision
```

---

## 📂 Project Structure

```
.
├── SilentFaceAntiSpoofing/             # Anti-spoofing model code
│   ├── src/                             # Model scripts (train/test/utils)
│   ├── weights/                         # Pretrained MiniFASNet models
│   └── ...                              # Other necessary files
├── __pycache__/                        # Python cache
├── db/                                 # Stores rough encodings
│   └── encodings.pickle
├── dlib-19.24.99-cp312-cp312-win_amd64.whl    # Dlib wheel (Windows)
├── dlib_face_recognition_resnet_model_v1.dat.bz2  # Dlib face model
├── shape_predictor_68_face_landmarks.dat.bz2     # Face landmarks model
├── encodings.pickle                    # Pickled face real encodings
├── modelinit.ipynb                     # Notebook to run main logic
├── requirements.txt                    # Python dependencies
├── README.md                           # GitHub README
└── Local_Readme.md                     # Detailed local setup guide

```

---

## 🚀 How to Use

### ✅ Step 1: Register a New Face

Run the encoding script to capture and save a new face embedding:

```bash
python encodings.py
```

- The script will open your webcam.
- Capture images and generate `.pkl` file of face encodings.

---

### ✅ Step 2: Run the Main System

Start the face recognition and spoof detection system:

```bash
python test.py
```

What happens:

- Webcam input is streamed live
- Face is recognized using DeepFace
- Spoof check (liveness detection) is performed
- Access decision (Allow/Deny) is made
- Action is logged with timestamp
- Door open is simulated (can be integrated with hardware)

---

## 📒 Logs & Outputs

- Access logs are automatically saved to:  
  `attendance_logs/access_log.csv`

Each entry contains:

- User status (recognized/spoof)
- Date and time
- Pass/Fail result
