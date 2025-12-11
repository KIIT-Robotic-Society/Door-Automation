
# **KRS Door Automation**

![Platform](https://img.shields.io/badge/platform-Ubuntu%2022.04-blue)
![Docker](https://img.shields.io/badge/docker-supported-blue)
![Python](https://img.shields.io/badge/python-3.10-yellow)
![C++](https://img.shields.io/badge/C++-Hardware%20Control-green)
![AI](https://img.shields.io/badge/AI-Face%20Recognition-orange)
![License](https://img.shields.io/badge/license-MIT-green)

KRS Door Automation is an **Edge-AI powered door security system** combining:

* **Face Recognition (Dlib)**
* **Anti-Spoofing / Liveness Detection (SilentFace)**
* **ToF-based Human Detection (VL53L0X)**
* **High-performance GPIO hardware control (libgpiod + C++)**
* **Dockerized runtime (CPU, GPU, Raspberry Pi modes)**

Built for **maximum speed, reliability, and security** on **resource-limited devices** like Raspberry Pi.

---

# 📌 **Table of Contents**

* [Introduction](#introduction)
* [Multithreading & Parallel Processing Architecture](#multithreading--parallel-processing-architecture)
* [Key Features](#key-features)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Deployment](#deployment)
* [Directory Structure](#directory-structure)
* [Usage Overview](#usage-overview)
* [Hardware Requirements](#hardware-requirements)
* [Electronics & Power Architecture](#electronics--power-architecture)
* [API Reference](#api-reference)
* [Security Notes](#security-notes)
* [Future Roadmap](#future-roadmap)

---

# Introduction

Modern access-control systems demand **speed**, **accuracy**, and **spoof-proof verification**.

KRS Door Automation delivers this by combining:

* **Deep-learning face recognition**
* **SilentFace anti-spoofing (MiniFASNet)**
* **ToF sensor for human presence detection**
* **Optimized multi-thread processing**
* **Real-time C++ hardware controller**

Suitable for:

✔ Research Labs
✔ Corporate Offices
✔ Hostels & Residential Buildings
✔ Industrial & IoT Security


## ⚡ Multithreading & Parallel Processing Architecture

### **1️⃣ Multiprocessing Layer — FastAPI + ML Worker**

`/live/start` launches a **separate ML process**, preventing the API from freezing:

```python
live_process = multiprocessing.Process(target=live_worker)
```

#### This ensures:

* API always stays responsive
* ML cannot block or overload the server
* Raspberry Pi stays cool and smooth

---

### **2️⃣ Python Internal Threading — ML Pipeline Optimization**

Inside the ML worker:

#### **FrameProcessor Thread**

```python
class FrameProcessor(Thread):
    def run():
        # Anti-spoof + Recognition
```

Runs asynchronously:

* Reads frames
* Anti-spoofing
* Face recognition
* Sends results to shared dictionary

#### **ThreadPoolExecutor — Parallel Anti-Spoofing!**

```python
ThreadPoolExecutor(max_workers=4)
```

SilentFace loads **multiple MiniFASNet models** and evaluates them *in parallel* → doubling performance.

---

## **3️⃣ C++ Hardware Controller Threads**

Your C++ controller runs FOUR independent threads:

| Thread                     | Purpose                                 |
| -------------------------- | --------------------------------------- |
| **Heartbeat Thread**       | Checks `/heartbeat` every 5s            |
| **Sensor Thread**          | Reads ToF sensor every 50ms             |
| **Live Controller Thread** | Starts/stops ML when needed             |
| **Polling Thread**         | Reads `/live/status` for final decision |

🔥 **ML runs only when someone approaches the door → huge CPU savings.**


---

# Key Features

### 🧠 Face Recognition

* Dlib model
* Fast encoding matching
* Supports multi-image encodings

### 🛡 Anti-Spoofing (SilentFace)

* Multi-model MiniFASNet
* Prevents photo, screen replay, and printed images

### 🔧 Hardware Integration

* ToF sensing (VL53L0X)
* MOSFET-driven solenoid lock
* GPIO status LEDs
* Real-time control in C++

### 🐳 Docker Support

Modes:

* cpu
* gpu
* pi

---

# Tech Stack

| Layer            | Technology                                       |
| ---------------- | ------------------------------------------------ |
| API              | FastAPI                                          |
| Face Recognition | Dlib                                             |
| Anti-Spoofing    | SilentFace (MiniFASNet)                          |
| Hardware         | C++ + libgpiod                                   |
| Parallel ML      | multiprocessing + threading + ThreadPoolExecutor |
| Build            | CMake                                            |
| Deployment       | Docker + Docker Compose                          |

---

# Installation

```bash
sudo apt update
sudo apt install -y git python3 python3-venv cmake docker docker-compose
```

---

# Deployment

```bash
git clone https://github.com/KIIT-Robotic-Society/Door-Automation.git
cd Door-Automation
sudo bash launch_docker.sh cpu   # or gpu / pi
sudo bash launch.sh
```

---

# Directory Structure

```
.
├── app.py
├── CMakeLists.txt
├── dlib_face_recognition_resnet_model_v1.dat.bz2
├── docker-compose.yaml
├── dockerfile
├── encodings.pickle
├── include
│   ├── I2Cdev.cpp
│   ├── I2Cdev.hpp
│   ├── json_fwd.hpp
│   ├── json.hpp
│   ├── single.cpp
│   ├── single.hpp
│   ├── VL53L0X.cpp
│   ├── VL53L0X_defines.hpp
│   └── VL53L0X.hpp
├── launch_docker.sh
├── launch.sh
├── modelinit.ipynb
├── server.py
├── SilentFaceAntiSpoofing
│   ├── datasets
│   ├── images
│   ├── resources
│   ├── src
│   └── requirements.txt
└── src
    └── main.cpp

```

---

Here is the **README-formatted**, polished, and professional **Usage Overview** section—ready to paste directly into your README.md:

---

# **Usage Overview**

The system operates in the following sequence:

   1. **Monitor environment using the ToF distance sensor**

   * If **no person** is detected → system stays in **IDLE mode**
   * If a **person enters range** → ML pipeline is automatically activated

2. **Start AI recognition (`/live/start`)**

   * Raspberry Pi GPIO **22 = ML Active LED ON**
   * Python ML worker begins running in a separate process
   * C++ hardware controller switches to active monitoring state

3. **Camera captures frames only during ML mode**

   * Reduces CPU load
   * Prevents thermal throttling
   * Extends hardware life

4. **Run face detection + recognition**

   * RetinaFace detects the face
   * Dlib computes face embeddings
   * Matches known users from `encodings.pickle`

5. **Run SilentFace anti-spoofing**

   * Multiple MiniFASNet models run in parallel
   * Prevents printed-photo, replay video, and phone screen spoof attacks

---

## **🔓 If a valid, real, registered user is detected**

* Unlock door **(GPIO17 = HIGH)**
* Wait for the configured unlock duration
* Lock door **(GPIO17 = LOW)**
* Log the event with a timestamp in `system.log`
* Stop ML pipeline and return to idle mode

---

## **❌ If invalid user or timeout occurs**

* ML pipeline automatically stops (`/live/stop`)
* GPIO22 (ML Active) turns **OFF**
* GPIO27 (Idle LED) turns **ON**
* System returns to low-power **IDLE mode**

---

# Hardware Requirements

* Raspberry Pi 4
* VL53L0X
* USB Camera
* 12V Solenoid Lock
* D418 MOSFET Module
* PCBs (Power, Fan, Lock, LEDs)

---

# 🔌 Electronics & Power Architecture

### **PCB 1** — AC → 12V → 5V USB (Pi + Display)

### **PCB 2** — AC → 5V (Cooling Fan)

### **PCB 3** — AC → 12V (Solenoid Lock) + MOSFET Driver

### **PCB 4** — LED Indicators

### GPIO Mapping

| GPIO Pin | Function          |
| -------- | ----------------- |
| **17**   | Door Lock Trigger |
| **27**   | System Idle LED   |
| **22**   | ML Active LED     |

---

# API Reference

### **GET /heartbeat**

```json
{"status": "live"}
```

### **POST /live/start**

Starts ML process.

### **POST /live/stop**

Stops ML process.

### **GET /live/status**

Returns:

```json
{
  "status": "running",
  "name": "user",
  "label": 1
}
```

---

# Security Notes

* All AI runs **locally**
* No cloud storage
* Encodings stored offline
* Anti-spoofing blocks photo/video attacks
* Docker sandboxing protects system

---

# Future Roadmap

* Web dashboard
* RFID/NFC + Face MFA
* BLE Token Authentication
* YOLO-based liveness
* Cloud analytics
* Multi-door deployment

---
