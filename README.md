# 🔐 Door Automation System — KRS Project

A secure, smart biometric access system for the KIIT Robotics Society room — combining real-time face recognition and spoof detection. This project ensures only genuine users can unlock access using AI-driven verification.

---

## 🚀 Project Objective

To develop a **contactless, real-time, and spoof-proof** door access control system using facial recognition and liveness detection. This solution automates door entry for registered users, while protecting against spoofing attacks (e.g. printed photos or videos).

---

## ✨ Key Features

- 🔍 **Real-Time Face Recognition**  
  Utilizes `face_recognition` library for accurate, low-latency identification.

- 🧠 **Liveness Detection with Anti-Spoofing**  
  Integrated with [SilentFaceAntiSpoofing](https://github.com/minivision-ai/SilentFaceAntiSpoofing) and MiniFASNet models.

- 🔐 **Secure Login/Logout System**  
  Maintains access logs with timestamps for all entry and exit events.

- 👤 **User Authentication**  
  Only registered faces are allowed — unauthorized users are denied.

- 🧻 **Log Files & Tracking**  
  Tracks entries/exits with visual proof and timestamps.

---

## 🧩 Tech Stack

| Category         | Technology Used                     |
|------------------|-------------------------------------|
| **Language**      | Python 3.x                          |
| **Face Recognition** | `face_recognition` library             |
| **Spoof Detection** | SilentFaceAntiSpoofing (MiniFASNet)  |
| **Face Detection**  | RetinaFace                         |
| **Model Inference** | PyTorch                           |
| **UI & Control**    | OpenCV, Custom GUI (Tkinter)      |
| **Data Handling**   | Pickle, OS, Time modules           |

---

## 🏗️ System Flow Overview

          ┌───────────────┐
          │  Camera Feed  │
          └──────┬────────┘
                 │
                 ▼
      ┌────────────────────────┐
      │ Face Detection &       │
      │ Face Recognition       │
      └────────┬───────────────┘
               │
               ▼
      ┌────────────────────────┐
      │  Spoof Detection /     │
      │  Liveness Verification │
      └───────┬───────┬────────┘
              │       │
              │       ▼
              │   [Liveness Fail]
              ▼
     [Liveness Pass]
              │
              ▼
       ┌─────────────┐
       │  Open Door  │
       └─────┬───────┘
             │
             ▼
      ┌────────────────────────────┐
      │ Log Activity with Timestamp│
      │      (Success / Fail)      │
      └────────────────────────────┘

---

## 🌱 Future Scope

- 🔧 Integration with IoT relay for actual door opening
- 📲 Mobile-based registration and admin control panel
- 🌐 Web dashboard for real-time monitoring and access logs
- 📦 Dockerization and container deployment
