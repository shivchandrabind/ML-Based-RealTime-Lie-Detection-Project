# ML-Based-RealTime-Lie-Detection-Project
A Realtime lie detection project made by using machine learning.


# 🧠 Real-Time Lie Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-green?logo=opencv)
![Librosa](https://img.shields.io/badge/Audio-Librosa-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Passing](https://img.shields.io/badge/Build-Passing-brightgreen)

---

## 🧾 Overview

The **Real-Time Lie Detection System** uses **AI-based facial expression and audio analysis** to detect deception.  
It combines **deep learning, voice stress analysis, and computer vision** to classify human behavior in real time.

---

## ✨ Features

✅ **Facial Micro-Expression Recognition** using CNN models  
🎙️ **Voice Stress Detection** using MFCC + Neural Networks  
🖐️ **Hand & Body Movement Tracking** (optional via Mediapipe)  
⚡ **Real-Time Predictions** with webcam & microphone input  
🧠 **Trained Models (.pth)** for instant use  
📊 **Dashboard Integration** (Flask/React optional)  
💾 **Logging System** to store each session’s results

---

## 🧩 Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Language** | Python 3.8+ |
| **AI / ML** | PyTorch, Scikit-learn |
| **Vision** | OpenCV, Mediapipe |
| **Audio Processing** | Librosa, PyAudio |
| **Utilities** | NumPy, Pandas, Matplotlib |
| **Deployment** | Flask / React (optional) |

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

git clone https://github.com/shivchandrabind/ML-Based-RealTime-Lie-Detection-Project.git
cd LieDetectionProject
-->Then Move to Correct Location where realtime_lie_detector.py file is save (i.e. in scripts folder)
 In Terminal Type:
 cd scripts

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate          # For Windows
source venv/bin/activate       # For macOS/Linux

3️⃣ Install Required Dependencies
pip install -r requirements.txt

If you don’t have the file yet, install manually:

pip install opencv-python mediapipe torch torchvision torchaudio numpy pandas librosa pyaudio scikit-learn matplotlib

▶️ How to Run
🎥 Real-Time Lie Detection
python scripts/realtime_lie_detector.py

🎧 Audio Prediction
python scripts/predict_audio.py

🧠 Image-Based Prediction
python scripts/predict_single_image.py

🏋️‍♂️ Model Training
python scripts/train_model.py


📁 Project Structure

LieDetectionProject/
│
├── data/                      # Datasets (not uploaded)
├── preprocessing/              # Data preprocessing scripts
├── saved_models/               # Trained model files (.pth)
│   ├── lie_detection_model.pth
│   └── audio_lie_model.pth
├── scripts/                    # Core project scripts
│   ├── realtime_lie_detection.py
│   ├── predict_audio.py
│   ├── extract_features.py
│   ├── train_model.py
│   ├── audio_dataset_loader.py
│   └── dataset_loader.py
├── venv/                       # Virtual environment
├── requirements.txt
└── README.md


🧠 Models
Model	Description	Location
lie_detection_model.pth	Facial micro-expression model	saved_models/
audio_lie_model.pth	Audio stress analysis model	saved_models/

📊 Datasets Used
SAMM Dataset – Facial micro-expressions
UBFC Dataset – Physiological + video-based
Custom Audio Data – Recorded speech samples

📂 Note: Datasets are not included in the repository due to size and license restrictions.
➡️ Download manually or contact the author for access.

💡 Example Output
Console:
Recording audio...
Audio: Truth (99.9%) | Face: Happy
.
.
Recording audio...
Audio: Truth (100%) | Face: Surprised

🧰 Troubleshooting
1. ModuleNotFoundError: No module named 'cv2'
→ Activate your environment and install missing package:
venv\Scripts\activate
pip install opencv-python

2. Model Not Found:
Ensure the file lie_detection_model.pth exists in saved_models/.

🌟 Future Enhancements
Add transformer-based emotion analysis
Improve real-time accuracy using hybrid multimodal fusion
Add Flask-based monitoring dashboard
Include multi-person detection support

👨‍💻 Author
Developed by: Shiv Chandra Bind
📧 Email: bdrdshivchandra9125@gmail.com
📍 India

🪪 License
This project is released under the MIT License — use and modify freely for research and learning.

💫 Contribute
Pull requests are welcome!
To contribute:
git checkout -b feature/new-feature
git commit -m "Added new functionality"
git push origin feature/new-feature
Then open a PR 🎉

📌 Citation
If you use this project in your work:
@misc{LieDetectionProject2025,
  author = {Shiv Chandra Bind},
  title = {Real-Time Lie Detection System: AI-based Multimodal Deception Analysis},
  year = {2025},
  howpublished = {\url{https://github.com/shivchandrabind/ML-Based-RealTime-Lie-Detection-Project}}
}

🖼️ Demo Preview (Optional)
![Demo Screenshot](preview.png)
![Live Detection](demo.mp4)

---
Thank you !
