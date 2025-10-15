import cv2
import torch
import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wav
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time
import os

# ==== Facial Model Setup ====
base_dir = os.path.dirname(os.path.dirname(__file__))
face_model_path = os.path.join(base_dir, 'saved_models', 'lie_detection_model.pth')
face_model = models.resnet18(weights=None)
face_model.fc = torch.nn.Linear(face_model.fc.in_features, 4)
face_model.load_state_dict(torch.load(face_model_path, map_location="cpu"))
face_model.eval()

face_labels = ['neutral', 'happy', 'sad', 'surprised']  # adjust as per your training

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Audio Model Setup ====
class AudioMLP(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

audio_model_path = os.path.join(base_dir, 'saved_models', 'audio_lie_model.pth')
print("Loading audio model from:", audio_model_path)


dummy_audio = np.zeros(32)
audio_model = AudioMLP(input_size=32)
audio_model.load_state_dict(torch.load(audio_model_path, map_location="cpu"))
audio_model.eval()

# ==== Real-Time Audio Recorder ====
def record_audio(duration=3, fs=16000):
    print("🎙️ Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    return audio, fs

def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr, n_mfcc=32)
    return np.mean(mfcc.T, axis=0)


# ==== MAIN LOOP ====
cap = cv2.VideoCapture(0)
print("📡 Real-time Lie Detection Started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Facial Prediction ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_label = "No Face"
    face_conf = 0.0

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = face_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = face_model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(prob).item()
            face_label = face_labels[pred]
            face_conf = prob[0][pred].item()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{face_label} ({face_conf:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        break  # just use the first detected face

    # --- Audio Capture and Prediction ---
    audio, sr = record_audio()
    mfcc_feat = extract_mfcc(audio, sr)
    audio_tensor = torch.tensor(mfcc_feat, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = audio_model(audio_tensor)
        audio_prob = torch.nn.functional.softmax(output, dim=1)
        audio_pred = torch.argmax(audio_prob).item()
        audio_label = "Lie" if audio_pred == 1 else "Truth"
        audio_conf = audio_prob[0][audio_pred].item()

    # --- Fusion Display ---
    fusion_text = f"🧠 Audio: {audio_label} ({audio_conf*100:.1f}%) | Face: {face_label}"
    print(fusion_text)
    cv2.putText(frame, fusion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.imshow("Lie Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
