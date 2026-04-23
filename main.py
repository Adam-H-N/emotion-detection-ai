import cv2
import mediapipe as mp
from fer import FER
import pyttsx3
import threading
from collections import deque
import time
import tkinter as tk
from PIL import Image, ImageTk

# ---------------- INIT ----------------
detector = FER(mtcnn=True)
engine = pyttsx3.init()
last_emotion = ""
last_speak_time = 0

emotion_history = deque(maxlen=10)

cap = cv2.VideoCapture(0)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("800x600")

label_video = tk.Label(root)
label_video.pack()

emotion_label = tk.Label(root, text="Emotion: ", font=("Arial", 24))
emotion_label.pack(pady=10)

bars = {}
for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
    frame = tk.Frame(root)
    frame.pack(anchor="w", padx=20)
    tk.Label(frame, text=emotion.capitalize(), width=10).pack(side="left")
    bar = tk.Label(frame, bg="blue", height=1)
    bar.pack(side="left", fill="x")
    bars[emotion] = bar

# ---------------- FUNCTIONS ----------------
def speak_emotion(emotion):
    global last_speak_time
    engine.say(f"You look {emotion}")
    engine.runAndWait()
    last_speak_time = time.time()

def update():
    global last_emotion

    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    emotions = detector.detect_emotions(frame)

    if emotions:
        emotion_probs = emotions[0]["emotions"]
        emotion_history.append(emotion_probs)

        avg_emotions = {
            k: sum(d[k] for d in emotion_history) / len(emotion_history)
            for k in emotion_probs
        }

        dominant = max(avg_emotions, key=avg_emotions.get)
        confidence = int(avg_emotions[dominant] * 100)

        emotion_label.config(text=f"Emotion: {dominant} ({confidence}%)")

        for e, bar in bars.items():
            bar.config(width=int(avg_emotions[e] * 40))

        if (
            dominant != last_emotion
            and confidence > 50
            and time.time() - last_speak_time > 3
        ):
            threading.Thread(
                target=speak_emotion,
                args=(dominant,),
                daemon=True
            ).start()
            last_emotion = dominant

    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    root.after(10, update)

# ---------------- EXIT ----------------
def quit_app(event=None):
    cap.release()
    root.destroy()

root.bind("<q>", quit_app)
root.bind("<Q>", quit_app)

update()
root.mainloop()
