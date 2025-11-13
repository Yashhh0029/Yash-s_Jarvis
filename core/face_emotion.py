import cv2
import numpy as np
from deepface import DeepFace
from core.memory_engine import JarvisMemory
from core.speech_engine import speak
from core.voice_effects import JarvisEffects
import time

memory = JarvisMemory()
jarvis_fx = JarvisEffects()


class FaceEmotionAnalyzer:
    """Analyzes facial expressions to detect mood and sync with Jarvis’s emotional state."""

    def __init__(self):
        print("📸 Face Emotion Analyzer Initialized")

    def capture_emotion(self):
        """Capture one frame and analyze user emotion."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera access failed, Yash.", mood="alert")
            return None

        # cinematic touch
        speak("Let me take a quick look at you, Yash.", mood="happy")
        time.sleep(0.7)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            speak("I couldn’t capture your face clearly.", mood="alert")
            return None

        # DeepFace expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # analyze emotion
            analysis = DeepFace.analyze(
                rgb_frame,
                actions=['emotion'],
                enforce_detection=False
            )

            # correct indexing: DeepFace returns a dict
            dominant_emotion = analysis.get('dominant_emotion', 'neutral').lower()

            print(f"🧠 Detected Emotion: {dominant_emotion}")

            # map to Jarvis internal moods
            mood_map = {
                "happy": "happy",
                "sad": "serious",
                "angry": "serious",
                "neutral": "neutral",
                "surprise": "happy",
                "fear": "alert",
                "disgust": "serious"
            }

            mood = mood_map.get(dominant_emotion, "neutral")

            # update memory + mood tone
            memory.set_mood(mood)
            jarvis_fx.mood_tone(mood)

            # dynamic natural responses
            if dominant_emotion in ["happy", "surprise"]:
                speak(
                    f"You look {dominant_emotion} today, Yash. That spark suits you perfectly.",
                    mood="happy"
                )

            elif dominant_emotion in ["sad", "fear", "disgust"]:
                speak(
                    f"You seem a bit {dominant_emotion} today. Even the strongest have such days.",
                    mood="serious"
                )

            elif dominant_emotion == "angry":
                speak("You look tense, Yash. Try taking a few deep breaths.", mood="serious")

            elif dominant_emotion == "neutral":
                speak("You look calm and steady — a balanced state of mind.", mood="neutral")

            else:
                speak("You seem thoughtful, Yash.", mood="neutral")

            return dominant_emotion

        except Exception as e:
            print(f"⚠️ Emotion analysis failed: {e}")
            speak("I couldn’t analyze your expression properly.", mood="alert")
            return None
