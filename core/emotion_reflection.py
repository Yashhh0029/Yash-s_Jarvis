# core/emotion_reflection.py
import datetime
import random
from core.memory_engine import JarvisMemory
from core.speech_engine import speak

# IMPORTANT: use the SAME shared memory instance
memory = JarvisMemory()


class JarvisEmotionReflection:
    """Keeps track of emotional trends and reflects naturally."""

    def __init__(self):
        # ensure history exists
        if "emotion_history" not in memory.memory:
            memory.memory["emotion_history"] = []
            memory._save_memory()
        print("🧠 Emotion Reflection Engine Ready")

    # ----------------------------------------------------------
    def add_emotion(self, mood):
        """Store last 10 emotional states safely."""
        if mood not in ["happy", "serious", "neutral", "alert"]:
            # prevent invalid moods from breaking reflection
            mood = "neutral"

        entry = {
            "mood": mood,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history = memory.memory.get("emotion_history", [])
        history.append(entry)

        # keep only latest 10
        memory.memory["emotion_history"] = history[-10:]
        memory._save_memory()

    # ----------------------------------------------------------
    def reflect(self):
        """Analyze mood patterns and reflect naturally."""
        history = memory.memory.get("emotion_history", [])

        if not history:
            speak("I don’t have enough emotional data yet, Yash.", mood="neutral")
            return

        # recent moods
        last = history[-1]["mood"]
        prev = history[-2]["mood"] if len(history) > 1 else None

        # ------------------------------------------------------
        # MOOD TRANSITION DETECTION
        # ------------------------------------------------------
        if prev and last != prev:
            transitions = {
                ("serious", "happy"): "You seem brighter than before, Yash.",
                ("happy", "serious"): "You seem a little quieter today. Everything okay?",
                ("neutral", "happy"): "You sound livelier than last time.",
                ("neutral", "serious"): "You feel more focused than before.",
                ("alert", "happy"): "You sound calmer and happier now.",
                ("happy", "alert"): "Something seems to be bothering you."
            }

            line = transitions.get((prev, last))
            if line:
                speak(line, mood=last)
                return  # prevent double speaking

        # ------------------------------------------------------
        # DOMINANT MOOD ANALYSIS (last 5 moods)
        # ------------------------------------------------------
        moods = [h["mood"] for h in history[-5:]]
        freq = {m: moods.count(m) for m in set(moods)}
        dom = max(freq, key=freq.get)  # most common mood

        reflections = {
            "happy": [
                "You’ve been positive lately, it’s refreshing.",
                "I’ve noticed a cheerful tone from you, Yash."
            ],
            "serious": [
                "You’ve seemed focused lately, Yash.",
                "I sensed calm seriousness in our recent talks."
            ],
            "neutral": [
                "You’ve been balanced and calm recently.",
                "You’ve sounded steady — I like that consistency."
            ],
            "alert": [
                "You’ve been a bit tense lately. I hope you're doing okay.",
                "Your mood seems a little off — I'm here if you want to talk."
            ]
        }

        speak(random.choice(reflections.get(dom, reflections["neutral"])), mood=dom)
