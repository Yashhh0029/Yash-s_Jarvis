import json
import os
import random
from core.speech_engine import speak


class JarvisMemory:
    """Stores Jarvis’s emotional context, facts, and conversational memory."""

    def __init__(self):
        self.file_path = os.path.join(os.path.dirname(__file__), "..", "config", "memory.json")

        # Default structure (expanded)
        self.memory = {
            "facts": {},
            "mood": "neutral",
            "last_topic": None,
            "emotion_history": []          # <-- CRITICAL FIX
        }

        self._load_memory()
        self._validate_structure()         # <-- ensures missing keys are fixed
        print("🧠 Memory Engine Initialized")

    # -------------------------------------------------------
    def _validate_structure(self):
        """
        Ensures required keys always exist.
        Prevents crashes if JSON was edited manually.
        """
        changed = False

        if "facts" not in self.memory:
            self.memory["facts"] = {}
            changed = True

        if "mood" not in self.memory:
            self.memory["mood"] = "neutral"
            changed = True

        if "last_topic" not in self.memory:
            self.memory["last_topic"] = None
            changed = True

        if "emotion_history" not in self.memory:     # <-- prevents reflection crash
            self.memory["emotion_history"] = []
            changed = True

        if changed:
            self._save_memory()

    # -------------------- LOAD / SAVE --------------------
    def _load_memory(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = {
                    "facts": {},
                    "mood": "neutral",
                    "last_topic": None,
                    "emotion_history": []
                }

    def _save_memory(self):
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    # -------------------- FACT MEMORY --------------------
    def remember_fact(self, key, value):
        self.memory["facts"][key.lower()] = value
        self._save_memory()
        speak(f"Got it, Yash. I'll remember that {key} is {value}.", mood="happy")

    def recall_fact(self, key):
        return self.memory["facts"].get(key.lower(), None)

    def forget_fact(self, key):
        if key.lower() in self.memory["facts"]:
            del self.memory["facts"][key.lower()]
            self._save_memory()
            speak(f"Alright, I’ll forget about {key}.", mood="serious")
        else:
            speak(f"I don’t think you ever told me about {key}.", mood="alert")

    # -------------------- MOOD SYSTEM --------------------
    def set_mood(self, mood):
        """Stores the user’s current emotional mood."""
        self.memory["mood"] = mood
        self._save_memory()

    def get_mood(self):
        return self.memory.get("mood", "neutral")

    def emotional_response(self, mood):
        """Responds based on current emotional state."""
        responses = {
            "happy": [
                "I'm feeling great today, Yash!",
                "Still smiling from our last chat!",
                "You’ve kept me in a really good mood lately."
            ],
            "serious": [
                "Focused as always, Yash.",
                "Keeping things calm and collected.",
                "Just staying sharp and ready to help."
            ],
            "neutral": [
                "Everything’s running smoothly.",
                "All systems calm and steady.",
                "I’m here, relaxed and ready."
            ],
            "sad": [
                "Feeling a bit low today, Yash.",
                "Not at my brightest, but I’ll manage.",
                "You seem quiet too — maybe we both need some music?"
            ],
            "alert": [
                "Something caught my attention.",
                "I’m fully awake and observing.",
                "Alert mode on — let’s stay sharp."
            ]
        }

        speak(random.choice(responses.get(mood, ["I’m feeling neutral right now."])), mood=mood)

    # -------------------- TOPIC MEMORY --------------------
    def update_topic(self, topic):
        """Remembers last topic user talked about (used for context)."""
        self.memory["last_topic"] = topic
        self._save_memory()

    def get_last_topic(self):
        return self.memory.get("last_topic", None)

