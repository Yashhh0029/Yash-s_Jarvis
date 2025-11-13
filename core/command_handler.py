# core/command_handler.py
import os
import webbrowser
import datetime
import random
import psutil

from core.speech_engine import speak, jarvis_fx
from core.conversation_core import JarvisConversation
from core.memory_engine import JarvisMemory
from core.emotion_reflection import JarvisEmotionReflection

# Initialize core modules
memory = JarvisMemory()
reflection = JarvisEmotionReflection()


class JarvisCommandHandler:
    """Cinematic Jarvis — AI brain for all commands, mood logic, and contextual responses."""

    def __init__(self):
        print("🧠 Jarvis Command Handler Ready")
        self.name = "Jarvis"
        self.user = "Yash"
        self.conversation = JarvisConversation()

    # ----------------------------------------------------------
    def process(self, command):
        """Main logic — interpret and respond to user commands."""
        command = command.lower().strip()
        print(f"🎯 Command received: {command}")

        if not command:
            return

        # ==========================
        # GREETINGS
        # ==========================
        if any(word in command for word in ["hello", "hi", "hey", "good morning", "good evening"]):
            responses = [
                f"Hey {self.user}, nice to hear your voice again.",
                f"Hello {self.user}, how are you feeling today?",
                f"Hey there, {self.user}. Systems are calm and listening."
            ]
            speak(random.choice(responses), mood="happy")
            return

        # ==========================
        # TIME & DATE
        # ==========================
        elif "time" in command:
            now = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"It’s {now}, {self.user}.", mood="neutral")
            return

        elif "date" in command:
            today = datetime.date.today().strftime("%A, %B %d, %Y")
            speak(f"Today is {today}.", mood="neutral")
            return

        # ==========================
        # BATTERY STATUS
        # ==========================
        elif "battery" in command:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    percent = battery.percent
                    plugged = "charging" if battery.power_plugged else "on battery"
                    speak(f"Battery is at {percent} percent and currently {plugged}.", mood="serious")
                else:
                    speak("I couldn’t access the battery information right now.", mood="alert")
            except Exception:
                speak("Something went wrong while checking the battery.", mood="alert")
            return

        # ==========================
        # OPEN WEBSITES & APPS
        # ==========================
        elif "open youtube" in command:
            speak("On it, Yash. Launching YouTube.", mood="happy")
            webbrowser.open("https://www.youtube.com")
            return

        elif "open google" in command:
            speak("Opening Google. Ready to explore the web.", mood="happy")
            webbrowser.open("https://www.google.com")
            return

        elif "open spotify" in command or "play music" in command:
            speak("Let’s set the mood. Opening Spotify.", mood="happy")
            webbrowser.open("https://open.spotify.com")
            return

        elif "open camera" in command or "start camera" in command:
            speak("Activating your camera — let’s see that face.", mood="happy")
            os.system("start microsoft.windows.camera:")
            return

        # ==========================
        # SYSTEM UTILITIES
        # ==========================
        elif "screenshot" in command:
            try:
                import pyautogui
                file_name = f"screenshot_{datetime.datetime.now().strftime('%H%M%S')}.png"
                pyautogui.screenshot(file_name)
                speak(f"Screenshot captured and saved as {file_name}.", mood="happy")
            except Exception:
                speak("Couldn’t take a screenshot right now.", mood="alert")
            return

        # ==========================
        # PERSONALITY & REFLECTION
        # ==========================
        elif "how are you" in command:
            current_mood = memory.get_mood()
            mood_responses = {
                "happy": "I’m feeling great today, thanks to your vibe.",
                "serious": "I’m focused and ready, as always.",
                "neutral": "All systems calm and ready for your next idea."
            }
            speak(mood_responses.get(current_mood, "I’m balanced and alert."), mood=current_mood)
            return

        elif "how have i been" in command or "how was my mood" in command:
            reflection.reflect()
            return

        elif "who are you" in command:
            speak(f"I’m {self.name}, your personal AI — designed and fine-tuned by you, {self.user}.", mood="serious")
            return

        elif "thank you" in command or "thanks" in command:
            speak("Always for you, Yash.", mood="happy")
            return

        # ==========================
        # FACTS, FUN & QUOTES
        # ==========================
        elif "fact" in command:
            facts = [
                "Did you know? The human brain generates enough electricity to power an LED bulb.",
                "A day on Venus lasts longer than its year.",
                "The first computer mouse was made of wood.",
                "Octopuses have three hearts and blue blood."
            ]
            speak(random.choice(facts), mood="happy")
            return

        elif "joke" in command:
            jokes = [
                "Why did the AI break up with its computer? It had too many bugs.",
                "My WiFi told me a joke, but it wasn’t connected.",
                "Sometimes I debug myself just to feel something."
            ]
            speak(random.choice(jokes), mood="happy")
            return

        elif "quote" in command:
            quotes = [
                "The future belongs to those who code it.",
                "Don’t count the days, make the days count.",
                "Power isn’t in circuits, it’s in focus and consistency."
            ]
            speak(random.choice(quotes), mood="serious")
            return

        # ==========================
        # EMOTION RECOGNITION
        # ==========================
        elif any(x in command for x in ["analyze face", "read my face", "emotion"]):
            try:
                from core.face_emotion import FaceEmotionAnalyzer
                analyzer = FaceEmotionAnalyzer()
                emotion = analyzer.capture_emotion()
                if emotion:
                    memory.set_mood(emotion)
                    reflection.add_emotion(memory.get_mood())
            except Exception as e:
                print(f"⚠️ Face analysis error: {e}")
                speak("I couldn’t analyze your face right now.", mood="alert")
            return

        # ==========================
        # MEMORY INTERACTIONS
        # ==========================
        elif "remember that" in command or "remember" in command:
            parts = command.replace("remember that", "").replace("remember", "").strip().split(" is ")
            if len(parts) == 2:
                key, value = parts
                memory.remember_fact(key.lower().strip(), value.lower().strip())
            else:
                speak("Say it like — remember that my laptop is Lenovo.", mood="alert")
            return

        elif "what is" in command:
            key = command.replace("what is", "").strip().lower()
            value = memory.recall_fact(key)
            if value:
                speak(f"You told me that {key} is {value}.", mood="happy")
            else:
                speak(f"I don’t remember anything about {key}.", mood="alert")
            return

        elif "forget" in command:
            key = command.replace("forget", "").strip().lower()
            memory.forget_fact(key)
            return

        # ==========================
        # SYSTEM SHUTDOWN
        # ==========================
        elif any(x in command for x in ["shutdown", "exit", "power off", "sleep now"]):
            speak("Alright, Yash. Powering down softly.", mood="serious")
            jarvis_fx.stop_all()
            os._exit(0)

        # ==========================
        # FALLBACK TO CONVERSATION
        # ==========================
        else:
            # conversation response + mood tracking
            response = self.conversation.respond(command)
            memory.update_mood_from_text(response)
