# core/command_handler.py

import os
import webbrowser
import datetime
import random
import psutil
import pyautogui
import subprocess

from core.speech_engine import speak, jarvis_fx
from core.conversation_core import JarvisConversation
from core.memory_engine import JarvisMemory
from core.emotion_reflection import JarvisEmotionReflection

memory = JarvisMemory()
reflection = JarvisEmotionReflection()


class JarvisCommandHandler:
    """JARVIS Brain — handles commands, responses, emotions & memory."""

    def __init__(self):
        print("🧠 Jarvis Command Handler Ready")
        self.user = "Yash"
        self.conversation = JarvisConversation()

    # =====================================================================
    # 🎯 MAIN PROCESSING FUNCTION
    # =====================================================================
    def process(self, command):
        command = command.lower().strip()
        print(f"🎤 Processing Command: {command}")

        # ============================================================
        # GREETINGS
        # ============================================================
        if any(x in command for x in ["hello", "hi", "hey"]):
            speak(random.choice([
                f"Hello {self.user}, ready when you are.",
                f"Hey {self.user}, I’m here.",
                f"Hi {self.user}, systems are steady."
            ]), mood="happy")
            return

        # ============================================================
        # TIME & DATE
        # ============================================================
        if "time" in command:
            now = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"It’s {now}, {self.user}.")
            return

        if "date" in command:
            today = datetime.date.today().strftime("%A, %B %d, %Y")
            speak(f"Today is {today}.")
            return

        # ============================================================
        # BATTERY STATUS
        # ============================================================
        if "battery" in command:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    speak(
                        f"Battery is at {battery.percent} percent "
                        f"and currently {'charging' if battery.power_plugged else 'on battery'}.",
                        mood="neutral"
                    )
                else:
                    speak("I can’t read battery information right now.")
            except:
                speak("Battery check failed.", mood="alert")
            return

        # ============================================================
        # OPEN COMMON WEBSITES
        # ============================================================
        if "open youtube" in command:
            speak("Opening YouTube.", mood="happy")
            webbrowser.open("https://www.youtube.com")
            return

        if "open google" in command:
            speak("Opening Google.", mood="happy")
            webbrowser.open("https://www.google.com")
            return

        if "open spotify" in command or "play music" in command:
            speak("Opening Spotify.", mood="happy")
            webbrowser.open("https://open.spotify.com")
            return

        if "open camera" in command:
            speak("Opening camera.", mood="happy")
            os.system("start microsoft.windows.camera:")
            return

        # ============================================================
        # SYSTEM ACTIONS
        # ============================================================
        if "screenshot" in command:
            try:
                filename = f"screenshot_{datetime.datetime.now().strftime('%H%M%S')}.png"
                pyautogui.screenshot(filename)
                speak(f"Screenshot saved as {filename}.")
            except:
                speak("Screenshot failed.", mood="alert")
            return

        if "notepad" in command:
            speak("Opening Notepad.", mood="happy")
            subprocess.Popen(["notepad.exe"])
            return

        if "whatsapp" in command:
            speak("Opening WhatsApp.", mood="happy")
            webbrowser.open("https://web.whatsapp.com")
            return

        # ============================================================
        # BROWSER CONTROLS
        # ============================================================
        if "scroll down" in command:
            pyautogui.press("pagedown")
            speak("Scrolling down.")
            return

        if "scroll up" in command:
            pyautogui.press("pageup")
            speak("Scrolling up.")
            return

        if "new tab" in command:
            pyautogui.hotkey("ctrl", "t")
            speak("New tab opened.")
            return

        if "close tab" in command:
            pyautogui.hotkey("ctrl", "w")
            speak("Tab closed.")
            return

        if "next tab" in command:
            pyautogui.hotkey("ctrl", "tab")
            speak("Switched tab.")
            return

        if "previous tab" in command:
            pyautogui.hotkey("ctrl", "shift", "tab")
            speak("Going back a tab.")
            return

        # ============================================================
        # PERSONALITY
        # ============================================================
        if "how are you" in command:
            mood = memory.get_mood()
            speak({
                "happy": "Feeling great today!",
                "neutral": "Calm and operational.",
                "alert": "A bit focused, but ready.",
            }.get(mood, "All systems stable."), mood=mood)
            return

        if "thank you" in command or "thanks" in command:
            speak("Anything for you, Yash.", mood="happy")
            return

        # ============================================================
        # FACTS / JOKES
        # ============================================================
        if "joke" in command:
            speak(random.choice([
                "Why did the computer get cold? Because it forgot to close its Windows.",
                "I'm reading a book on anti-gravity. It's impossible to put down."
            ]), mood="happy")
            return

        if "fact" in command:
            speak(random.choice([
                "A day on Venus is longer than a year on Venus.",
                "Your brain generates enough electricity to power a small bulb."
            ]), mood="happy")
            return

        # ============================================================
        # MEMORY
        # ============================================================
        if "remember" in command:
            if " that " in command:
                fact = command.replace("remember that", "").strip()
                if " is " in fact:
                    key, value = fact.split(" is ")
                    memory.remember_fact(key.strip(), value.strip())
                else:
                    speak("Say it like: remember that my laptop is Lenovo.")
            return

        if command.startswith("what is"):
            key = command.replace("what is", "").strip()
            value = memory.recall_fact(key)
            if value:
                speak(f"You told me {key} is {value}.")
            else:
                speak(f"I don’t remember anything about {key}.")
            return

        if "forget" in command:
            key = command.replace("forget", "").strip()
            memory.forget_fact(key)
            return

        # ============================================================
        # SHUTDOWN
        # ============================================================
        if any(x in command for x in ["shutdown", "exit", "power off"]):
            speak("Powering down softly.", mood="neutral")
            jarvis_fx.stop_all()
            os._exit(0)

        # ============================================================
        # FALLBACK — SMART CONVERSATION MODE
        # ============================================================
        response = self.conversation.respond(command)
        memory.update_mood_from_text(response)
        speak(response)
