# core/background_listener.py
"""
Background Listener for Jarvis
- Runs continuously
- Listens for wake word
- Hands over control to command handler
"""

import speech_recognition as sr
import time
import threading
import traceback

from core.command_handler import JarvisCommandHandler
from core.speech_engine import speak

WAKE_WORDS = ["hey jarvis", "jarvis", "ok jarvis"]


class BackgroundListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.handler = JarvisCommandHandler()
        self.running = False

        # Recognizer tuning
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def start(self):
        if self.running:
            return

        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        print("🎧 Jarvis background listener started")

    def stop(self):
        self.running = False

    def _listen_loop(self):
        # Initial ambient noise calibration
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while self.running:
            try:
                with sr.Microphone() as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=None,
                        phrase_time_limit=5
                    )

                text = self.recognizer.recognize_google(audio).lower()
                print(f"🎤 Heard: {text}")

                if self._is_wake_word(text):
                    speak("Yes Yash?")
                    self._capture_command()

            except sr.UnknownValueError:
                pass
            except Exception as e:
                print("⚠️ Listener error:", e)
                time.sleep(1)

    def _is_wake_word(self, text):
        return any(w in text for w in WAKE_WORDS)

    def _capture_command(self):
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=8
                )

            command = self.recognizer.recognize_google(audio)
            print(f"🧠 Command: {command}")

            threading.Thread(
                target=self.handler.process,
                args=(command,),
                daemon=True
            ).start()

        except sr.UnknownValueError:
            speak("I didn't catch that.")
        except sr.WaitTimeoutError:
        # user did not speak in time – normal case
            pass
        except Exception:
            traceback.print_exc()
            speak("Something went wrong.")



__all__ = ["BackgroundListener"]
