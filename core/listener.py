import speech_recognition as sr
import threading
import time
import random
from core.speech_engine import speak
from core.voice_effects import JarvisEffects
from core.command_handler import JarvisCommandHandler
from core.memory_engine import JarvisMemory

# Initialize modules
jarvis_fx = JarvisEffects()
memory = JarvisMemory()
handler = JarvisCommandHandler()


class JarvisListener:
    """Continuously listens for the wake word and processes user commands naturally."""

    def __init__(self):
        print("🎙 Initializing Jarvis Listener (Google Speech Engine)...")

        self.recognizer = sr.Recognizer()

        # SAFEST microphone detection
        try:
            self.microphone = sr.Microphone()
            print("🎧 Using default primary microphone")
        except Exception:
            print("⚠️ No microphone found.")
            raise

        self.listening = False
        self.running = True

        # 🎧 Calibrate once
        with self.microphone as source:
            print("🎧 Calibrating microphone... please wait")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ Microphone ready. Waiting for wake word...")

        threading.Thread(target=self._continuous_listen, daemon=True).start()

    # -------------------------------------------------------
    def _continuous_listen(self):
        """Background listener for the hotword."""
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=4)

                text = self._recognize_speech(audio)
                if not text:
                    continue

                print(f"🗣 Heard: {text}")

                wake_words = [
                    "hey jarvis", "ok jarvis", "okay jarvis",
                    "hi jarvis", "hello jarvis", "jarvis", 
                    "jarvis bolo", "jarvis haan"
                ]

                if any(word in text for word in wake_words):
                    self._activate_jarvis()

            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                print("⚠️ Google Speech API unreachable.")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Listener error: {e}")
                time.sleep(1)

    # -------------------------------------------------------
    def _recognize_speech(self, audio):
        """Convert speech to text."""
        try:
            return self.recognizer.recognize_google(audio).lower().strip()
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            speak("I’m having trouble connecting to Google’s servers right now.", mood="alert")
            return None
        except Exception as e:
            print(f"⚠️ Speech recognition error: {e}")
            return None

    # -------------------------------------------------------
    def _activate_jarvis(self):
        """Triggers command mode after hearing the hotword."""
        if self.listening:
            return

        self.listening = True
        print("\n🎯 Hotword detected — activating Jarvis...\n")

        # Soft tone but NOT startup tone (fix conflict with face scan ambient)
        jarvis_fx.play_ack()

        # Dynamic mood-based acknowledgment
        mood = memory.get_mood()
        responses = {
            "happy": [
                "Yes Yash, I’m listening!",
                "Hey Yashu, what’s up?",
                "I’m here, go ahead."
            ],
            "serious": [
                "Yes, I’m here, Yash. What’s next?",
                "Ready for your instruction.",
                "Go ahead — focused and ready."
            ],
            "neutral": [
                "Listening, Yash.",
                "I’m all ears.",
                "Yes, what can I do for you?"
            ]
        }

        speak(random.choice(responses.get(mood, responses["neutral"])), mood=mood)
        time.sleep(0.7)

        # Now record command
        try:
            with self.microphone as source:
                print("🎤 Listening for your command...")
                audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=8)

            command = self._recognize_speech(audio)
            if command:
                print(f"📡 Command recognized: {command}")
                handler.process(command)

                # FOLLOW-UP ONLY IF handler didn't speak already
                if not handler.conversation.last_was_long:
                    speak(random.choice([
                        "Done.", "Got it.", "Command executed.", "All set, Yash."
                    ]), mood="happy")
            else:
                speak("Sorry, I didn’t catch that.", mood="alert")

        except sr.WaitTimeoutError:
            speak("I didn’t hear anything, Yash.", mood="alert")
        except Exception as e:
            print(f"⚠️ Error while processing command: {e}")
            speak("Something went wrong while listening.", mood="alert")
        finally:
            self.listening = False
            print("🎧 Returning to standby mode.\n")

    # -------------------------------------------------------
    def stop(self):
        """Stop listener gracefully."""
        self.running = False
        print("🛑 Jarvis Listener stopped.")
