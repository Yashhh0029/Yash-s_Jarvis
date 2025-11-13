# listener.py  — Drop-in replacement (upgraded)
import speech_recognition as sr
import threading
import time
import random
import webbrowser
import os
import platform
import pyautogui
import pygetwindow as gw
import keyboard

from core.speech_engine import speak
from core.voice_effects import JarvisEffects
from core.command_handler import JarvisCommandHandler
from core.memory_engine import JarvisMemory

# Attempt safer import of main (FACE_VERIFIED may or may not exist)
try:
    import main
except Exception:
    main = None

# Initialize modules
jarvis_fx = JarvisEffects()
memory = JarvisMemory()
handler = JarvisCommandHandler()


# Small helper: is face verified (reads main.FACE_VERIFIED if present)
def is_face_verified():
    try:
        if main and hasattr(main, "FACE_VERIFIED"):
            return bool(getattr(main, "FACE_VERIFIED"))
    except Exception:
        pass
    return False


# App shortcuts — extend as you like (Windows-centric defaults)
APP_COMMANDS = {
    "notepad": lambda: os.system("start notepad"),
    "calculator": lambda: os.system("start calc"),
    "chrome": lambda: os.system("start chrome"),
    "edge": lambda: os.system("start msedge"),
    "vscode": lambda: os.system("code"),
    "code": lambda: os.system("code"),
    "spotify": lambda: webbrowser.open("https://open.spotify.com"),
    # you can add full exe paths if needed
}


class JarvisListener:
    """Continuously listens for the wake word and processes commands naturally."""

    def __init__(self):
        print("🎙 Initializing Jarvis Listener (Google Speech Engine)...")

        self.recognizer = sr.Recognizer()

        try:
            self.microphone = sr.Microphone()
            print("🎧 Using default primary microphone")
        except Exception as e:
            print("⚠️ No microphone detected or microphone init failed:", e)
            raise

        self.listening = False
        self.running = True

        # Calibrate ambient noise once
        with self.microphone as source:
            print("🎧 Calibrating microphone (1s)...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                print("⚠️ Calibrate failed:", e)

        print("✅ Microphone ready — waiting for wake word.")
        threading.Thread(target=self._continuous_listen, daemon=True).start()

    # ---------------- background hotword loop ----------------
    def _continuous_listen(self):
        wake_words = [
            "hey jarvis", "ok jarvis", "okay jarvis",
            "hi jarvis", "hello jarvis", "jarvis",
            "jarvis bolo", "jarvis haan"
        ]

        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=4)

                text = self._recognize_speech(audio)
                if not text:
                    continue

                print(f"🗣 Heard: {text}")

                if any(w in text for w in wake_words):
                    # remove wake word and any immediate filler
                    cleaned = text
                    for w in wake_words:
                        cleaned = cleaned.replace(w, "").strip()
                    self._activate_jarvis(initial_command=cleaned)

            except sr.UnknownValueError:
                continue
            except Exception as e:
                print("⚠️ Listener error:", e)
                time.sleep(0.6)

    # ---------------- speech -> text ----------------
    def _recognize_speech(self, audio):
        try:
            return self.recognizer.recognize_google(audio).lower().strip()
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            speak("I’m having trouble connecting to Google’s servers right now.", mood="neutral")
            return None
        except Exception as e:
            print("⚠️ Speech recognition error:", e)
            return None

    # ---------------- hotword activation ----------------
    def _activate_jarvis(self, initial_command=None):
        if self.listening:
            return

        self.listening = True
        print("\n🎯 Hotword detected — activating Jarvis...\n")

        # Listening ping
        try:
            jarvis_fx.play_listening()
        except Exception:
            pass

        # Limited mode check
        if not is_face_verified():
            speak("Limited mode active. Your face wasn't verified earlier. I can still help.", mood="neutral")
        else:
            mood = memory.get_mood()
            responses = {
                "happy": ["Yes Yash, I’m listening!", "Hey Yashu, what’s up?", "I’m here — go ahead."],
                "serious": ["Yes, I’m here, Yash. What’s next?", "Ready for your instruction.", "Go ahead — focused and ready."],
                "neutral": ["Listening, Yash.", "I’m all ears.", "Yes, what can I do for you?"]
            }
            speak(random.choice(responses.get(mood, responses["neutral"])), mood=mood)

        time.sleep(0.4)

        # If user already spoke command after wake word: handle immediately
        if initial_command and len(initial_command) > 1:
            print("⚡ Immediate command after wakeword:", initial_command)
            self._process_command(initial_command)
            self.listening = False
            return

        # Otherwise listen for the full command
        try:
            with self.microphone as source:
                print("🎤 Listening for your command...")
                audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=10)

            command = self._recognize_speech(audio)
            self._process_command(command)

        except sr.WaitTimeoutError:
            speak("I didn’t hear anything, Yash.", mood="neutral")
        except Exception as e:
            print("⚠️ Error capturing command:", e)
            speak("Something went wrong while listening.", mood="neutral")

        self.listening = False
        print("🎧 Returning to standby mode.\n")

    # ---------------- process a recognized command ----------------
    def _process_command(self, command):
        if not command:
            speak("Sorry, I didn’t catch that.", mood="neutral")
            return

        print(f"📡 Command recognized: {command}")

        # Priority: search commands
        if any(k in command for k in ["search", "find", "look up", "dhund", "search kar"]):
            self._handle_search(command)
            return

        # Type message / type this
        if any(phrase in command for phrase in ["type", "type this", "type message", "type that"]):
            # If user already said "type <something>" treat rest as text
            # Else prompt user for what to type
            # Remove leading "type" words if they exist
            text = command
            for p in ["type this", "type message", "type that", "type", "type kar"]:
                text = text.replace(p, "").strip()
            if text:
                to_type = text
            else:
                speak("What should I type?", mood="neutral")
                to_type = self._listen_for_short_text()
                if not to_type:
                    speak("I didn’t get the text to type.", mood="neutral")
                    return

            self._auto_type_text(to_type)
            return

        # Tab & browser controls
        if any(kw in command for kw in ["new tab", "open tab", "close tab", "next tab", "previous tab", "prev tab", "switch tab"]):
            self._handle_tab_command(command)
            return

        # Open app or website
        if command.startswith("open ") or command.startswith("launch "):
            self._handle_open(command)
            return

        # Media controls
        if any(kw in command for kw in ["play", "pause", "play pause", "volume up", "volume down", "mute"]):
            self._handle_media(command)
            return

        # Default: pass to handler (command_handler)
        try:
            handler.process(command)
            last_was_long = getattr(handler.conversation, "last_was_long", False)
            if not last_was_long:
                speak(random.choice(["Done.", "Got it.", "All set, Yash."]), mood="happy")
        except Exception as e:
            print("⚠️ Handler.process error:", e)
            speak("I couldn't do that, Yash.", mood="neutral")

    # ---------------- typed input helper ----------------
    def _auto_type_text(self, text):
        """Type text into active window with typewriter sound fx per char."""
        try:
            speak(f"Typing: {text}", mood="neutral")
            # Focus active window and type slowly
            active = gw.getActiveWindow()
            if not active:
                time.sleep(0.2)  # still try typing
            # sometimes a quick click to focus helps
            try:
                active.activate()
                time.sleep(0.12)
            except Exception:
                pass

            for ch in text:
                pyautogui.write(ch)
                try:
                    jarvis_fx.typing_effect()
                except Exception:
                    pass
                time.sleep(0.03)
            time.sleep(0.05)
            speak("Typed.", mood="happy")
        except Exception as e:
            print("⚠️ Auto-type failed:", e)
            speak("I couldn't type that into the window.", mood="neutral")

    def _listen_for_short_text(self, timeout=6):
        """Listen for one short sentence to type (used by typing mode)."""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
            text = self._recognize_speech(audio)
            return text
        except Exception:
            return None

    # ---------------- context-aware search ----------------
    def _handle_search(self, command):
        """Context-aware searching with typewriter effect and sound."""
        try:
            # remove search keywords
            query = command
            for k in ["search", "find", "look up", "dhund", "search kar", "on youtube", "on google"]:
                query = query.replace(k, "")
            query = query.strip()
            if not query:
                speak("What would you like me to search for?", mood="neutral")
                query = self._listen_for_short_text()
                if not query:
                    speak("Okay, cancelled.", mood="neutral")
                    return

            print("🔍 Query:", query)

            active = gw.getActiveWindow()
            active_title = (active.title.lower() if active else "")
            print("🌐 Active window:", active_title)

            # Helper typewriter that also plays typing fx
            def typewrite_fx(text):
                for ch in text:
                    pyautogui.write(ch)
                    try:
                        jarvis_fx.typing_effect()
                    except Exception:
                        pass
                    time.sleep(0.03)

            # If youtube tab active, press '/' to focus search box
            if "youtube" in active_title:
                speak(f"Searching YouTube for {query}.", mood="happy")
                try:
                    pyautogui.press("/")
                    time.sleep(0.15)
                    typewrite_fx(query)
                    pyautogui.press("enter")
                except Exception:
                    # fallback to opening results page
                    webbrowser.open_new_tab(f"https://www.youtube.com/results?search_query={query}")
                return

            # If google tab active, try to focus search box
            if "google" in active_title or "bing" in active_title:
                speak(f"Searching Google for {query}.", mood="happy")
                try:
                    # ctrl+k or ctrl+l works in many browsers; use ctrl+l then type URL search to be robust
                    pyautogui.hotkey("ctrl", "l")
                    time.sleep(0.12)
                    typewrite_fx(f"https://www.google.com/search?q={query}")
                    pyautogui.press("enter")
                except Exception:
                    webbrowser.open_new_tab(f"https://www.google.com/search?q={query}")
                return

            # If user specifically asked YouTube
            if "youtube" in command:
                speak(f"Opening YouTube for {query}.", mood="happy")
                webbrowser.open_new_tab(f"https://www.youtube.com/results?search_query={query}")
                return

            # Default open new Google search tab
            speak(f"Searching the web for {query}.", mood="happy")
            webbrowser.open_new_tab(f"https://www.google.com/search?q={query}")

        except Exception as e:
            print("⚠️ Search handler exception:", e)
            speak("Couldn't perform that search, Yash.", mood="neutral")

    # ---------------- tab commands ----------------
    def _handle_tab_command(self, command):
        try:
            cmd = command.lower()
            # New tab
            if any(x in cmd for x in ["new tab", "open tab"]):
                speak("Opening a new tab.", mood="neutral")
                pyautogui.hotkey("ctrl", "t")
                return

            # Close tab
            if "close tab" in cmd or "close the tab" in cmd:
                speak("Closing current tab.", mood="neutral")
                pyautogui.hotkey("ctrl", "w")
                return

            # Next tab
            if any(x in cmd for x in ["next tab", "switch tab", "move to next tab"]):
                speak("Switching to next tab.", mood="neutral")
                pyautogui.hotkey("ctrl", "tab")
                return

            # Previous tab
            if any(x in cmd for x in ["previous tab", "prev tab", "previous", "last tab"]):
                speak("Switching to previous tab.", mood="neutral")
                pyautogui.hotkey("ctrl", "shift", "tab")
                return

            # fallback
            speak("Tab command not recognized.", mood="neutral")
        except Exception as e:
            print("⚠️ Tab command error:", e)
            speak("I couldn't change tabs, Yash.", mood="neutral")

    # ---------------- open/launch commands ----------------
    def _handle_open(self, command):
        try:
            cmd = command.lower()
            # open <app>
            words = cmd.replace("open ", "").replace("launch ", "").strip()

            # if user said "open youtube" or "open google"
            if "youtube" in words:
                speak("Opening YouTube.", mood="happy")
                webbrowser.open_new_tab("https://www.youtube.com")
                return
            if "google" in words:
                speak("Opening Google.", mood="happy")
                webbrowser.open_new_tab("https://www.google.com")
                return

            # check app mapping
            for name, fn in APP_COMMANDS.items():
                if name in words:
                    try:
                        speak(f"Opening {name}.", mood="neutral")
                        fn()
                        return
                    except Exception as e:
                        print("⚠️ App open failed:", e)
                        break

            # fallback: try to open as URL
            if words.startswith("http") or "." in words:
                speak(f"Opening {words}.", mood="happy")
                if not words.startswith("http"):
                    words = "https://" + words
                webbrowser.open_new_tab(words)
                return

            # fallback search
            speak(f"I couldn't find an app called {words}. Searching the web for it.", mood="neutral")
            webbrowser.open_new_tab(f"https://www.google.com/search?q={words}")

        except Exception as e:
            print("⚠️ Open handler error:", e)
            speak("I couldn't open that, Yash.", mood="neutral")

    # ---------------- media controls ----------------
    def _handle_media(self, command):
        cmd = command.lower()
        try:
            if any(k in cmd for k in ["play pause", "play/pause", "toggle play"]):
                # Media key
                keyboard.send("play/pause media")
                speak("Toggled play pause.", mood="neutral")
                return
            if "play " in cmd or cmd.strip() == "play":
                keyboard.send("play/pause media")
                speak("Play command sent.", mood="neutral")
                return
            if "pause" in cmd:
                keyboard.send("play/pause media")
                speak("Pause command sent.", mood="neutral")
                return
            if "volume up" in cmd or "increase volume" in cmd:
                keyboard.send("volume up")
                speak("Volume up.", mood="neutral")
                return
            if "volume down" in cmd or "decrease volume" in cmd:
                keyboard.send("volume down")
                speak("Volume down.", mood="neutral")
                return
            if "mute" in cmd:
                keyboard.send("volume mute")
                speak("Toggled mute.", mood="neutral")
                return
            # fallback
            speak("Media command not recognized.", mood="neutral")
        except Exception as e:
            print("⚠️ Media control error:", e)
            speak("I couldn't control the media, Yash.", mood="neutral")

    # ---------------- stop listener ----------------
    def stop(self):
        self.running = False
        print("🛑 Jarvis Listener stopped.")


# ---------- standalone run ----------
if __name__ == "__main__":
    l = JarvisListener()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        l.stop()
