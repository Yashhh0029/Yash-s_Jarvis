# PART 1/4 — core/listener.py
# Single-mic background listener — final upgraded version (Part 1/4)

import speech_recognition as sr
import threading
import time
import random
import webbrowser
import os
import pyautogui
import pygetwindow as gw
import keyboard
import traceback
import re
import subprocess
from queue import Queue, Empty
from typing import Optional

# Local modules (best-effort imports)
from core.speech_engine import speak
from core.voice_effects import JarvisEffects
from core.command_handler import JarvisCommandHandler
from core.memory_engine import JarvisMemory

# --- Music intent & search platform configuration ---

# Words that clearly mean "Jarvis should handle music",
# not just press media keys.
MUSIC_INTENT_KEYWORDS = [
    "song",
    "music",
    "playlist",
    "spotify",
    "youtube",
    "yt",
    "gaana",
    "jio saavn",
    "jiosaavn",
    "wynk",
    "local",
]

# Platforms used by _handle_search
PLATFORM_SEARCH_URLS: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "wikipedia": "https://en.wikipedia.org/wiki/{q}",
}


# Optional integrations (tolerate missing)
try:
    import core.brain as brain_module
except Exception:
    brain_module = None

try:
    import core.sleep_manager as sleep_manager
except Exception:
    sleep_manager = None

try:
    import core.state as state
except Exception:
    # minimal dummy state so attribute access won't crash
    class _State:
        pass
    state = _State()

try:
    import core.voice_effects as voice_effects
except Exception:
    voice_effects = None

# singletons
try:
    jarvis_fx = JarvisEffects()
except Exception:
    # fallback no-op
    class _NoFx:
        def play_listening(self):
            pass

        def play_success(self):
            pass

        def play_ambient(self):
            pass

        def typing_effect(self):
            pass

    jarvis_fx = _NoFx()

memory = JarvisMemory()
handler = JarvisCommandHandler()

# ---------------- FINAL WAKE WORDS (confirmed by you) ----------------
# Will only match whole words / phrases using regex with word boundaries.
_WAKE_PHRASES = [
    "hey jarvis",
    "oye jarvis",
    "okay jarvis",
    "i am back jarvis",
    "okay i am back",
    "jarvis i am here",
    "jarvis",
    "jar","hey dude","dude"
]

# compile regex patterns for robust matching (case-insensitive)
_WAKE_PATTERNS = [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in _WAKE_PHRASES]

# tuning constants
_WAKE_TIMEOUT = 5
_WAKE_PHRASE_TIME_LIMIT = 7

_ACTIVE_TIMEOUT = 8
_ACTIVE_PHRASE_TIME_LIMIT = 10

_DEFAULTS = {
    "energy_threshold": 450,
    "dynamic_energy_threshold": True,
    "pause_threshold": 0.8,
    "non_speaking_duration": 0.6,
}

# ---------------- Desktop Intent Resolver ----------------
def resolve_desktop_intent(text: str):
    t = text.lower()

    if any(p in t for p in ["volume up", "increase volume", "louder"]):
        return {"intent": "volume", "action": "up"}

    if any(p in t for p in ["volume down", "lower volume", "quieter"]):
        return {"intent": "volume", "action": "down"}

    if "mute" in t:
        return {"intent": "volume", "action": "mute"}

    if any(p in t for p in ["show desktop", "take me to desktop", "minimize all"]):
        return {"intent": "desktop", "action": "show"}

    if any(p in t for p in ["close this", "close window", "close app"]):
        return {"intent": "window", "action": "close"}

    if any(p in t for p in ["switch window", "change window"]):
        return {"intent": "window", "action": "switch"}

    if any(p in t for p in ["open downloads", "open download folder"]):
        return {"intent": "folder", "name": "downloads"}

    if any(p in t for p in ["open documents", "open document folder"]):
        return {"intent": "folder", "name": "documents"}

    # ✅ STEP 1 — generic folder open (ADD THIS HERE)
    m = re.search(r"open\s+(.*?)\s+folder", t)
    if m:
        return {
            "intent": "folder_search",
            "name": m.group(1).strip()
        }

    return None

# active-mode inactivity before going back to wake-only
ACTIVE_INACTIVITY_DEFAULT = 70

# ---------------- Intent Cleaning Layer ----------------
def clean_intent_text(text: str) -> str:
    if not text:
        return ""

    t = text.lower().strip()

    # remove wake words
    wake_words = [
        "jarvis",
        "hey jarvis",
        "oye jarvis",
        "okay jarvis",
        "jar",
    ]
    for w in wake_words:
        t = re.sub(rf"\b{re.escape(w)}\b", "", t)

    # remove command verbs
    command_words = [
        "search",
        "find",
        "look up",
        "open",
        "launch",
        "play",
        "start",
        "type",
        "go to",
        "show",
    ]
    for c in command_words:
        t = re.sub(rf"\b{re.escape(c)}\b", "", t)

    # remove platform words
    platforms = [
        "on youtube",
        "youtube",
        "google",
        "bing",
        "amazon",
        "flipkart",
        "wikipedia",
    ]
    for p in platforms:
        t = re.sub(rf"\b{re.escape(p)}\b", "", t)

    # normalize whitespace
    t = re.sub(r"\s+", " ", t)

    return t.strip()

class JarvisListener:
    """
    Robust listener using listen_in_background + single consumer thread.
    Paste all 4 parts in order to replace core/listener.py
    """

    def __init__(self, active_inactivity_timeout: int = ACTIVE_INACTIVITY_DEFAULT):
        print("🎙 Initializing Jarvis Listener (Google STT — final upgraded)...")
        self.recognizer = sr.Recognizer()
        self._init_recognizer_defaults()
        self._last_intent = None
        
        # browser tab tracking (best-effort, logical index)
        self._current_tab_index = 1
        self._max_tabs_guess = 15  # soft limit to avoid infinite loops


        # Microphone object (single)
        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            print("⚠️ Microphone init failed:", e)
            raise
        # short-term intent context (for follow-ups like "play it")
        self._last_intent = None


        # locks & flags
        self._lock = threading.RLock()
        self._active_mode_lock = threading.RLock()
        self.running = True

        # active-mode state
        self.active_mode = False
        self.listening = False
        self.active_inactivity_timeout = int(active_inactivity_timeout)
        self._last_active_command_ts = 0.0

        # speaking flag to avoid TTS pickup
        self._is_speaking = False
        self._speak_lock = threading.Lock()

        # debounce repeated wake fragments
        self._last_wake_ts = 0.0
        self._wake_debounce_seconds = 1.0

        # audio queue (background callback → consumer)
        self._audio_queue: "Queue[Optional[sr.AudioData]]" = Queue(maxsize=40)

        # background listener handle (callable to stop)
        self._bg_stop_fn = None

        # consumer thread
        self._consumer_thread = threading.Thread(
            target=self._audio_consumer_loop,
            daemon=True,
            name="JarvisAudioConsumer",
        )
        self._consumer_thread.start()

        # Ensure state.SYSTEM_SPEAKING exists if possible
        try:
            setattr(state, "SYSTEM_SPEAKING", bool(getattr(state, "SYSTEM_SPEAKING", False)))
        except Exception:
            pass

        print("✅ Microphone ready — starting background listener and waiting for wake word.")

        # best-effort start sleep manager
        try:
            if sleep_manager and hasattr(sleep_manager, "start_manager"):
                sleep_manager.start_manager()
        except Exception:
            print("⚠️ sleep_manager.start_manager() failed (ignored).")

        # Start background listener (keeps mic stream open)
        try:
            self._bg_stop_fn = self.recognizer.listen_in_background(
                self.microphone,
                self._background_callback,
                phrase_time_limit=_WAKE_PHRASE_TIME_LIMIT,
            )
        except Exception as e:
            print("⚠️ listen_in_background failed:", e)
            traceback.print_exc()
            self._bg_stop_fn = None
            

    # PART 2/4 — core/listener.py (continued)

    # ---------------- recognizer init & safety ----------------
    def _init_recognizer_defaults(self):
        """Set safe values and do a one-time ambient calibration (best-effort)."""
        try:
            self.recognizer.energy_threshold = int(_DEFAULTS.get("energy_threshold", 300))
            self.recognizer.dynamic_energy_threshold = bool(_DEFAULTS.get("dynamic_energy_threshold", True))

            pause = float(_DEFAULTS.get("pause_threshold", 0.6))
            non_speaking = float(_DEFAULTS.get("non_speaking_duration", 0.3))
            if pause < non_speaking:
                pause = non_speaking + 0.2

            try:
                self.recognizer.pause_threshold = pause
            except Exception:
                pass

            try:
                setattr(self.recognizer, "non_speaking_duration", non_speaking)
            except Exception:
                pass

            # ambient calibration (best-effort)
            try:
                with sr.Microphone() as src:
                    self.recognizer.adjust_for_ambient_noise(src, duration=1)
            except Exception:
                pass

             # ✅ ADD THIS PART (CLAMP SENSITIVITY)
            try:
                if self.recognizer.energy_threshold < 250:
                   self.recognizer.energy_threshold = 250
                elif self.recognizer.energy_threshold > 800:
                   self.recognizer.energy_threshold = 800
            except Exception:
                pass

        except Exception as e:
            print("⚠️ _init_recognizer_defaults failed:", e)
            traceback.print_exc()

    # ---------------- background callback ----------------
    def _background_callback(self, recognizer, audio):
        """Called by listen_in_background — push audio to queue without blocking."""
        try:
            # push non-blocking; drop if queue full
            self._audio_queue.put_nowait(audio)
        except Exception:
            # queue full or other error — drop safely
            pass

    # ---------------- audio consumer loop ----------------
    def _audio_consumer_loop(self):
        """
        Consumer: pull audio chunks, recognize them, and route either:
         - to wake detection (when not active)
         - to active command processing (when active)
        """
        while self.running:
            try:
                audio = self._audio_queue.get(timeout=0.4)
            except Empty:
                # check for inactivity
                if self.active_mode and (time.time() - self._last_active_command_ts > self.active_inactivity_timeout):
                    self._exit_active_mode()
                continue

            # skip processing if Jarvis is speaking
            if getattr(state, "SYSTEM_SPEAKING", False) or self._is_speaking:
                continue

            text = None
            try:
                text = self._recognize_from_audio(audio)
            except Exception as e:
                # safe continue on any recognition exception
                print("⚠️ recognition exception:", e)
                continue

            if not text:
                continue

            normalized = text.lower().strip()
            print(f"🗣 Heard: {normalized}")

            # If active, treat everything as command input
            if self.active_mode:
                self._last_active_command_ts = time.time()
                try:
                    self._process_command(normalized)
                except Exception as e:
                    print("⚠️ active command error:", e)
                continue

            # Not active → check wake phrases (use whole-phrase regex)
            triggered = False
            for pat in _WAKE_PATTERNS:
                if pat.search(normalized):
                    triggered = True
                    break

            if triggered:
                now = time.time()
                # debounce fast repeats like "jar jar jar" fragments
                if now - self._last_wake_ts < self._wake_debounce_seconds:
                    continue
                self._last_wake_ts = now

                # remove the matched wake phrase(s) from the text to get any embedded command
                cleaned = normalized
                for pat in _WAKE_PATTERNS:
                    cleaned = pat.sub("", cleaned).strip()

                # if in sleep mode, wake properly
                if getattr(state, "MODE", "active") == "sleep":
                    try:
                        self._wake_from_sleep()
                    except Exception:
                        pass
                    time.sleep(0.12)

                # spawn active mode thread (non-blocking)
                threading.Thread(
                    target=self._enter_active_command_mode,
                    args=(cleaned,),
                    daemon=True,
                ).start()

    def _exit_active_mode(self):
        """
        Resets the assistant to passive mode if the user stops talking.
        """
        with self._active_mode_lock:
            if not self.active_mode:
                return
            print("⏳ Silence detected. Exiting active mode.")
            self.active_mode = False
            self.listening = False
        try:
            state.LAST_INTERACTION = time.time()
        except Exception:
            pass

    # ---------------- recognize wrapper ----------------
    def _recognize_from_audio(self, audio, retries=1):
        if not audio:
            return None
        try:
            text = self.recognizer.recognize_google(audio)
            if text:
                return text.lower().strip()
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print("⚠️ STT RequestError:", e)
            return None
        except AssertionError as ae:
            # reapply safe defaults and retry once
            print("⚠️ SR AssertionError during recognition:", ae)
            self._init_recognizer_defaults()
            if retries > 0:
                time.sleep(0.08)
                return self._recognize_from_audio(audio, retries=retries - 1)
            return None
        except Exception as e:
            if retries > 0:
                time.sleep(0.12)
                return self._recognize_from_audio(audio, retries=retries - 1)
            print("⚠️ _recognize_from_audio failed:", e)
            traceback.print_exc()
            return None

    # ---------------- active-mode entry ----------------
    def _enter_active_command_mode(self, initial_command=None):
        with self._active_mode_lock:
            if self.active_mode:
                # already active — process initial command if present
                if initial_command:
                    try:
                        self._process_command(initial_command)
                    except Exception:
                        pass
                return
            self.active_mode = True
            self.listening = True
            self._last_active_command_ts = time.time()

        # update last interaction
        try:
            state.LAST_INTERACTION = time.time()
        except Exception:
            pass

        # play listening FX
        try:
            jarvis_fx.play_listening()
        except Exception:
            pass

        # speak wake line (face-aware + brain)
        try:
            mood = memory.get_mood()
            last_topic = getattr(state, "LAST_TOPIC", None)
            if is_face_verified():
                if brain_module and hasattr(brain_module.brain, "generate_wakeup_line"):
                    speak(brain_module.brain.generate_wakeup_line(mood=mood, last_topic=last_topic), mood=mood)
                else:
                    speak("Yes Yash, I’m listening.", mood=mood)
            else:
                # limited mode: if face not verified, be explicit
                speak("Limited mode active. I couldn't verify your face earlier.", mood="neutral")
        except Exception:
            speak("Yes Yash, I’m here.", mood="neutral")

        # process an embedded short command that followed the wake phrase
        if initial_command and len(initial_command.strip()) > 0:
            try:
                self._process_command(initial_command)
            except Exception as e:
                print("⚠️ error processing initial embedded command:", e)

        # active mode now stays alive until consumer thread times out inactivity
        try:
            while self.active_mode and self.running:
                time.sleep(0.25)
        finally:
            with self._active_mode_lock:
                self.listening = False
                self.active_mode = False
                try:
                    state.LAST_INTERACTION = time.time()
                except Exception:
                    pass

    # PART 3/4 — core/listener.py (continued)

    # -------------------------------------------------------
    # CORE COMMAND PROCESSOR (ROUTER ONLY)
    # -------------------------------------------------------
    def process_command(self, command):
        """
        Public API for executing commands.
        Other modules must call this method.
        """
        return self._process_command(command)

    def _process_command(self, command):
        """
        Listener responsibility ONLY:
        - normalize command
        - explain failures
        - route execution to command_handler
        """

        # ---------------- BASIC VALIDATION ----------------
        if not command or not command.strip():
            speak("Sorry, I didn’t catch that.", mood="neutral")
            return

        command = command.lower().strip()

        # ---------------- WHY / FAILURE EXPLANATION ----------------
        if any(p in command for p in [
            "why didn't you",
            "why did you not",
            "why you couldn't",
            "what went wrong",
            "why failed",
        ]):
            last = memory.get_last_action()

            if not last:
                speak("I don't recall any recent action.", mood="neutral")
                return

            if last.get("status") == "success":
                speak("That action was completed successfully.", mood="neutral")
                return

            reason = last.get("reason", "something unexpected happened")
            speak(f"I couldn't complete it because {reason}.", mood="neutral")
            return

        # ---------------- REMOVE WAKE WORDS ----------------
        command = re.sub(
            r"\b(jarvis|hey jarvis|oye jarvis|okay jarvis|jar)\b",
            "",
            command
        ).strip()

        command = re.sub(r"\s+", " ", command)

        if not command:
            speak("Yes?", mood="neutral")
            return

        # ---------------- SLEEP MODE GUARD ----------------
        if getattr(state, "MODE", "active") == "sleep":
            speak("Say 'Hey Jarvis' to wake me completely.", mood="neutral")
            return

        # ---------------- UPDATE ACTIVITY ----------------
        try:
            state.LAST_INTERACTION = time.time()
            self._last_active_command_ts = time.time()
        except Exception:
            pass

        print(f"📡 Routed Command → {command}")

        # ---------------- SINGLE EXECUTION HANDOFF ----------------
        try:
            handler.process(command)
            return
        except Exception as e:
            print("⚠️ handler error:", e)
            traceback.print_exc()
            speak("I couldn't do that, Yash.", mood="neutral")
            return

    # -------------------------------------------------------
    # AUTO-TYPING
    def _auto_type_text(self, text):
        try:
            speak(f"Typing: {text}", mood="neutral")
            active = gw.getActiveWindow()
            if not active:
                pyautogui.click(300, 300)
            else:
                try:
                    active.activate()
                except Exception:
                    pass

            for ch in text:
                pyautogui.write(ch)
                try:
                    jarvis_fx.typing_effect()
                except Exception:
                    pass
                time.sleep(0.03)

            speak("Typed.", mood="happy")
        except Exception as e:
            print("⚠️ _auto_type_text error:", e)
            traceback.print_exc()
            speak("I couldn't type that.", mood="neutral")

    # -------------------------------------------------------
    # Listen for a short follow-up while active (uses queue)
    def _listen_for_short_text(self, timeout=6):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                audio = self._audio_queue.get(timeout=0.5)
            except Empty:
                continue
            text = self._recognize_from_audio(audio)
            if text:
                return text
        return None

    def _click_first_youtube_result(self, index=1, wait_seconds=2.5):
        """
        After a YouTube search, click roughly where the first result is.
        Prefer the window whose title matches the last search query.
        """
        try:
            # give page time to load results
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            target_window = None
            windows = []
            try:
                windows = gw.getAllWindows()
            except Exception:
                windows = []

            # build keywords from last search query
            last_query = getattr(self, "_last_search_query", None)
            if last_query:
                keywords = [w for w in last_query.split() if len(w) > 2]
            else:
                keywords = []

            for w in windows:
                title = (w.title or "").lower()
                if "youtube" not in title:
                    continue

                # Prefer youtube window whose title contains search keywords
                if keywords and any(k in title for k in keywords):
                    target_window = w
                    break

                # Fallback: remember first youtube window
                if target_window is None:
                    target_window = w

            if target_window is None:
                # fallback: use active window if nothing found
                target_window = gw.getActiveWindow()
                if not target_window:
                    return False

            try:
                target_window.activate()
                time.sleep(0.2)
            except Exception:
                pass

            wx, wy, ww, wh = target_window.left, target_window.top, target_window.width, target_window.height

            # Approx area of nth video result (dynamic index-based)
            x = int(wx + ww * 0.35)
            y = int(wy + wh * (0.30 + (index - 1) * 0.12))

            pyautogui.click(x, y)
            speak(f"Playing result number {index}.", mood="happy")
            return True
        except Exception as e:
            print("⚠️ _click_first_youtube_result error:", e)
            traceback.print_exc()
            return False


    def _handle_play_it_followup(self,index=1):
        """
        Follow-up command: 'play it' / 'play this' / 'start it'
        → click first result on YouTube search page if available.
        """
        try:
            active = gw.getActiveWindow()
            if not active:
                speak("I don't see anything to play.", mood="neutral")
                return

            title = (active.title or "").lower()
            if "youtube" not in title:
                speak("I can auto-play only from YouTube search results right now.", mood="neutral")
                return

            ok = self._click_first_youtube_result(index=index, wait_seconds=0.0)

            if not ok:
                speak("I couldn't click the result.", mood="neutral")
        except Exception as e:
            print("⚠️ _handle_play_it_followup error:", e)
            traceback.print_exc()
            speak("I couldn't play that.", mood="neutral")

    
    # -------------------------------------------------------
    # SEARCH HANDLER
    def _handle_search(self, command):
        try:
            # ---- RESET SEARCH CONTEXT (CRITICAL) ----
            self._last_search_query = None
            self._last_intent = None
            original = command
            query = command
            for k in [
                "search",
                "find",
                "look up",
                "dhund",
                "search kar",
                "on youtube",
                "on google",
                "on flipkart",
                "on amazon",
                "on bing",
            ]:
                query = query.replace(k, "")
            query = clean_intent_text(query)
            self._last_search_query = query.lower()


            # detect platform
            platform = None
            for p in PLATFORM_SEARCH_URLS.keys():
                if p in original:
                    platform = p
                    break

            if not query:
                speak("What should I search?", mood="neutral")
                query = self._listen_for_short_text()
                if not query:
                    speak("Search cancelled.", mood="neutral")
                    memory.record_action(
                    intent="search",
                    command=command,
                    status="failed",
                    reason="user cancelled the search"
                    )
                    return

            def type_fx(t):
                for ch in t:
                    pyautogui.write(ch)
                    try:
                        jarvis_fx.typing_effect()
                    except Exception:
                        pass
                    time.sleep(0.03)

            # platform fast path
            if platform:
                win = self._find_and_activate_window(platform)
                if win:
                    try:
                        wx, wy, ww, wh = win.left, win.top, win.width, win.height
                        pyautogui.click(int(wx + ww * 0.5), int(wy + wh * 0.12))
                        time.sleep(0.18)
                        type_fx(query)
                        pyautogui.press("enter")
                        speak(f"Searching {platform} for {query}", mood="happy")
                        memory.record_action(
                        intent="search",
                        command=command,
                        status="success"
                        )
                        return
                    except Exception:
                        pass
                url = PLATFORM_SEARCH_URLS[platform].format(q=query.replace(" ", "+"))
                speak(f"Opening {platform} results for {query}.", mood="happy")
                webbrowser.open_new_tab(url)
                memory.record_action(
                intent="search",
                command=command,
                status="success"
                )
                return

            # if active browser window — search there
            active = gw.getActiveWindow()
            title = (active.title or "").lower() if active else ""

            if "youtube" in title:
                try:
                    pyautogui.press("/")
                    time.sleep(0.12)
                    type_fx(query)
                    pyautogui.press("enter")
                    speak(f"Searching YouTube for {query}", mood="happy")
                    return

                    # store intent context for follow-ups
                    self._last_intent = {
                    "type": "youtube_search" if platform == "youtube" else "web_search",
                    "query": query,
                    "platform": platform or "google",
                    "timestamp": time.time(),
                    }
                    memory.record_action(
                    intent="search",
                    command=command,
                    status="success"
                    )
                    return
                except Exception:
                    pass

            if any(b in title for b in ["chrome", "edge", "firefox", "brave", "msedge"]):
                try:
                    speak(f"Searching Google for {query}", mood="happy")
                    pyautogui.hotkey("ctrl", "l")
                    time.sleep(0.08)
                    pyautogui.hotkey("ctrl", "a")
                    pyautogui.press("backspace")
                    type_fx(query)
                    pyautogui.press("enter")
                    memory.record_action(
                    intent="search",
                    command=command,
                    status="success"
                    )
                    return
                except Exception:
                    pass

            # fallback click & type
            if active:
                try:
                    wx, wy, ww, wh = active.left, active.top, active.width, active.height
                    pyautogui.click(int(wx + ww * 0.5), int(wy + wh * 0.12))
                    time.sleep(0.12)
                    type_fx(query)
                    pyautogui.press("enter")
                    speak("Searching.", mood="happy")
                    memory.record_action(
                    intent="search",
                    command=command,
                    status="success"
                    )
                    return
                except Exception:
                    pass

            # final fallback: open web search
            speak(f"Searching Google for {query}.", mood="happy")
            webbrowser.open_new_tab(PLATFORM_SEARCH_URLS["google"].format(q=query.replace(" ", "+")))
            memory.record_action(
            intent="search",
            command=command,
            status="success"
            )

            self._last_intent = {
              "type": "web_search",
               "query": query,
               "timestamp": time.time()
              }

        except Exception as e:
            memory.record_action(
                intent="search",
                command=command,
                status="failed",
                reason=str(e)
            )
            print("⚠️ Search Error:", e)
            traceback.print_exc()
            speak("Couldn't search that, Yash.", mood="neutral")


    # PART 4/4 — core/listener.py (final)

    # -------------------------------------------------------
    # FIND WINDOW
    def _find_and_activate_window(self, keyword):
        try:
            keyword = keyword.lower()
            for w in gw.getAllWindows():
                title = (w.title or "").lower()
                if keyword in title:
                    try:
                        w.activate()
                        time.sleep(0.18)
                        return w
                    except Exception:
                        continue
        except Exception as e:
            print("⚠️ _find_and_activate_window error:", e)
            traceback.print_exc()
        return None


    # TAB CONTROL
    def _handle_tab_command(self, cmd):
        try:
            cmd = cmd.lower()

            # open new tab
            if "new tab" in cmd or "open tab" in cmd:
                pyautogui.hotkey("ctrl", "t")
                self._current_tab_index += 1
                speak("Opened a new tab.", mood="neutral")
                return

            # close current tab
            if "close tab" in cmd or "close current tab" in cmd:
                pyautogui.hotkey("ctrl", "w")
                self._current_tab_index = max(1, self._current_tab_index - 1)
                speak("Closed the current tab.", mood="neutral")
                return

            # close all tabs
            if "close all tabs" in cmd:
                for _ in range(self._max_tabs_guess):
                    pyautogui.hotkey("ctrl", "w")
                    time.sleep(0.05)
                self._current_tab_index = 1
                speak("Closed all tabs.", mood="neutral")
                return

            # next tab
            if "next tab" in cmd:
                pyautogui.hotkey("ctrl", "tab")
                self._current_tab_index += 1
                speak("Next tab.", mood="neutral")
                return

            # previous tab
            if "previous tab" in cmd or "prev tab" in cmd:
                pyautogui.hotkey("ctrl", "shift", "tab")
                self._current_tab_index = max(1, self._current_tab_index - 1)
                speak("Previous tab.", mood="neutral")
                return

            # go to nth tab (1st, 2nd, 3rd...)
            m = re.search(r"\b(\d+)(st|nd|rd|th)?\s+tab\b", cmd)
            if m:
                target = int(m.group(1))
                if target < 1 or target > self._max_tabs_guess:
                    speak("That tab number seems out of range.", mood="neutral")
                    return

                # brute-force cycle tabs until we land near target
                for _ in range(abs(target - self._current_tab_index)):
                    if target > self._current_tab_index:
                        pyautogui.hotkey("ctrl", "tab")
                    else:
                        pyautogui.hotkey("ctrl", "shift", "tab")
                    time.sleep(0.05)

                self._current_tab_index = target
                speak(f"Switched to tab {target}.", mood="neutral")
                return

        except Exception as e:
            print("⚠️ Tab control error:", e)
            speak("I couldn't control the tabs.", mood="neutral")


    def _open_in_chrome(self, url: str) -> bool:
        """
        Try to open a URL specifically in Google Chrome.
        Returns True if successful, False if Chrome not found.
        """
        try:
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ]
            for path in chrome_paths:
                if os.path.exists(path):
                    subprocess.Popen([path, url])
                    return True
        except Exception as e:
            print("⚠️ _open_in_chrome error:", e)
            traceback.print_exc()
        return False

    # -------------------------------------------------------
    # OPEN / LAUNCH
    def _handle_open(self, command):
        try:
            words = (
                command.lower()
                .replace("open ", "")
                .replace("launch ", "")
                .strip()
            )

            # OPEN YOUTUBE
            if "youtube" in words:
                speak("Opening YouTube.", mood="happy")
                try:
                    if self._open_in_chrome("https://www.youtube.com"):
                        return
                except Exception:
                    pass

                webbrowser.open_new_tab("https://www.youtube.com")
                return

            # OPEN GOOGLE
            if "google" in words:
                speak("Opening Google.", mood="happy")
                webbrowser.open_new_tab("https://www.google.com")
                return

            # OPEN URL
            if "." in words:
                if not words.startswith("http"):
                    words = "https://" + words

                speak(f"Opening {words}.", mood="happy")
                webbrowser.open_new_tab(words)
                return

            # FALLBACK SEARCH
            speak(f"Searching {words}.", mood="neutral")
            webbrowser.open_new_tab(
                f"https://www.google.com/search?q={words.replace(' ', '+')}"
            )

        except Exception as e:
            print("⚠️ _handle_open error:", e)
            traceback.print_exc()
            speak("Couldn't open that.", mood="neutral")


    def _handle_skip_youtube_ad(self):
        """
        Best-effort YouTube ad skip.
        YouTube does NOT expose a reliable skip mechanism.
        This tries once safely and exits.
        """
        try:
            active = gw.getActiveWindow()
            if not active:
                speak("No active window detected.", mood="neutral")
                return

            title = (active.title or "").lower()
            if "youtube" not in title:
                speak("Skip ad works only on YouTube.", mood="neutral")
                return

            # Wait a bit — skip button appears after ~5 seconds
            time.sleep(2.5)

            # Heuristic: tab-focus + enter
            for _ in range(6):
                pyautogui.press("tab")
                time.sleep(0.2)

            pyautogui.press("enter")

            speak("If the ad was skippable, I tried skipping it.", mood="neutral")

        except Exception as e:
            print("⚠️ _handle_skip_youtube_ad error:", e)
            speak("I couldn't skip the ad.", mood="neutral")

    # -------------------------------------------------------
    # MEDIA
    def _handle_media(self, command):
        try:
            cmd = command.lower()
            if "play" in cmd or "pause" in cmd:
                keyboard.send("play/pause media")
                speak("Done.", mood="neutral")
                return
            if "volume up" in cmd:
                keyboard.send("volume up")
                speak("Volume up.", mood="neutral")
                return
            if "volume down" in cmd:
                keyboard.send("volume down")
                speak("Volume down.", mood="neutral")
                return
            if "mute" in cmd:
                keyboard.send("volume mute")
                speak("Muted.", mood="neutral")
                return
        except Exception as e:
            print("⚠️ _handle_media error:", e)
            traceback.print_exc()
            speak("Media control failed.", mood="neutral")

    # -------------------------------------------------------
    # WAKE FROM SLEEP
    def _wake_from_sleep(self):
        try:
            if getattr(state, "MODE", None) != "sleep":
                return

            try:
                jarvis_fx.play_success()
            except Exception:
                pass

            mood = memory.get_mood()
            try:
                line = brain_module.brain.generate_wakeup_line(
                    mood=mood,
                    last_topic=getattr(state, "LAST_TOPIC", None),
                )
            except Exception:
                line = "I'm awake."

            speak(line, mood=mood)

            try:
                ov = getattr(voice_effects, "overlay_instance", None)
                if ov:
                    ov.set_status("Listening…")
                    ov.set_mood(mood)
                    ov.setWindowOpacity(1.0)
            except Exception:
                pass

            state.MODE = "active"
            state.LAST_INTERACTION = time.time()

        except Exception as e:
            print("⚠️ Wake-from-sleep error:", e)
            traceback.print_exc()

    # -------------------------------------------------------
    def _get_overlay_if_available(self):
        try:
            return getattr(voice_effects, "overlay_instance", None)
        except Exception:
            return None

    # -------------------------------------------------------
    # set_speaking exposed for speech_engine to register a hook
    def set_speaking(self, speaking: bool):
        try:
            with self._speak_lock:
                self._is_speaking = bool(speaking)
                try:
                    state.SYSTEM_SPEAKING = bool(speaking)
                except Exception:
                    pass
                base = int(_DEFAULTS.get("energy_threshold", 300))
                if speaking:
                    # raise threshold while TTS runs to avoid pickup
                    try:
                        self.recognizer.energy_threshold = max(
                            base * 3,
                            getattr(self.recognizer, "energy_threshold", base),
                        )
                    except Exception:
                        pass
                else:
                    try:
                        self.recognizer.energy_threshold = base
                    except Exception:
                        pass
        except Exception:
            pass

    # -------------------------------------------------------
    # STOP
    def stop(self):
        print("🛑 Listener stopping...")
        self.running = False

        try:
            if self._bg_stop_fn:
                try:
                    self._bg_stop_fn(wait_for_stop=False)
                except TypeError:
                    try:
                        self._bg_stop_fn()
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self._consumer_thread.join(timeout=1.0)
        except Exception:
            pass

        print("🛑 Listener stopped.")


# -------------------------------------------------------
# MAIN RUNNER (for local testing)
if __name__ == "__main__":
    L = JarvisListener()
    # optionally, register hook with speech engine:
    try:
        from core.speech_engine import register_listener_hook

        register_listener_hook(L.set_speaking)
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        L.stop()
