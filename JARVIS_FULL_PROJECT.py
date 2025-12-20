# ==============================================================
#  JARVIS – FULL COMBINED PROJECT SOURCE CODE
#  Auto-generated from /Jarvis directory
# ==============================================================



##############################################################################################################
# FILE: Jarvis\README.md
##############################################################################################################



##############################################################################################################
# FILE: Jarvis\init.py
##############################################################################################################



##############################################################################################################
# FILE: Jarvis\main.py
##############################################################################################################

# main.py — ENTRY POINT ONLY (UI + backend thread)
"""
JARVIS AI PIPELINE

1) ai_chat.py
   - Talks to LLM
   - Generates RAW text (no emotion, no styling)

2) conversation_core.py
   - Decides WHAT to say
   - Handles topic, intent, fallback
   - Returns RAW text only

3) command_handler.py
   - Enhances response (emotion, cinematic style)
   - Executes actions
   - Speaks output (TTS)

IMPORTANT:
- enhance_response() must be called ONLY in command_handler
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import threading
from PyQt5 import QtWidgets

from core.interface import InterfaceOverlay
from core.runtime import jarvis_startup   # <-- uses your FaceAuth + startup logic


if __name__ == "__main__":
    print("CURRENT DIR =", os.getcwd())
    print("FILES =", os.listdir())

    app = QtWidgets.QApplication(sys.argv)
    overlay = InterfaceOverlay()

    try:
        overlay.run()
    except Exception:
        overlay.show()

    backend = threading.Thread(target=jarvis_startup, args=(overlay,), daemon=True)
    backend.start()

    sys.exit(app.exec_())


##############################################################################################################
# FILE: Jarvis\start_jarvis.py
##############################################################################################################

from core.background_listener import BackgroundListener
import time

def main():
    print("JARVIS STARTED (BACKGROUND MODE)")  # TEMP DEBUG

    listener = BackgroundListener()
    listener.start()

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()


##############################################################################################################
# FILE: Jarvis\test_background.py
##############################################################################################################

from core import background_listener
import time

listener = background_listener.BackgroundListener()
listener.start()

print("Jarvis running in background. Press Ctrl+C to stop.")

while True:
    time.sleep(10)


##############################################################################################################
# FILE: Jarvis\utils.py
##############################################################################################################



##############################################################################################################
# FILE: Jarvis\config\memory.json
##############################################################################################################

{
  "facts": {},
  "mood": "neutral",
  "last_topic": "why",
  "emotion_history": [
    {
      "mood": "neutral",
      "time": "2025-12-20 01:24:08"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 01:26:17"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 01:26:32"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 01:57:25"
    },
    {
      "mood": "happy",
      "time": "2025-12-20 01:58:06"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 01:58:50"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 14:49:15"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 14:49:42"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 14:49:57"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 14:52:21"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 14:52:58"
    },
    {
      "mood": "neutral",
      "time": "2025-12-20 14:53:09"
    }
  ]
}

##############################################################################################################
# FILE: Jarvis\config\nlp_history.txt
##############################################################################################################

open google and search apm
what are you doing
hey jarvis
hey jarvis
hey jarvis
hey jarvis
I am sad
I am low
I am hurt
I am empty
I am stress
I am overthinking
I am happy
continue
continue
continue
continue
continue
continue
What is AI
What is Java
What is life
What is love
What is blockchain
hmm
okay
random line
what do you think
open youtube
type hello
play music
search jarvis ai
increase volume
i am back
close youtube
i can't see
are you ok
i am back
low brightness
do you know about me
yes
close
close
take me to des
take me to desktop


##############################################################################################################
# FILE: Jarvis\config\settings.json
##############################################################################################################



##############################################################################################################
# FILE: Jarvis\core\__init__.py
##############################################################################################################



##############################################################################################################
# FILE: Jarvis\core\ai_chat.py
##############################################################################################################

# core/ai_chat.py — FINAL UPGRADED VERSION (Dynamic Model + Memory Fix)

"""
AI Chat Brain for Jarvis
- Automatically detects installed Ollama models.
- Falls back to conversation_core if Ollama is offline.
- Injects memory + mood + last topic correctly.
- Optimized for Voice (Short answers, No Markdown).
"""

import json
import time
import requests

# -------------------------------------------
# Optional HTTP client for Ollama raw requests
# -------------------------------------------
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

# -------------------------------------------
# Optional direct Ollama Python package
# -------------------------------------------
try:
    import ollama
    _HAS_OLLAMA_PKG = True
except ImportError:
    ollama = None
    _HAS_OLLAMA_PKG = False

# -------------------------------------------
# Local fallback convo
# -------------------------------------------
try:
    from core.conversation_core import JarvisConversation
    _HAS_CONV = True
except ImportError:
    JarvisConversation = None
    _HAS_CONV = False

# -------------------------------------------
# Memory + State
# -------------------------------------------
try:
    from core.context import memory
except ImportError:
    memory = None

import core.state as state


# ============================================================
#   OLLAMA CLIENT (Smart Model Selection)
# ============================================================
class OllamaClient:
    def __init__(self, default_model="llama3"):
        self.base_url = "http://localhost:11434"
        self.model = self._find_best_model(default_model)
        print(f"🧠 AI Brain linked to model: {self.model}")

    def _find_best_model(self, preferred):
        """Checks available models and picks the best one installed."""
        try:
            # 1. Try to fetch list of installed models
            response = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                
                # Check for preferred specific match
                for m in models:
                    if preferred in m: return m
                
                # Check for known good fallback models
                for fallback in ["llama3.1", "llama3", "mistral", "phi3", "gemma", "llama2"]:
                    for m in models:
                        if fallback in m: return m
                
                # If any model exists, return the first one
                if models: return models[0]
                
        except Exception:
            pass
        
        # Default blind return if check fails
        return preferred

    def available(self):
        """Check if Ollama server is actually running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return r.status_code == 200
        except:
            return False

    def ask(self, system_prompt, user_prompt):
        """Send request to Ollama."""
        # Method A: Python Library (Preferred)
        if _HAS_OLLAMA_PKG:
            try:
                out = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return out.get("message", {}).get("content", "").strip()
            except Exception as e:
                # print(f"Ollama Pkg Error: {e}") # Debug only
                pass

        # Method B: HTTPX / Raw Request (Backup)
        try:
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Use requests if httpx missing, or httpx if available
            if _HAS_HTTPX:
                r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=20.0)
                data = r.json()
            else:
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=20.0)
                data = r.json()

            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"⚠️ AI Connection Error: {e}")
            return None


# ============================================================
#   LOCAL FALLBACK (If Brain is Offline)
# ============================================================
class LocalFallback:
    def __init__(self):
        self.conv = JarvisConversation() if _HAS_CONV else None

    def ask(self, prompt):
        if not self.conv:
            return "Systems offline. I can't think right now."
        try:
            r = self.conv.respond(prompt)
            return r or "I didn't catch that."
        except:
            return "I couldn't process that locally."


# ============================================================
#   AI CHAT BRAIN (MAIN LOGIC)
# ============================================================
class AIChatBrain:
    def __init__(self):
        # We start with a generic name, the client will resolve the specific tag
        self.ollama = OllamaClient(default_model="llama3")
        self.fallback = LocalFallback()

    # ---------------------------------------------------------
    # Build the personality + memory prompt
    # ---------------------------------------------------------
    def _build_system_prompt(self):
        # 1. Get Mood
        mood = "neutral"
        if memory:
            mood = memory.get_mood()
        
        # 2. Get Last Topic
        last_topic = getattr(state, "LAST_TOPIC", "general chat")

        # 3. Get Memories (FIXED: accessing dictionary directly)
        mem_text = "No prior facts."
        if memory and hasattr(memory, "memory"):
            facts_dict = memory.memory.get("facts", {})
            if facts_dict:
                # Format: "- User's name is Yash"
                fact_list = [f"- {k} is {v}" for k, v in facts_dict.items()]
                mem_text = "\n".join(fact_list)

        return f"""
You are Jarvis, a highly intelligent, witty, and loyal AI assistant.
User: Yash.

YOUR STATE:
- Current Mood: {mood}
- Previous Topic: {last_topic}

MEMORY OF USER:
{mem_text}

INSTRUCTIONS:
1. RESPONSE FORMAT: You are talking via Text-to-Speech.
   - Keep answers SHORT (1-2 sentences max) unless asked for details.
   - DO NOT use Markdown (no **bold**, no # headers).
   - DO NOT use code blocks unless explicitly asked to write code.
   - Use natural, conversational English (occasional Hinglish is okay).

2. PERSONALITY:
   - If Yash is sad, be comforting and gentle.
   - If Yash is happy, be energetic.
   - If asked to do a task, say "On it" or "Sure thing".
   - Never say "As an AI language model". You are Jarvis.

3. GOAL: Be a helpful friend, not just a robot.
"""

    # ---------------------------------------------------------
    # Main Public Function: ASK
    # ---------------------------------------------------------
    def ask(self, prompt: str):
        if not prompt:
            return "I'm listening."

        # 1. Build context
        system_prompt = self._build_system_prompt()

        # 2. Try Ollama (The Brain)
        if self.ollama.available():
            ans = self.ollama.ask(system_prompt, prompt)
            if ans:
                # Update global topic state if successful
                try:
                    state.LAST_TOPIC = prompt
                except: pass
                return ans.strip() if ans else None


        # 3. Failover to Local (Scripted)
        print("⚠️ Ollama offline/unreachable. Using local fallback.")
        return self.fallback.ask(prompt)


# ============================================================
# Export singleton
# ============================================================
ai_chat_brain = AIChatBrain()

##############################################################################################################
# FILE: Jarvis\core\ai_client.py
##############################################################################################################

# core/ai_client.py
"""
AI client adapter for Jarvis — compatibility + robust fallbacks.

Behavior:
- Prefer core.ai_chat.ai_chat_brain (the new Ollama-based module).
- Else try a lightweight Ollama HTTP wrapper (if httpx is installed).
- Else fallback to core.conversation_core.JarvisConversation.
- Provides a small stable API:
    ai_client.available() -> bool
    ai_client.ask(prompt: str, timeout: float|None = None) -> str|None

This file should be drop-in compatible with other modules that call
`ai_chat_brain.ask(...)` or expect an `ai_client` style object.
"""

import time
import threading
import traceback

# Try to use the new consolidated ai_chat if present
try:
    from core.ai_chat import ai_chat_brain as _new_ai_brain  # final stable ai_chat (ollama wrapper)
    _HAS_NEW_AICHAT = True
except Exception:
    _new_ai_brain = None
    _HAS_NEW_AICHAT = False

# Try HTTP client for raw Ollama if needed
try:
    import httpx
    _HAS_HTTPX = True
except Exception:
    httpx = None
    _HAS_HTTPX = False

# Try local fallback conversation
try:
    from core.conversation_core import JarvisConversation
    _HAS_CONV = True
except Exception:
    JarvisConversation = None
    _HAS_CONV = False

# Optional memory (not required but used if available)
try:
   from core.context import memory as _MEMORY

except Exception:
    _MEMORY = None

# state (for optional context update)
try:
    import core.state as state
except Exception:
    state = None

# Default Ollama HTTP config (used only if ai_chat not available and httpx present)
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_TIMEOUT = 20.0


class _HTTPollama:
    """Small, resilient Ollama HTTP client used as a fallback."""
    def __init__(self, host=_DEFAULT_OLLAMA_HOST, model=_DEFAULT_OLLAMA_MODEL, timeout=_DEFAULT_TIMEOUT):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = float(timeout)

    def available(self) -> bool:
        if not _HAS_HTTPX:
            return False
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def ask(self, user_prompt: str, system_prompt: str = "", timeout: float | None = None) -> str | None:
        if not _HAS_HTTPX:
            return None
        try:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""}
                ]
            }
            to = timeout or self.timeout
            with httpx.Client(timeout=to) as client:
                resp = client.post(f"{self.host}/api/chat", json=body)
            if resp.status_code != 200:
                return None
            data = resp.json()
            # expected format: {"message":{"role":"assistant","content":"..."}}
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    return data["message"].get("content", "").strip() or None
                if "content" in data:
                    return str(data.get("content", "")).strip() or None
            return None
        except Exception:
            # don't crash; return None on failure
            return None


class LocalConvWrapper:
    """Wrap JarvisConversation to provide ask() semantics."""
    def __init__(self):
        self.conv = JarvisConversation() if _HAS_CONV else None

    def available(self):
        return self.conv is not None

    def ask(self, prompt: str, timeout: float | None = None) -> str:
        if not self.conv:
            return f"I heard: {prompt or ''}"
        try:
            return self.conv.respond(prompt or "") or "I couldn't form a reply."
        except Exception:
            return "Local conversation engine failed."


class AIClient:
    """
    Unified AI client adapter used by other modules.

    Usage:
        from core.ai_client import ai_client
        if ai_client.available():
            reply = ai_client.ask("hello jarvis")
    """

    def __init__(self):
        # priority: new ai_chat (preferred), HTTP Ollama, local conv
        self._source = None
        self._http = _HTTPollama()
        self._local = LocalConvWrapper()
        self._lock = threading.Lock()

        # prefer new ai_chat if present
        if _HAS_NEW_AICHAT and _new_ai_brain is not None:
            self._source = ("ai_chat", _new_ai_brain)
        elif self._http.available():
            self._source = ("http_ollama", self._http)
        elif self._local.available():
            self._source = ("local_conv", self._local)
        else:
            self._source = ("none", None)

    def refresh_source(self):
        """Try to detect a better source (non-blocking)."""
        try:
            if _HAS_NEW_AICHAT and _new_ai_brain is not None:
                self._source = ("ai_chat", _new_ai_brain)
                return
            if self._http.available():
                self._source = ("http_ollama", self._http)
                return
            if self._local.available():
                self._source = ("local_conv", self._local)
                return
            self._source = ("none", None)
        except Exception:
            self._source = ("none", None)

    def available(self) -> bool:
        """Is there any usable backend?"""
        s, impl = self._source
        if s == "ai_chat":
            return impl is not None
        if s == "http_ollama":
            return impl is not None and impl.available()
        if s == "local_conv":
            return impl is not None and impl.available()
        # try to refresh quickly
        self.refresh_source()
        s, impl = self._source
        return s in ("ai_chat", "http_ollama", "local_conv") and impl is not None

    def ask(self, prompt: str, timeout: float | None = None) -> str:
        """Ask the best available backend for a reply. Returns a string (never raises)."""
        if not prompt:
            return ""

        # quick opportunistic refresh
        try:
            if self._source[0] == "none":
                self.refresh_source()
        except Exception:
            pass

        s, impl = self._source

        # 1) Preferred new ai_chat module (core.ai_chat.ai_chat_brain)
        if s == "ai_chat" and impl is not None:
            try:
                # its API is ask(prompt) returning a string
                return impl.ask(prompt) or "I couldn't get a response."
            except Exception:
                traceback.print_exc()
                # try to fall through to other backends

        # 2) HTTP Ollama fallback
        if (s == "http_ollama" or (impl is None and self._http.available())) and _HAS_HTTPX:
            try:
                resp = self._http.ask(prompt, system_prompt="", timeout=timeout)
                if resp:
                    # update last topic if state available
                    try:
                        if state is not None:
                            state.LAST_TOPIC = prompt
                    except:
                        pass
                    return resp
            except Exception:
                traceback.print_exc()

        # 3) Local conversation fallback
        try:
            if self._local and self._local.available():
                return self._local.ask(prompt)
        except Exception:
            traceback.print_exc()

        # Final safe fallback: echo politely
        return "Sorry Yash, I'm not connected to a model right now."

    # Async helper (fire-and-forget thread, stores latest reply callback)
    def ask_async(self, prompt: str, callback=None, timeout: float | None = None):
        """
        Ask in a background thread. callback(reply_str) will be called with the reply.
        """
        def _worker(q, cb, to):
            try:
                r = self.ask(q, timeout=to)
            except Exception:
                r = "I couldn't get an answer right now."
            if cb:
                try:
                    cb(r)
                except Exception:
                    pass

        t = threading.Thread(target=_worker, args=(prompt, callback, timeout), daemon=True)
        t.start()
        return t


# Export singleton
ai_client = AIClient()

# Backwards-compat convenience: old code sometimes expected `ai_chat_brain` name
# We attempt to expose a minimal compatible object.
class _CompatWrapper:
    def __init__(self, client):
        self._client = client

    def available(self):
        return self._client.available()

    def ask(self, prompt):
        return self._client.ask(prompt)

# Expose `ai_chat_brain` if not already used by new ai_chat
ai_chat_brain = _new_ai_brain if _HAS_NEW_AICHAT and _new_ai_brain is not None else _CompatWrapper(ai_client)


##############################################################################################################
# FILE: Jarvis\core\background_listener.py
##############################################################################################################

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


##############################################################################################################
# FILE: Jarvis\core\brain.py
##############################################################################################################

# core/brain.py — IIT-Level Enhanced Friend Brain (UPGRADED)

import html
from typing import Optional

# Core systems
from core.context import memory
from core.emotion_reflection import JarvisEmotionReflection
import core.nlp_engine as nlp
import core.state as state

# Optional LLM (local/Ollama/OpenAI)
try:
    import openai
except:
    openai = None

reflection = JarvisEmotionReflection()

class Brain:
    def __init__(self):
        self.personality = "friend_balanced"
        self._markov_chance = 0.15
        self._friend_tease_chance = 0.15
        self._max_len = 450
        print("🧠 Brain online — mode:", self.personality)

    # -----------------------------------------
    # EMOTION DETECTION (text-based)
    # -----------------------------------------
    def detect_text_emotion(self, text: Optional[str]) -> str:
        if not text:
            return "neutral"

        t = text.lower()

        groups = {
            "serious": ["sad", "hurt", "empty", "broken", "lonely", "upset", "low"],
            "alert": ["angry", "pissed", "furious", "scared", "fear", "panic", "worried", "stress"],
            "happy": ["happy", "great", "awesome", "nice", "amazing"],
            "neutral": ["bored", "meh", "okay", "fine"]
        }

        for mood, words in groups.items():
            if any(w in t for w in words):
                return mood

        return "neutral"

    # -----------------------------------------
    # EMOTION FUSION
    # -----------------------------------------
    def fuse_emotions(self, face=None, text=None, tone=None):
        votes = []

        # face → strongest
        face_map = {
            "happy": "happy",
            "surprise": "happy",
            "neutral": "neutral",
            "sad": "serious",
            "angry": "alert",
            "fear": "alert",
            "disgust": "serious"
        }
        if face:
            votes.append(face_map.get(face.lower(), "neutral"))

        # text
        if text:
            votes.append(self.detect_text_emotion(text))

        # tone
        if tone:
            votes.append(tone)

        # last memory
        last_mood = None
        try:
            hist = memory.memory.get("emotion_history", [])
            if hist:
                last_mood = hist[-1]["mood"]
                votes.append(last_mood)
        except:
            pass

        if not votes:
            return memory.get_mood() or "neutral"

        weight = {"happy": 3, "serious": 2, "alert": 2, "neutral": 1}
        tally = {}
        for v in votes:
            tally[v] = tally.get(v, 0) + weight.get(v, 1)

        final = max(tally, key=tally.get)

        try:
            memory.set_mood(final)
            reflection.add_emotion(final)
        except:
            pass

        return final
    # -------------------------------------------------------
    # CINEMATIC + FRIEND MODE WAKE-UP LINE
    # -------------------------------------------------------
    def generate_wakeup_line(self, mood=None, last_topic=None):
        mood = (mood or memory.get_mood() or "neutral").lower()

        lines = {
            "happy": [
                "Aye Yashu, I’m online and vibing.",
                "Energy high — what’s our plan, Yash?",
                "Back online — and kinda excited."
            ],
            "serious": [
                "Ready. Focused. Tell me the task.",
                "I’m online — let’s get this done properly.",
                "Fully active. Just say the word."
            ],
            "alert": [
                "I’m here — fully attentive.",
                "Alert mode off, listening now.",
                "You called — I’m focused."
            ],
            "neutral": [
                "I’m awake, Yash. What’s next?",
                "Alright — talk to me.",
                "Online. Listening."
            ]
        }

        base = random.choice(lines.get(mood, lines["neutral"]))

        # Friend tease (rare)
        tease = ""
        if random.random() < self._friend_tease_chance and mood != "serious":
            tease = random.choice([
                "Hope it’s something interesting.",
                "If it’s snacks — I’m all ears.",
                "You woke me up like a pro."
            ])

        # Markov flavor
        extra = ""
        if random.random() < self._markov_chance:
            try:
                extra = nlp._markov_generate() or ""
            except:
                extra = ""

        # Topic continuation
        topic_add = ""
        if last_topic and random.random() < 0.30:
            topic_add = f"We were talking about {last_topic}."

        parts = [base, tease, extra, topic_add]
        return " ".join([p for p in parts if p]).strip()

    # -------------------------------------------------------
    # EMOTIONAL SUPPORT ENGINE (FRIEND MODE)
    # -------------------------------------------------------
    def generate_emotional_support(self, user_feeling, mood=None):
        if not user_feeling:
            return "I’m here, Yash. Tell me what’s going on."

        t = user_feeling.lower()

        templates = {
            "sad": [
                "Come here, Yashu… talk to me. I'm right here.",
                "It’s okay to feel sad — I’m listening."
            ],
            "hurt": [
                "You don’t have to handle that pain alone.",
                "Tell me what hurt you — I’m here."
            ],
            "angry": [
                "Slow breath. Tell me what triggered you.",
                "I’m with you, Yash. Take it one line at a time."
            ],
            "stress": [
                "Let’s breathe in… and out. I’m here with you.",
                "One step at a time — tell me the heaviest part."
            ],
            "empty": [
                "Feeling empty is exhausting… sit with me.",
                "We’ll figure this out together, Yash."
            ],
            "happy": [
                "That’s the energy I love! Keep shining.",
                "You sound bright — I like that!"
            ]
        }

        for key, msgs in templates.items():
            if key in t:
                msg = random.choice(msgs)
                if random.random() < 0.25:
                    msg += " Want me to help with something specific?"
                return msg

        # Generic fallback
        fallback = random.choice([
            "I’m with you, Yash. Tell me a little more.",
            "Whatever it is — you’re not alone.",
        ])
        if random.random() < 0.25:
            fallback += " Should I give a suggestion?"
        return fallback

    # -------------------------------------------------------
    # TOPIC CONTINUATION GENERATOR
    # -------------------------------------------------------
    def generate_continuation(self, topic):
        if not topic:
            topic = "that topic"

        variants = [
            f"Want to go deeper into {topic}?",
            f"{topic} has more layers — want me to break them down?",
            f"We can push {topic} further — detailed or simple?"
        ]

        extra = ""
        if random.random() < self._markov_chance:
            try:
                extra = nlp._markov_generate()
            except:
                extra = ""

        return f"{random.choice(variants)} {extra}".strip()

    # -------------------------------------------------------
    # SHORT KNOWLEDGE ANSWERS (FRIEND MODE)
    # -------------------------------------------------------
    def answer_question(self, query, topic, mood=None):
        topic = (topic or "").lower()

        base_info = {
            "ai": "AI is basically pattern learning — machines understanding data like a human brain does.",
            "ml": "Machine learning allows systems to improve using past data — without explicit programming.",
            "java": "Java runs on the JVM, making it portable, secure, and fast.",
            "python": "Python is simple, powerful, and perfect for AI.",
            "life": "Life isn't solved — it's understood slowly.",
            "love": "Love is timing, effort, and understanding.",
            "daa": "DAA helps measure algorithm efficiency and complexity."
        }

        if topic in base_info:
            resp = base_info[topic]
        else:
            resp = f"{topic} is interesting — want a simple explanation or detailed?"

        # Markov flavor
        extra = ""
        if random.random() < 0.20:
            try:
                extra = nlp._markov_generate()
            except:
                extra = ""

        # Friendly signoff
        tail = ""
        if random.random() < 0.20:
            tail = random.choice([
                "Need an example?",
                "Want a summary?",
                "Shall I go deeper?"
            ])

        out = f"{resp} {extra} {tail}".strip()
        return out[:400]
    # -------------------------------------------------------
    # FALLBACK REPLY (When no command matched)
    # -------------------------------------------------------
    def fallback_reply(self, original_text=None):
        seeds = [
            "I might need a bit more clarity — say it in another way?",
            "Hmm… didn’t fully get that. Want me to search it?",
            "Try rephrasing that for me, Yashu."
        ]

        adds = [
            "I can open apps, search, or explain things — what do you need?",
            "Want me to check online?",
            "Should I give a short summary or a deep explanation?"
        ]

        base = random.choice(seeds)

        if random.random() < 0.35:
            base += " " + random.choice(adds)

        if random.random() < 0.25:
            base += " (I’m right here.)"

        return base

    # -------------------------------------------------------
    # FINAL RESPONSE ENHANCEMENT (Adds prefix/suffix/mood polish)
    # -------------------------------------------------------
    def enhance_response(self, text, mood=None, last_topic=None):
        if not text:
            return ""

        mood = (mood or memory.get_mood() or "neutral").lower()
        clean = text.strip()

        # Mood-based flavor
        prefix = ""
        suffix = ""

        if mood == "happy":
            prefix = random.choice(["Nice!", "Sweet!", "Love it!"]) if random.random() < 0.40 else ""
            suffix = random.choice(["That felt smooth.", "Good call."]) if random.random() < 0.25 else ""
        elif mood == "serious":
            prefix = random.choice(["Understood.", "Affirmative."]) if random.random() < 0.45 else ""
            suffix = random.choice(["Proceeding.", "Handled."]) if random.random() < 0.20 else ""
        elif mood == "alert":
            prefix = random.choice(["On it.", "Right away."]) if random.random() < 0.55 else ""
            suffix = random.choice(["Be careful.", "Done swiftly."]) if random.random() < 0.22 else ""
        else:  # neutral
            prefix = random.choice(["Okay.", "Alright."]) if random.random() < 0.22 else ""

        # Friendly “nudge”
        nudge = ""
        if random.random() < 0.20:
            nudge = random.choice([
                " Need anything else?",
                " Want me to continue?",
                " Shall I look up more?"
            ])

        # Flavor from markov chain
        markov = ""
        try:
            if random.random() < self._markov_chance:
                markov = nlp._markov_generate() or ""
        except:
            markov = ""

        # Combine everything
        parts = [prefix, clean, suffix, markov, nudge]
        out = " ".join([p for p in parts if p]).strip()

        # Slight topic reminder
        if last_topic and random.random() < 0.12:
            out += f" (about {last_topic})"

        # Safety trim
        if len(out) > 400:
            out = out[:370].rsplit(" ", 1)[0] + "..."

        return out

    # -------------------------------------------------------
    # LLM RESPONSE POST-PROCESSOR
    # (Wraps Ollama/ChatGPT responses with mood + personality)
    # -------------------------------------------------------
    def postprocess_reply(self, llm_reply, mood=None, last_topic=None):
        if not llm_reply:
            return self.fallback_reply()

        reply = llm_reply.strip()

        # Shorten long LLM outputs
        if len(reply) > 250:
            idx = reply.find(".", 180)
            if idx != -1:
                reply = reply[:idx+1]
            else:
                reply = reply[:300].rsplit(" ", 1)[0] + "..."

        # Apply personality polishing
        try:
            pretty = self.enhance_response(reply, mood=mood, last_topic=last_topic)
        except:
            pretty = reply

        # Store topic memory
        try:
            if last_topic:
                memory.update_topic(last_topic)
            else:
                # auto-extract short topic
                tokens = reply.split()
                if len(tokens) > 2:
                    memory.update_topic(" ".join(tokens[:3]))
        except:
            pass

        # sync global topic
        try:
            state.LAST_TOPIC = last_topic or state.LAST_TOPIC
        except:
            pass

        return pretty


# -------------------------------------------------------
# SINGLETON INSTANCE
# -------------------------------------------------------
brain = Brain()
print("🧠 Brain fully initialized with FRIEND + CINEMATIC mode.")


##############################################################################################################
# FILE: Jarvis\core\command_handler.py
##############################################################################################################

import os
import webbrowser
import psutil
import pyautogui
import subprocess
import time
import traceback
import threading
import functools
import random
import datetime

from core.whatsapp_selenium import send_whatsapp_message
from core.intent_parser import parse_intent
AUTOMATION_LOCK = threading.Lock()
WHATSAPP_CONFIRM = None


THINKING_DELAY = 1.2      # seconds before saying "thinking"
THINKING_COOLDOWN = 4.0   # minimum gap between thinking messages
_last_thinking_time = 0
PENDING_INTENT = None
# ---------------- YOUTUBE STATE ----------------
YOUTUBE_ACTIVE = False

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

# Desktop control - instantiate safely
try:
    from core.desktop_control import DesktopControl
    desktop = DesktopControl()
except Exception:
    desktop = None

from core.speech_engine import speak, jarvis_fx
from core.conversation_core import JarvisConversation
from core.context import memory
from core.emotion_reflection import JarvisEmotionReflection

reflection = JarvisEmotionReflection()

# NEW: Phase-2 skill modules
try:
    from core.document_reader import document_reader
except Exception:
    document_reader = None

try:
    from core.video_reader import video_reader
except Exception:
    video_reader = None

try:
    from core.music_player import music_player
except Exception:
    music_player = None

try:
    from core.music_stream import music_stream
except Exception:
    music_stream = None

# NEW IMPORTS (Brain + State + AI)
import core.brain as brain_module
import core.state as state

# Attempt to import AI chat backend (ollama wrapper or other). If unavailable we'll fallback.
try:
    from core.ai_chat import ai_chat_brain
    AI_CHAT_AVAILABLE = True
except Exception:
    ai_chat_brain = None
    AI_CHAT_AVAILABLE = False

def log_command(cmd):
    with open("jarvis.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} | {cmd}\n")


class JarvisCommandHandler:
    """JARVIS Brain — handles commands, responses, emotions & memory."""

    def __init__(self, ai_think_message=None):
        print("🧠 Jarvis Command Handler Ready")
        self.user = "Yash"
        self.conversation = JarvisConversation()
        self._ai_lock = threading.Lock()
        self.ai_think_message = ai_think_message

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def process(self, command):
        global PENDING_INTENT , YOUTUBE_ACTIVE

        if not command:
            return

        log_command(command)
        raw_command = command
        command = command.lower().strip()

        from core.context import get_last_action, set_last_action

        print(f"🎤 Processing Command: {command}")
        
        # ==================================================
        # WHATSAPP MESSAGE — CLEAN MULTI TURN FLOW (SELENIUM)
        # ==================================================
        if PENDING_INTENT and PENDING_INTENT.get("intent") == "whatsapp_message":

            # STEP 2 — capture contact name
            if PENDING_INTENT["contact"] is None:
                PENDING_INTENT["contact"] = raw_command.strip()
                speak(
                    f"What should I send to {PENDING_INTENT['contact']}?",
                    mood="neutral"
                )
                return

            # STEP 3 — capture message
            if PENDING_INTENT["message"] is None:
                PENDING_INTENT["message"] = raw_command.strip()
                PENDING_INTENT["confirm"] = True

                speak(
                    f'I am about to send "{PENDING_INTENT["message"]}" '
                    f'to {PENDING_INTENT["contact"]}. Should I send it?',
                    mood="neutral"
                )
                return

            # STEP 4 — confirmation
            if PENDING_INTENT.get("confirm"):
                if any(x in command for x in ["yes", "send", "confirm", "do it"]):
                    contact = PENDING_INTENT["contact"]
                    message = PENDING_INTENT["message"]

                    speak(f"Sending message to {contact}.", mood="happy")

                    send_whatsapp_message(contact, message)

                else:
                    speak("Message cancelled.", mood="neutral")

                PENDING_INTENT = None
                return

        # ==================================================
        # START WHATSAPP MESSAGE MODE (ONLY ENTRY POINT)
        # ==================================================
        if (
            PENDING_INTENT is None
            and "whatsapp" in command
            and any(word in command for word in ["send", "message", "msg"])
        ):
            PENDING_INTENT = {
                "intent": "whatsapp_message",
                "contact": None,
                "message": None
            }
            speak("Whom should I message?", mood="neutral")
            return

        # --------------------------------------------------------------
        # YOUTUBE: PLAY Nth VIDEO AFTER SEARCH
        # --------------------------------------------------------------
        if (
            state.LAST_APP_CONTEXT == "youtube"
            and state.LAST_YOUTUBE_SEARCH
            and "play" in command
            and "video" in command
        ):

            number_map = {
                "first": 1, "1st": 1, "one": 1,
                "second": 2, "2nd": 2, "two": 2,
                "third": 3, "3rd": 3, "three": 3,
                "fourth": 4, "4th": 4,
                "fifth": 5, "5th": 5
            }

            index = 1
            for word, num in number_map.items():
                if word in command:
                    index = num
                    break

            speak(f"Playing video number {index}.", mood="happy")

            time.sleep(3)
            pyautogui.click(500, 400)   # force browser focus
            time.sleep(0.2)
            pyautogui.hotkey("home")
            time.sleep(0.3)

            pyautogui.press(
                "tab",
                presses=6 + index * 2,
                interval=0.15
            )
            pyautogui.press("enter")

            state.LAST_YOUTUBE_SEARCH = False
            return

        # --------------------------------------------------------------
        # INTENT THINKING GATE (NO ACTION, NO AUTOMATION)
        # --------------------------------------------------------------
        intent = parse_intent(raw_command)

        intent_name = intent.get("intent")
        params = intent.get("params", {})
        confidence = intent.get("confidence", 0.0)

        print("🧠 INTENT PARSED:", intent)

        # --------------------------------------------------------------
        # INCOMPLETE INTENT → ASK FOLLOW-UP (HUMAN BEHAVIOR)
        # --------------------------------------------------------------
        if intent_name == "open_app" and not params.get("app"):
            PENDING_INTENT = intent
            speak("What should I open?")
            return

        if intent_name == "search" and not params.get("query"):
            speak("What should I search for?")
            PENDING_INTENT = intent
            return

        if intent_name == "type_text" and not params.get("text"):
            PENDING_INTENT = intent
            speak("What should I type?")
            return

        # helper: enhanced speak
        def speak_enhanced(text, mood=None):
            try:
                out = brain_module.brain.enhance_response(
                    text,
                    mood=mood,
                    last_topic=memory.get_last_topic()
                )
            except:
                out = text
            try:
                speak(out, mood=mood)
            except:
                speak(text)

        # --------------------------------------------------------------
        # QUICK IMMEDIATE ACTIONS
        # --------------------------------------------------------------

        # BRIGHTNESS CONTROL
        if any(x in command for x in ["increase brightness", "brightness up", "bright up"]):
            try:
                if desktop: desktop.increase_brightness()
                speak_enhanced("Increasing brightness, Yash.", mood="happy")
            except:
                speak("Couldn't change brightness right now.", mood="alert")
            return

        if any(x in command for x in ["decrease brightness", "brightness down", "dim"]):
            try:
                if desktop: desktop.decrease_brightness()
                speak_enhanced("Okay Yash, dimming the screen.", mood="serious")
            except:
                speak("Couldn't change brightness right now.", mood="alert")
            return

        # VOLUME
        if any(x in command for x in ["volume up", "increase volume", "sound up"]):
            try:
                if desktop: desktop.volume_up()
                speak_enhanced("Raising the volume.", mood="happy")
            except:
                speak("Couldn't change volume.", mood="alert")
            return

        if any(x in command for x in ["volume down", "sound down", "low volume"]):
            try:
                if desktop: desktop.volume_down()
                speak_enhanced("Lowering the volume.", mood="neutral")
            except:
                speak("Couldn't change volume.", mood="alert")
            return

        # MUTE
        if "mute" in command and "unmute" not in command:
            try:
                if desktop: desktop.mute()
                speak_enhanced("Muted.", mood="neutral")
            except:
                speak("Failed to mute.", mood="alert")
            return

        if "unmute" in command:
            try:
                if desktop: desktop.unmute()
                speak_enhanced("Unmuted.", mood="happy")
            except:
                speak("Failed to unmute.", mood="alert")
            return

        # --------------------------------------------------------------
        # DOCUMENT / VIDEO MODULES
        # --------------------------------------------------------------
        # Document Reading
        try:
            if document_reader and (
                "read" in command or
                ("summarize" in command and any(ext in command for ext in [".pdf", ".docx", ".txt", ".md"]))
            ):
                tokens = command.split()
                path_candidate = None
                for tok in tokens:
                    if any(tok.endswith(ext) for ext in [".pdf", ".doc", ".docx", ".txt", ".md"]):
                        path_candidate = tok
                        break

                if path_candidate:
                    path = os.path.abspath(path_candidate)
                    if os.path.exists(path):
                        if "summarize" in command:
                            speak("Summarizing the document…", mood="neutral")
                            threading.Thread(target=document_reader.read, args=(path, True), daemon=True).start()
                            return
                        else:
                            speak("Reading the document…", mood="neutral")
                            threading.Thread(target=document_reader.read, args=(path, False), daemon=True).start()
                            return

                # fallback: pick latest doc
                docs = [
                    f for f in os.listdir('.')
                    if any(f.lower().endswith(ext) for ext in [".pdf", ".docx", ".txt", ".md"])
                ]

                if docs:
                    chosen = os.path.abspath(docs[-1])
                    speak(
                        f"Reading latest document: {os.path.basename(chosen)}",
                        mood="neutral"
                    )
                    self._ai_pipeline_worker(raw_command)
                    return

                # 👇 ask user if nothing detected
                speak("Please tell me the document file name or full path.", mood="neutral")
                return
  
        except:
            pass

        # Video Summarization
        try:
            if video_reader and ("summarize video" in command or "summarize" in command):
                tokens = command.split()
                path_candidate = None
                for tok in tokens:
                    if any(tok.endswith(ext) for ext in [".mp4", ".mkv", ".mov"]):
                        path_candidate = tok
                        break
                if path_candidate:
                    path = os.path.abspath(path_candidate)
                    if os.path.exists(path):
                        speak("Summarizing the video…", mood="neutral")
                        threading.Thread(target=video_reader.summarize, args=(path,), daemon=True).start()
                        return

                vids = [
                    f for f in os.listdir('.')
                    if any(f.lower().endswith(ext) for ext in [".mp4", ".mkv", ".mov"])
                ]

                if vids:
                    chosen = os.path.abspath(vids[-1])
                    speak(
                        f"Summarizing latest video: {os.path.basename(chosen)}",
                        mood="neutral"
                    )
                    threading.Thread(
                        target=video_reader.summarize,
                        args=(chosen,),
                        daemon=True
                    ).start()
                    return

                # 👇 ask user if nothing detected
                speak("Please tell me the video file name or full path.", mood="neutral")
                return

        except:
            pass

        # --------------------------------------------------------------
        # LOCAL MUSIC / STREAMING
        # --------------------------------------------------------------
        try:
            # local file
            if music_player and ("play " in command and any(ext in command for ext in [".mp3", ".wav", ".ogg"])):
                tokens = command.split()
                for tok in tokens:
                    if any(tok.endswith(ext) for ext in [".mp3", ".wav", ".ogg"]):
                        path = os.path.abspath(tok)
                        if os.path.exists(path):
                            threading.Thread(target=music_player.play, args=(path,), daemon=True).start()
                            return

            # stream query (SONG ONLY — do not hijack video)
            if (
                music_stream
                and "play song" in command
                and state.LAST_APP_CONTEXT != "youtube"
            ):
                q = command.replace("play song", "", 1).strip()
                if q:
                    threading.Thread(
                        target=music_stream.play,
                        args=(q,),
                        daemon=True
                    ).start()
                    return

        except:
            pass
        # --------------------------------------------------------------
        # MUSIC CONTROLS
        # --------------------------------------------------------------
        try:
            if music_player and any(k in command for k in ["pause music", "pause song", "pause"]):
                music_player.pause()
                return

            if music_player and any(k in command for k in ["resume music", "resume song", "resume"]):
                music_player.resume()
                return

            if music_player and any(k in command for k in ["stop music", "stop song", "stop playback"]):
                music_player.stop()
                return

            if music_player and any(k in command for k in ["next song", "next track", "next"]):
                music_player.next()
                return

            if music_player and any(k in command for k in ["previous song", "prev song", "previous", "prev"]):
                music_player.previous()
                return

            # set volume to %
            if music_player and "set volume to" in command:
                try:
                    pct = int(''.join(c for c in command.split("set volume to", 1)[1] if c.isdigit()))
                    v = max(0, min(100, pct)) / 100.0
                    music_player.set_volume(v)
                    return
                except:
                    pass
        except:
            pass

        # --------------------------------------------------------------
        # DESKTOP WINDOWS / SYSTEM CONTROLS
        # --------------------------------------------------------------
        if any(x in command for x in ["show desktop", "minimize all"]):
            try:
                if desktop: desktop.show_desktop()
                speak_enhanced("Taking you to the desktop.", mood="neutral")
            except:
                speak("Couldn't switch to desktop.", mood="alert")
            return

        if "close window" in command:
            try:
                if desktop: desktop.close_window()
                speak_enhanced("Window closed.", mood="neutral")
            except:
                speak("Couldn't close window.", mood="alert")
            return

        if "maximize window" in command:
            try:
                if desktop: desktop.maximize_window()
                speak_enhanced("Maximized.", mood="neutral")
            except:
                speak("Couldn't maximize window.", mood="alert")
            return

        if "minimize window" in command:
            try:
                if desktop: desktop.minimize_window()
                speak_enhanced("Minimized.", mood="neutral")
            except:
                speak("Couldn't minimize window.", mood="alert")
            return

        if any(x in command for x in ["next window", "switch window", "alt tab"]):
            try:
                if desktop: desktop.next_window()
                speak_enhanced("Switching window.", mood="neutral")
            except:
                speak("Couldn't switch window.", mood="alert")
            return

        if any(x in command for x in ["previous window", "alt tab back"]):
            try:
                if desktop: desktop.previous_window()
                speak_enhanced("Going back to previous window.", mood="neutral")
            except:
                speak("Couldn't switch back.", mood="alert")
            return

        # --------------------------------------------------------------
        # SYSTEM COMMANDS
        # --------------------------------------------------------------
        if any(x in command for x in ["lock screen", "lock pc"]):
            try:
                if desktop: desktop.lock_screen()
                speak("Locked. I’ll be waiting, Yash.", mood="neutral")
            except:
                speak("Couldn't lock the screen.", mood="alert")
            return

        if any(x in command for x in ["restart", "reboot"]):
            try:
                speak("Restarting the system… be right back.", mood="neutral")
                if desktop: desktop.restart_system()
                else: os.system("shutdown /r /t 1")
            except:
                speak("Restart failed.", mood="alert")
            return

        if any(x in command for x in ["care mode", "take care of me"]):
            try:
                if desktop:
                    desktop.decrease_brightness()
                    desktop.volume_down()
                speak("Of course Yashu… softer lights, calmer sound. I'm here.", mood="serious")
            except:
                speak("Couldn't switch to care mode.", mood="alert")
            return

        # --------------------------------------------------------------
        # GREETINGS
        # --------------------------------------------------------------
        if any(x in command for x in ["hello", "hi", "hey"]):
            speak(random.choice([
                f"Hello {self.user}, ready when you are.",
                f"Hey {self.user}, I’m here.",
                f"Hi {self.user}, systems active."
            ]), mood="happy")
            return

        # --------------------------------------------------------------
        # TIME / DATE
        # --------------------------------------------------------------
        if command == "time" or "what's the time" in command or "time kya" in command:
            now = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"It’s {now}, {self.user}.")
            return

        if "date" in command:
            today = datetime.date.today().strftime("%A, %B %d, %Y")
            speak(f"Today is {today}.")
            return

        # --------------------------------------------------------------
        # BATTERY
        # --------------------------------------------------------------
        if "battery" in command:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    speak(
                        f"Battery is at {battery.percent}% "
                        f"and {'charging' if battery.power_plugged else 'not charging'}.",
                        mood="neutral"
                    )
                else:
                    speak("I can't read battery info right now.")
            except:
                speak("Battery check failed.", mood="alert")
            return

        # --------------------------------------------------------------
        # OPEN WEBSITES
        # --------------------------------------------------------------
        if "open youtube" in command:
            webbrowser.open("https://www.youtube.com")
            speak("YouTube opened.", mood="happy")

            state.LAST_APP_CONTEXT = "youtube"
            state.LAST_YOUTUBE_SEARCH = False
            YOUTUBE_ACTIVE = True

            time.sleep(2)
            pyautogui.press("tab")
            return


 
        if "search youtube for" in command:
            query = command.replace("search youtube for", "").strip()

            webbrowser.open(
                f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            )

            state.LAST_APP_CONTEXT = "youtube"
            state.LAST_YOUTUBE_SEARCH = True
            YOUTUBE_ACTIVE = True

            set_last_action("youtube_search")


            speak(f"Searching YouTube for {query}", mood="happy")
            return


        if "open google" in command:
            speak("Opening Google.", mood="happy")
            webbrowser.open("https://www.google.com")
            return

        if "open spotify" in command:
            speak("Opening Spotify.", mood="happy")
            webbrowser.open("https://open.spotify.com")
            return

        if "open camera" in command:
            speak("Opening camera.", mood="happy")
            os.system("start microsoft.windows.camera:")
            return

        # --------------------------------------------------------------
        # SCREENSHOT
        # --------------------------------------------------------------
        if "screenshot" in command:
            try:
                filename = f"screenshot_{datetime.datetime.now().strftime('%H%M%S')}.png"
                pyautogui.screenshot(filename)
                speak(f"Screenshot saved as {filename}.")
            except:
                speak("Screenshot failed.", mood="alert")
            return

        # --------------------------------------------------------------
        # APPS
        # --------------------------------------------------------------
        if "notepad" in command:
            speak("Opening Notepad.", mood="happy")
            subprocess.Popen(["notepad.exe"])
            return

        # --------------------------------------------------------------
        # WHATSAPP OPEN / CLOSE (PRIORITY)
        # --------------------------------------------------------------
        if "whatsapp" in command and any(x in command for x in ["close", "exit", "quit", "band"]):
            try:
                pyautogui.hotkey("ctrl", "w")
                speak("Closing WhatsApp.", mood="neutral")
            except:
                speak("Couldn't close WhatsApp.", mood="alert")
            return

        if "open whatsapp" in command:
            speak("Opening WhatsApp.", mood="happy")
            webbrowser.open("https://web.whatsapp.com")
            return

        # --------------------------------------------------------------
        # BROWSER TAB CONTROLS
        # --------------------------------------------------------------
       
        # --------------------------------------------------------------
        # YOUTUBE: NEXT / PREVIOUS VIDEO
        # --------------------------------------------------------------
        if state.LAST_APP_CONTEXT == "youtube" and "play next video" in command:
            pyautogui.hotkey("shift", "n")
            speak("Playing next video.", mood="happy")
            return

        if state.LAST_APP_CONTEXT == "youtube" and (
            "play previous video" in command or "play prev video" in command
        ):
            pyautogui.hotkey("shift", "p")
            speak("Playing previous video.", mood="happy")
            return
        # --------------------------------------------------------------
        # YOUTUBE: PAUSE / RESUME VIDEO
        # --------------------------------------------------------------
        if state.LAST_APP_CONTEXT == "youtube" and "pause video" in command:
            pyautogui.press("k")
            speak("Video paused.", mood="neutral")
            return

        if state.LAST_APP_CONTEXT == "youtube" and (
            "resume video" in command or
            ("play video" in command and not state.LAST_YOUTUBE_SEARCH)
        ):
            pyautogui.press("k")
            speak("Resuming video.", mood="happy")
            return
        # --------------------------------------------------------------
        # YOUTUBE: MUTE / UNMUTE VIDEO
        # --------------------------------------------------------------
        if state.LAST_APP_CONTEXT == "youtube" and "mute video" in command:
            pyautogui.press("m")
            speak("Video muted.", mood="neutral")
            return

        if state.LAST_APP_CONTEXT == "youtube" and "unmute video" in command:
            pyautogui.press("m")
            speak("Video unmuted.", mood="happy")
            return
        # --------------------------------------------------------------
        # YOUTUBE: SEEK FORWARD / BACKWARD
        # --------------------------------------------------------------
        if state.LAST_APP_CONTEXT == "youtube" and "forward" in command:

            if "30" in command:
                presses = 6
            elif "10" in command:
                presses = 2
            else:
                presses = 1  # default = 5 seconds

            pyautogui.press("right", presses=presses, interval=0.15)
            speak("Forwarding video.", mood="neutral")
            return

        if state.LAST_APP_CONTEXT == "youtube" and any(x in command for x in ["backward", "back"]):

            if "30" in command:
                presses = 6
            elif "10" in command:
                presses = 2
            else:
                presses = 1  # default = 5 seconds

            pyautogui.press("left", presses=presses, interval=0.15)
            speak("Rewinding video.", mood="neutral")
            return
        # --------------------------------------------------------------
        # YOUTUBE: FULLSCREEN / EXIT FULLSCREEN
        # --------------------------------------------------------------
        if state.LAST_APP_CONTEXT == "youtube" and (
            "fullscreen video" in command or
            "full screen video" in command
        ):
            pyautogui.press("f")
            speak("Entering fullscreen.", mood="happy")
            return

        if state.LAST_APP_CONTEXT == "youtube" and (
            "exit fullscreen" in command or
            "close fullscreen" in command
        ):
            pyautogui.press("f")
            speak("Exiting fullscreen.", mood="neutral")
            return
        # --------------------------------------------------------------
        # BROWSER: BACK / FORWARD
        # --------------------------------------------------------------
        if any(x in command for x in ["go back", "browser back", "back page"]):
            pyautogui.hotkey("alt", "left")
            speak("Going back.", mood="neutral")
            return

        if any(x in command for x in ["go forward", "browser forward", "forward page"]):
            pyautogui.hotkey("alt", "right")
            speak("Going forward.", mood="neutral")
            return
        # --------------------------------------------------------------
        # YOUTUBE: PLAY CURRENTLY FOCUSED VIDEO
        # --------------------------------------------------------------
        if state.LAST_APP_CONTEXT == "youtube" and any(
            x in command for x in ["play this video", "play this one", "play this"]
        ):
            pyautogui.press("enter")
            set_last_action(None)
            speak("Playing this video.", mood="happy")
            state.LAST_APP_CONTEXT = "youtube"
            state.LAST_YOUTUBE_SEARCH = False
            return

        # --------------------------------------------------------------
        # YOUTUBE: CORE PLAYER CONTROLS (ALWAYS AVAILABLE)
        # --------------------------------------------------------------
        if YOUTUBE_ACTIVE and state.LAST_APP_CONTEXT == "youtube":

            # Play / Pause
            if any(command == x or command.startswith(x) for x in ["play", "pause", "resume"]):
                pyautogui.press("k")
                speak("Okay.", mood="neutral")
                return

            # Fullscreen toggle
            if any(x in command for x in ["fullscreen", "full screen"]):
                pyautogui.press("f")
                speak("Fullscreen.", mood="happy")
                return

            if any(x in command for x in ["exit fullscreen", "close fullscreen"]):
                pyautogui.press("f")
                speak("Exited fullscreen.", mood="neutral")
                return

            # Seek forward
            if "forward" in command:
                if "30" in command:
                    presses = 6
                elif "10" in command:
                    presses = 2
                else:
                    presses = 1
                pyautogui.press("right", presses=presses, interval=0.15)
                speak("Forwarding.", mood="neutral")
                return

            # Seek backward
            if any(x in command for x in ["backward", "rewind", "back"]):
                if "30" in command:
                    presses = 6
                elif "10" in command:
                    presses = 2
                else:
                    presses = 1
                pyautogui.press("left", presses=presses, interval=0.15)
                speak("Rewinding.", mood="neutral")
                return

        # --------------------------------------------------------------
        # SCROLL CONTROLS (FOCUS-SAFE & RELIABLE)
        # --------------------------------------------------------------
        if "scroll down" in command:
            # Force page focus (critical for YouTube / browsers)
            pyautogui.click(500, 400)
            time.sleep(0.1)

            amount = -600
            if any(x in command for x in ["little", "small", "slightly"]):
                amount = -300
            elif any(x in command for x in ["lot", "much", "fast"]):
                amount = -1200

            pyautogui.scroll(amount)
            speak("Scrolling down.")
            return

        if "scroll up" in command:
            # Force page focus (critical for YouTube / browsers)
            pyautogui.click(500, 400)
            time.sleep(0.1)

            amount = 600
            if any(x in command for x in ["little", "small", "slightly"]):
                amount = 300
            elif any(x in command for x in ["lot", "much", "fast"]):
                amount = 1200

            pyautogui.scroll(amount)
            speak("Scrolling up.")
            return

        # --------------------------------------------------------------
        # TAB NAVIGATION (NOT CLOSE)
        # --------------------------------------------------------------
        if "next tab" in command:
            pyautogui.hotkey("ctrl", "tab")
            speak("Switched tab.")
            return

        if "previous tab" in command or "prev tab" in command:
            pyautogui.hotkey("ctrl", "shift", "tab")
            speak("Going back a tab.")
            return
        # --------------------------------------------------------------
        # OPEN Nth TAB
        # --------------------------------------------------------------
        if "open" in command and "tab" in command:

            tab_map = {
                "first": "1", "1st": "1",
                "second": "2", "2nd": "2",
                "third": "3", "3rd": "3",
                "fourth": "4", "4th": "4",
                "fifth": "5", "5th": "5",
                "sixth": "6", "6th": "6",
                "seventh": "7", "7th": "7",
                "eighth": "8", "8th": "8",
                "ninth": "9", "9th": "9"
            }

            for word, key in tab_map.items():
                if word in command:
                    pyautogui.hotkey("ctrl", key)
                    speak(f"Opening {word} tab.")
                    return

        # --------------------------------------------------------------
        # PERSONALITY QUICK RESPONSES
        # --------------------------------------------------------------
        if "how are you" in command:
            mood = memory.get_mood()
            speak({
                "happy": "Feeling great today!",
                "neutral": "Calm and steady.",
                "alert": "Focused and ready.",
                "serious": "Here — just thinking deeply."
            }.get(mood, "All systems stable."), mood=mood)
            return

        if "thank you" in command or "thanks" in command:
            speak("Always for you, Yashu ❤️", mood="happy")
            return

        # --------------------------------------------------------------
        # FACTS / JOKES
        # --------------------------------------------------------------
        if "joke" in command:
            speak(random.choice([
                "Why did the computer get cold? Because it forgot to close its Windows.",
                "Parallel lines have so much in common. It’s a shame they’ll never meet."
            ]), mood="happy")
            return

        if "fact" in command:
            speak(random.choice([
                "Your brain generates enough electricity to power a small bulb.",
                "Honey never spoils — archaeologists found 3000-year-old honey still edible."
            ]), mood="happy")
            return

        # --------------------------------------------------------------
        # MEMORY COMMANDS
        # --------------------------------------------------------------
        if "remember" in command:
            try:
                if " that " in command:
                    fact = command.replace("remember that", "").strip()
                    if " is " in fact:
                        key, value = fact.split(" is ", 1)
                        memory.remember_fact(key.strip(), value.strip())
                        speak("Okay, I’ll remember that.", mood="neutral")
                    else:
                        speak("Say it like: remember that my laptop is Lenovo.")
                else:
                    speak("Please say it like: remember that ... is ...")
            except:
                speak("Couldn't save that memory.", mood="alert")
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
            try:
                memory.forget_fact(key)
                speak("Okay, I forgot it.", mood="neutral")
            except:
                speak("Couldn't forget that.", mood="alert")
            return

        # --------------------------------------------------------------
        # SHUTDOWN (CONFIRMATION MODE)
        # --------------------------------------------------------------
        if any(x in command for x in ["shutdown", "power off"]):
            speak("Are you sure you want to shut down?")
            PENDING_INTENT = {"intent": "shutdown_confirm"}
            return

        if PENDING_INTENT and PENDING_INTENT.get("intent") == "shutdown_confirm":
            if command in ["yes", "confirm", "do it", "yes shutdown"]:
                speak("Shutting down now.", mood="neutral")
                try:
                    jarvis_fx.stop_all()
                except:
                    pass
                os._exit(0)
            else:
                speak("Shutdown cancelled.")
            PENDING_INTENT = None
            return

        # --------------------------------------------------------------
        # AI / CONVERSATIONAL FALLBACK PIPELINE
        # --------------------------------------------------------------
        # Heuristic: “open / search / play / launch” should stay non-AI
        if (
           not YOUTUBE_ACTIVE
           and any(command.startswith(pref) for pref in ["open ", "launch ", "search ", "type "])
           and "tab" not in command
           and "window" not in command
            ):

            try:
                query = command
                for p in ["search ", "find ", "open ", "launch "]:
                    if query.startswith(p):
                        query = query.replace(p, "", 1).strip()
                if query:
                    webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
                    speak(f"I've searched for {query}.", mood="happy")
                    return
            except:
                pass

        # Spawn background AI worker
        try:
            t = threading.Thread(
                target=self._ai_pipeline_worker,
                args=(raw_command,),
                daemon=True
            )
            t.start()
        except:
            try:
                self._ai_pipeline_worker(raw_command)
            except:
                print("⚠️ Ultimate AI pipeline failure.")

    # --------------------------------------------------------------
    # AI WORKER (Background Thread)
    # --------------------------------------------------------------
    def _ai_pipeline_worker(self, raw_command):
        try:
            # --------------------------------------------------
            # Delayed, conditional "thinking" indicator
            # --------------------------------------------------
            cancel_thinking = threading.Event()

            thinking_thread = threading.Thread(
                target=maybe_say_thinking,
                args=(speak, cancel_thinking),
                daemon=True
            )
            thinking_thread.start()

            ai_response = None

            # 1) Try Ollama / local LLM first
            if AI_CHAT_AVAILABLE and ai_chat_brain:
                try:
                    ai_response = ai_chat_brain.ask(raw_command)
                except Exception:
                    ai_response = None

            # 2) Fallback to JarvisConversation
            if not ai_response:
                try:
                    ai_response = self.conversation.respond(raw_command)
                    cancel_thinking.set()
                except Exception:
                    ai_response = None

            # 3) Final fallback to brain.friend-mode reply
            if not ai_response:
                try:
                    ai_response = brain_module.brain.fallback_reply(raw_command)
                except Exception:
                    ai_response = "I didn’t get that — say it differently?"

            # ❗ Stop thinking message once response is ready (safe if already set)
            cancel_thinking.set()

            # 4) Mood reflection & store
            try:
                inferred = brain_module.brain.detect_text_emotion(ai_response)
                if inferred:
                    memory.set_mood(inferred)
            except Exception:
                pass

            # 5) Enhance with cinematic Jarvis styling (ONLY PLACE)
            try:
                enhanced = brain_module.brain.enhance_response(
                    ai_response,
                    mood=memory.get_mood(),
                    last_topic=memory.get_last_topic()
                )
            except Exception:
                enhanced = ai_response

            # 6) Keep system alive
            try:
                state.LAST_INTERACTION = time.time()
            except Exception:
                pass

            # 7) Speak AI response
            speak(enhanced)

        except Exception as e:
            try:
                cancel_thinking.set()
            except Exception:
                pass
            print("⚠️ AI error:", e)
            speak("Sorry, I couldn’t process that right now.")

def maybe_say_thinking(speak_fn, cancel_event):
    global _last_thinking_time

    time.sleep(THINKING_DELAY)

    # If response already arrived, do nothing
    if cancel_event.is_set():
        return

    # Throttle thinking messages
    if time.time() - _last_thinking_time < THINKING_COOLDOWN:
        return

    speak_fn("Thinking…")
    _last_thinking_time = time.time()



##############################################################################################################
# FILE: Jarvis\core\context.py
##############################################################################################################

# core/context.py
"""
Global shared context for Jarvis
Single source of truth for memory and state access
"""

from core.memory_engine import JarvisMemory
import core.state as state

memory = JarvisMemory()

def set_topic(topic):
    if topic:
        state.LAST_TOPIC = topic

def get_topic():
    return getattr(state, "LAST_TOPIC", None)

def set_mood(mood):
    if mood:
        state.JARVIS_MOOD = mood

def get_mood():
    return getattr(state, "JARVIS_MOOD", "neutral")

# --------------------------------------------------
# LAST ACTION TRACKING (YouTube / multi-turn logic)
# --------------------------------------------------

_LAST_ACTION = None

def set_last_action(action):
    """
    Store last high-level action like:
    'youtube_search'
    """
    global _LAST_ACTION
    _LAST_ACTION = action

def get_last_action():
    """
    Retrieve last stored action
    """
    return _LAST_ACTION


##############################################################################################################
# FILE: Jarvis\core\conversation_core.py
##############################################################################################################

# core/conversation_core.py
"""
Hybrid conversational core for Jarvis (Hybrid Mode - natural + consistent).
"""

import random
import re
from typing import Optional
from collections import deque

from core.brain import brain
from core.context import memory, set_topic, get_topic
from core.emotion_reflection import JarvisEmotionReflection

reflection = JarvisEmotionReflection()

import core.nlp_engine as nlp


def _word_bound_search(word_list, text):
    for w in word_list:
        if re.search(rf'\b{re.escape(w)}\b', text, flags=re.IGNORECASE):
            return True
    return False


class JarvisConversation:

    def __init__(self):
        print("🧩 Conversational Core Online (Hybrid Mode)")
        self.last_topic = None
        self.last_response = ""
        self.recent_fallbacks = deque(maxlen=8)
        self._recent_user_queries = deque(maxlen=12)

    def _estimate_sentiment(self, text: Optional[str]) -> str:
        if not text:
            return "neutral"

        t = text.lower()

        sad = ["sad", "low", "down", "hurt", "upset", "empty", "broken", "depressed", "lonely", "tear"]
        happy = ["happy", "great", "awesome", "nice", "good", "fantastic", "amazing", "glad", "yay", "excited"]
        angry = ["angry", "mad", "pissed", "furious", "hate", "annoyed"]
        anxious = ["scared", "worried", "anxious", "panic", "stressed", "stress", "overthinking", "nervous"]
        bored = ["bored", "meh", "boring", "idle"]

        score = 0
        if _word_bound_search(sad, t):
            score -= 2
        if _word_bound_search(angry, t):
            score -= 3
        if _word_bound_search(anxious, t):
            score -= 2
        if _word_bound_search(happy, t):
            score += 3
        if _word_bound_search(bored, t):
            score -= 1

        if re.search(r"\b(no|not|n't|never)\b", t):
            if _word_bound_search(happy, t):
                score -= 2
            if _word_bound_search(sad, t):
                score += 1

        if score >= 2:
            return "happy"
        if score <= -2:
            if _word_bound_search(angry, t):
                return "alert"
            return "serious"
        return "neutral"

    def _detect_topic(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None

        t = text.lower()
        topic_map = {
            "ai": ["ai", "artificial intelligence", "machine learning", "deep learning"],
            "java": ["java", "jvm", "spring"],
            "daa": ["daa", "dynamic programming", "algorithms", "graphs"],
            "python": ["python", "py"],
            "graphics": ["graphics", "opengl", "projection", "3d"],
            "dbms": ["dbms", "database", "sql", "mysql", "postgres", "oracle"],
            "blockchain": ["blockchain", "ethereum", "smart contract"],
            "gesture": ["gesture", "hand gesture", "gesture recognition"],
            "emotion": ["emotion", "mood", "feeling"],
            "life": ["life", "future", "career"],
            "love": ["love", "relationship", "gf", "bf", "crush"],
        }

        for key, aliases in topic_map.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", t):
                    return key

        m = re.search(r"\b([a-zA-Z]{3,20})\b", t)
        return m.group(1) if m else None

    def _continue_topic(self) -> str:
        if not self.last_topic:
            return "Continue what, Yashu? Remind me the topic and I'll follow up."

        base_templates = {
            "ai": "AI improves when you iterate on datasets and objectives. Want a small example or a project idea?",
            "java": "Java is excellent for large apps — focus on design patterns and testing. Want a sample structure?",
            "life": "Small consistent habits beat sudden bursts. Which habit should we plan first?",
            "love": "Communication and patience are key. Want a gentle script to start a conversation?",
            "daa": "For DAA, practice time/space trade-offs with real problems — want 3 practice problems?",
            "dbms": "Normalization and indexing matter. Want a quick explanation of normalization levels?"
        }

        return base_templates.get(
            self.last_topic,
            f"Let's explore more about {self.last_topic}. Which part interests you?"
        )

    def respond(self, text: Optional[str]) -> str:
        if not text:
            return "Yes Yashu? I'm listening."

        raw = text.strip()
        t = raw.lower()

        try:
            self._recent_user_queries.append(raw)
        except Exception:
            pass

        try:
            nlp.learn_async(raw)
        except Exception:
            pass

        try:
            mood = self._estimate_sentiment(t)
            memory.set_mood(mood)
            reflection.add_emotion(mood)
        except Exception:
            mood = memory.get_mood() or "neutral"

        if t in ("jarvis", "hey jarvis", "are you there", "yo jarvis", "jarvis bolo", "jarvis haan"):
            try:
                return brain.generate_wakeup_line(mood=memory.get_mood(), last_topic=self.last_topic)
            except Exception:
                return "Yes Yashu, I am here."

        try:
            if re.search(r"\b(i am|i'm|i feel|feeling)\b.*\b(sad|low|hurt|empty|depressed|lonely)\b", t):
                return brain.generate_emotional_support("sad", mood)
            if re.search(r"\b(i am|i'm|i feel|feeling)\b.*\b(happy|great|good|awesome|excited)\b", t):
                return brain.generate_emotional_support("happy", mood)
        except Exception:
            pass

        if any(w in t for w in ("continue", "more", "keep going", "tell me more")):
            return self._continue_topic()

        if re.search(r"\b(what|why|how|explain|help|define)\b", t):
            topic = self._detect_topic(t)
            self.last_topic = topic
            set_topic(topic)
            try:
                memory.update_topic(topic)
            except Exception:
                pass
            try:
                return brain.answer_question(t, topic, mood)
            except Exception:
                return f"I can explain {topic or 'that'} — short summary or a detailed explanation?"

        if any(w in t for w in ("open", "launch", "play", "type", "search", "screenshot", "volume", "brightness", "notepad", "whatsapp")):
            topic = self._detect_topic(t)
            self.last_topic = topic
            set_topic(topic)
            try:
                memory.update_topic(topic)
            except Exception:
                pass
            return random.choice([
                "On it, Yash. Doing that now.",
                "Got the command — executing.",
                "Alright — I'll take care of that."
            ])

        fallback_pool = [
            "Hmm… interesting. Want to explore that?",
            "Tell me more — I’m following you.",
            "You always think differently. What’s the next part?",
            "I can dive deeper into that if you want.",
            "Want a breakdown, a summary, or a story version?"
        ]

        reply = random.choice(fallback_pool)
        if self.recent_fallbacks and reply == self.recent_fallbacks[-1]:
            alt = [r for r in fallback_pool if r != reply]
            if alt:
                reply = random.choice(alt)

        try:
            recent_same = sum(1 for q in self._recent_user_queries if q.lower() == raw.lower())
            if recent_same >= 2:
                reply = "Seems like you want a clear answer. Do you want a short summary or a step-by-step example?"
        except Exception:
            pass

        try:
            self.last_topic = self._detect_topic(t)
            set_topic(self.last_topic)
            memory.update_topic(self.last_topic)
        except Exception:
            pass

        try:
            self.recent_fallbacks.append(reply)
        except Exception:
            pass

        self.last_response = reply
        return reply


##############################################################################################################
# FILE: Jarvis\core\desktop_control.py
##############################################################################################################

# core/desktop_control.py
"""
Desktop Executor
----------------
Single source of truth for all OS / desktop operations.
NO speech logic. NO AI logic.
"""

import os
import time
import subprocess
import ctypes
import pyautogui
import keyboard
import webbrowser


class DesktopControl:

    # ===================== VOLUME =====================
    def volume_control(self, action: str, smooth: bool = False):
        try:
            if smooth:
                key = "volume down" if action == "down" else "volume up"
                for _ in range(12):
                    keyboard.send(key)
                    time.sleep(0.05)
                return

            if action == "up":
                keyboard.send("volume up")
            elif action == "down":
                keyboard.send("volume down")
            elif action == "mute":
                keyboard.send("volume mute")
        except Exception:
            pass

    # ===================== BRIGHTNESS =====================
    def _set_brightness(self, level: int):
        level = max(0, min(100, int(level)))
        cmd = (
            "(Get-WmiObject -Namespace root/WMI "
            "-Class WmiMonitorBrightnessMethods)"
            f".WmiSetBrightness(1,{level})"
        )
        subprocess.call(["powershell.exe", "-Command", cmd])

    def _get_brightness(self) -> int:
        try:
            cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
            return int(subprocess.check_output(
                ["powershell.exe", "-Command", cmd]
            ).decode().strip())
        except:
            return 50

    def change_brightness(self, direction: str, smooth: bool = False):
        try:
            curr = self._get_brightness()
            if smooth:
                target = 10 if direction == "down" else 90
                step = -3 if direction == "down" else 3
                while (direction == "down" and curr > target) or \
                      (direction == "up" and curr < target):
                    curr += step
                    self._set_brightness(curr)
                    time.sleep(0.04)
            else:
                self._set_brightness(curr + (15 if direction == "up" else -15))
        except:
            pass
        

    # ===================== WINDOW =====================
    def show_desktop(self):
        keyboard.send("windows+d")

    def minimize_current(self):
        keyboard.send("windows+down")

    def maximize_current(self):
        keyboard.send("windows+up")

    def close_current(self):
        keyboard.send("alt+f4")

    def switch_window(self, reverse=False):
        if reverse:
            keyboard.send("alt+shift+tab")
        else:
            keyboard.send("alt+tab")

    # ===================== FILE / FOLDER =====================
    def open_folder(self, name: str):
        paths = {
            "downloads": os.path.join(os.environ["USERPROFILE"], "Downloads"),
            "documents": os.path.join(os.environ["USERPROFILE"], "Documents"),
            "desktop": os.path.join(os.environ["USERPROFILE"], "Desktop"),
            "pictures": os.path.join(os.environ["USERPROFILE"], "Pictures"),
        }
        path = paths.get(name)
        if path and os.path.exists(path):
            os.startfile(path)

     # ✅ STEP 2 — ADD THIS METHOD HERE
    def open_any_folder(self, folder_name: str):
        """
        Search common locations for a folder and open it if found.
        """
        search_roots = [
            os.environ["USERPROFILE"],
            os.path.join(os.environ["USERPROFILE"], "Desktop"),
            os.path.join(os.environ["USERPROFILE"], "Documents"),
            os.path.join(os.environ["USERPROFILE"], "Downloads"),
        ]

        folder_name = folder_name.lower()

        for root in search_roots:
            for dirpath, dirnames, _ in os.walk(root):
                for d in dirnames:
                    if d.lower() == folder_name:
                        os.startfile(os.path.join(dirpath, d))
                        return True

        return False        

    def open_file(self, path: str):
        if os.path.exists(path):
            os.startfile(path)

    # ===================== SYSTEM =====================
    def lock_screen(self):
        ctypes.windll.user32.LockWorkStation()

    def restart_system(self):
        os.system("shutdown /r /t 0")

    # ===================== UI / SETTINGS =====================
    def open_task_manager(self):
        os.system("start taskmgr")

    def open_settings(self):
        os.system("start ms-settings:")

    def toggle_dark_mode(self, enable: bool):
        val = 0 if enable else 1
        cmds = [
            rf"Set-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize -Name AppsUseLightTheme -Value {val}",
            rf"Set-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize -Name SystemUsesLightTheme -Value {val}",
        ]
        for c in cmds:
            subprocess.call(["powershell.exe", "-Command", c])

    # ===================== SCREENSHOT =====================
    def screenshot_clipboard(self):
        keyboard.send("printscreen")

    def screenshot_file(self):
        name = f"screenshot_{int(time.time())}.png"
        pyautogui.screenshot(name)
        return name

    # ===================== WEB =====================
    def open_url(self, url: str):
        webbrowser.open_new_tab(url)


##############################################################################################################
# FILE: Jarvis\core\document_reader.py
##############################################################################################################

# core/document_reader.py
"""
High-tech Document Reader + Summarizer for Jarvis.

Features:
- Reads PDF / DOCX / TXT / MD.
- Splits long documents into chunks (safe chunk-size).
- Optional improved summarization:
    - If OpenAI API key present -> uses OpenAI (chat/completions).
    - Else if local transformers summarization pipeline available -> uses it.
    - Else falls back to an in-process TextRank summarizer.
- Reads aloud using core.speech_engine.speak and can return textual summary.
- Non-blocking API: heavy ops run in background thread if used via `read_async` / `summarize_async`.
"""

import os
import threading
import math
from typing import Optional, List

# file readers
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

# optional advanced libs
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# optional OpenAI
try:
    import openai
    _OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))
except Exception:
    _OPENAI_AVAILABLE = False

from core.speech_engine import speak
import core.nlp_engine as nlp
from core.memory_engine import JarvisMemory

memory = JarvisMemory()


# -------------------------
# Utility: chunk text
# -------------------------
def _chunk_text(text: str, max_words: int = 350) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks


# -------------------------
# Simple TextRank fallback summarizer
# -------------------------
def _textrank_summarize(text: str, max_sentences: int = 5) -> str:
    # naive frequency-based sentence scorer (safe, no external deps)
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= max_sentences:
        return text
    # word freq
    words = re.findall(r"\w+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # score sentences
    sscore = []
    for s in sentences:
        s_words = re.findall(r"\w+", s.lower())
        if not s_words:
            sscore.append((0, s))
            continue
        score = sum(freq.get(w, 0) for w in s_words) / len(s_words)
        sscore.append((score, s))
    sscore.sort(reverse=True, key=lambda x: x[0])
    selected = [s for _, s in sscore[:max_sentences]]
    # keep original order
    selected_sorted = [s for s in sentences if s in selected]
    return " ".join(selected_sorted)


# -------------------------
# Summarizer orchestrator
# -------------------------
def _summarize_text(text: str, max_words_chunk=350, prefer="auto") -> str:
    """
    prefer: "openai", "transformers", "textrank", or "auto"
    """
    # Try OpenAI first if requested / available
    if prefer in ("openai", "auto") and _OPENAI_AVAILABLE:
        try:
            # chunk and prompt
            chunks = _chunk_text(text, max_words=max_words_chunk)
            summaries = []
            for ch in chunks:
                prompt = (
                    "Summarize the following text into 4-6 concise bullet points. "
                    "Be precise and keep technical terms if present:\n\n" + ch
                )
                # Use ChatCompletion if available
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list() else "gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=450
                    )
                    summary = resp.choices[0].message.content.strip()
                except Exception:
                    # fallback to completion
                    resp = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=450,
                        temperature=0.0
                    )
                    summary = resp.choices[0].text.strip()
                summaries.append(summary)
            return "\n\n".join(summaries)
        except Exception:
            pass

    # Try transformers summarizer if available
    if prefer in ("transformers", "auto") and _TRANSFORMERS_AVAILABLE:
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            chunks = _chunk_text(text, max_words=max_words_chunk)
            outs = []
            for ch in chunks:
                out = summarizer(ch, max_length=130, min_length=30, do_sample=False)
                outs.append(out[0]["summary_text"])
            return "\n\n".join(outs)
        except Exception:
            pass

    # Final fallback - textrank
    return _textrank_summarize(text, max_sentences=6)


# -------------------------
# DocumentReader
# -------------------------
class DocumentReader:
    def __init__(self):
        self._thread = None

    def _extract_text_pdf(self, path: str) -> str:
        if not PyPDF2:
            raise RuntimeError("PyPDF2 not installed")
        try:
            reader = PyPDF2.PdfReader(path)
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                pages.append(t)
            return "\n\n".join(pages)
        except Exception:
            return ""

    def _extract_text_docx(self, path: str) -> str:
        if not docx:
            raise RuntimeError("python-docx not installed")
        try:
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception:
            return ""

    def _extract_text_plain(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _read_chunks_aloud(self, chunks: List[str]):
        for ch in chunks:
            if ch.strip():
                speak(ch.strip(), mood="neutral")

    def read(self, path: str, summarize_first: bool = False, prefer_summarizer: str = "auto") -> Optional[str]:
        """
        Synchronous read: extracts text, optionally summarizes, then speaks and returns summary text.
        If document is very long, this may block. Use read_async for non-blocking.
        """
        if not path or not os.path.exists(path):
            speak("I couldn't find that file.", mood="alert")
            return None

        ext = os.path.splitext(path)[1].lower()
        text = ""
        try:
            if ext == ".pdf":
                text = self._extract_text_pdf(path)
            elif ext in (".docx", ".doc"):
                text = self._extract_text_docx(path)
            else:
                text = self._extract_text_plain(path)
        except Exception:
            text = ""

        if not text.strip():
            speak("The document is empty or unreadable.", mood="alert")
            return None

        # If summary requested, do summarization first and speak summary then offer to read details
        if summarize_first:
            speak("Creating a concise summary of this document...", mood="neutral")
            summary = _summarize_text(text, prefer=prefer_summarizer)
            # speak summary
            speak("Here is a short summary:", mood="happy")
            speak(summary, mood="neutral")
            # offer to read full text
            return summary

        # Normal read: chunk and speak
        chunks = _chunk_text(text, max_words=300)
        self._read_chunks_aloud(chunks)
        return None

    def read_async(self, path: str, summarize_first: bool = False, prefer_summarizer: str = "auto"):
        t = threading.Thread(target=self.read, args=(path, summarize_first, prefer_summarizer), daemon=True)
        t.start()
        self._thread = t
        return t


# singleton
document_reader = DocumentReader()


##############################################################################################################
# FILE: Jarvis\core\emotion_reflection.py
##############################################################################################################

# core/emotion_reflection.py
"""
Emotion Reflection Engine — Cinematic Version
Keeps emotional history and generates natural reflections.
Fully compatible with:
- brain.py (mood fusion)
- memory_engine.py (shared memory)
- conversation_core.py (dynamic mood flow)
"""

import datetime
import random

from core.context import memory as shared_memory
from core.speech_engine import speak

class JarvisEmotionReflection:
    """Tracks mood history and provides soft emotional insights."""

    def __init__(self):
        from core.memory_engine import JarvisMemory
        self.memory = JarvisMemory()

        # ensure shared emotional history exists only once
        if "emotion_history" not in shared_memory.memory:
            shared_memory.memory["emotion_history"] = []
            shared_memory._save_memory()
        print("🧠 Emotion Reflection Engine Ready")

    # ----------------------------------------------------------
    def add_emotion(self, mood: str):
        """Record mood safely (store only last 12 moods)."""
        if mood not in ["happy", "serious", "neutral", "alert"]:
            mood = "neutral"

        entry = {
            "mood": mood,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        hist = shared_memory.memory.get("emotion_history", [])
        hist.append(entry)

        # keep only last 12 moods
        shared_memory.memory["emotion_history"] = hist[-12:]
        shared_memory._save_memory()

    # ----------------------------------------------------------
    def reflect(self, last_topic=None):
        """
        Reflect on emotional patterns — called only when asked.
        Does NOT interrupt user naturally.
        """

        hist = shared_memory.memory.get("emotion_history", [])
        if not hist:
            speak("I don't have enough emotional data yet, Yash.", mood="neutral")
            return

        # most recent moods
        last = hist[-1]["mood"]
        prev = hist[-2]["mood"] if len(hist) > 1 else None

        # ------------------------------------------------------
        # Mood transition (cinematic)
        # ------------------------------------------------------
        if prev and last != prev:
            transitions = {
                ("serious", "happy"): [
                    "You seem brighter now, Yash.",
                    "Your voice feels lighter than before."
                ],
                ("happy", "serious"): [
                    "You feel quieter suddenly. Is something on your mind?",
                    "Your tone shifted… I'm here if you want to talk."
                ],
                ("alert", "happy"): [
                    "I can sense relief in your tone.",
                    "You seem calmer compared to earlier."
                ],
                ("happy", "alert"): [
                    "You sounded cheerful earlier… but now something feels tense.",
                    "Your energy changed suddenly. Want to talk?"
                ],
                ("neutral", "happy"): [
                    "You sound a bit more cheerful now.",
                    "A positive shift — I love that energy."
                ],
                ("neutral", "serious"): [
                    "You seem more focused than a moment ago.",
                    "Your tone feels a bit heavier… everything okay?"
                ]
            }

            key = (prev, last)
            if key in transitions:
                speak(random.choice(transitions[key]), mood=last)
                return

        # ------------------------------------------------------
        # Dominant mood analysis (last 6 moods)
        # ------------------------------------------------------
        moods = [m["mood"] for m in hist[-6:]]
        freq = {m: moods.count(m) for m in set(moods)}
        dominant = max(freq, key=freq.get)

        reflections = {
            "happy": [
                "You've sounded positive lately — it’s refreshing.",
                "Love this brightness in your tone, Yash."
            ],
            "serious": [
                "You've been calm and thoughtful recently.",
                "Your tone feels focused — I admire that."
            ],
            "neutral": [
                "Your mood seems steady and balanced.",
                "You've been consistent and composed lately."
            ],
            "alert": [
                "You’ve sounded a bit tense in recent moments.",
                "I sense stress in your tone… I’m right here for you."
            ]
        }

        line = random.choice(reflections.get(dominant, reflections["neutral"]))

        # Add cinematic continuity (optional)
        if last_topic:
            line += f" And earlier we were talking about {last_topic}. Want to continue?"

        speak(line, mood=dominant)


##############################################################################################################
# FILE: Jarvis\core\face_auth.py
##############################################################################################################

# core/face_auth.py
"""
Simple & Accurate face authentication using OpenCV LBPH + Haar cascade.
Usage:
    python -m core.face_auth enroll --name yash config/face_data/yash_reference.jpg
    python -m core.face_auth train
    python -m core.face_auth verify_image --name yash config/face_data/yash_reference.jpg
    python -m core.face_auth live_verify --name yash
"""

from __future__ import annotations
import os
import sys
import pathlib
import json
import time
import shutil
from typing import Tuple, List, Optional

# OpenCV + numpy (required)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception as e:
    print("⚠️ OpenCV not available:", e)
    OPENCV_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# -------------------- Paths & config --------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]  # project root (one level up from core/)
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

LBPH_MODEL_FILE = MODEL_DIR / "lbph_model.yml"
LABELS_JSON = MODEL_DIR / "labels.json"
CASCADE_FILE = MODEL_DIR / "haarcascade_frontalface_default.xml"

# Standard face size
STANDARD_FACE_SIZE = (200, 200)

# default LBPH threshold (lower = stricter). Tweak if necessary.
DEFAULT_THRESHOLD = 70.0

# -------------------- Helpers --------------------
def normalize_path(p: str) -> str:
    if not p:
        return p
    return os.path.normpath(p)

def save_json(path: pathlib.Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: pathlib.Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Ensure cascade --------------------
def ensure_cascade() -> bool:
    """
    Make sure there's a local copy of haarcascade in MODEL_DIR.
    Tries to copy from cv2.data.haarcascades if available.
    """
    try:
        if CASCADE_FILE.exists():
            return True
        if not OPENCV_AVAILABLE:
            return False
        cascade_src = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_src):
            shutil.copyfile(cascade_src, str(CASCADE_FILE))
            return True
    except Exception:
        pass
    return False

# -------------------- Image reading --------------------
def read_image(path: str):
    path = normalize_path(path)
    if not path:
        return None
    # handle unicode / Windows long paths by using numpy + imdecode
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        if arr is not None and arr.size:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    # fallback
    try:
        return cv2.imread(path)
    except Exception:
        return None

def to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------- Face detection & preprocessing --------------------
def detect_faces_in_image(img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) -> List[Tuple[int,int,int,int]]:
    if not OPENCV_AVAILABLE:
        return []
    if not ensure_cascade():
        return []
    gray = to_gray(img)
    if gray is None:
        return []
    cascade = cv2.CascadeClassifier(str(CASCADE_FILE))
    faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    try:
        return faces.tolist() if hasattr(faces, "tolist") else list(faces)
    except Exception:
        return list(faces)

def crop_face(img, box, margin=0.25):
    if img is None:
        return None
    x, y, w, h = box
    h_img, w_img = img.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)
    return img[y1:y2, x1:x2]

def prepare_face_for_training(img):
    """
    Return grayscale (STANDARD_FACE_SIZE) face or None.
    Uses largest detected face; falls back to center crop if no face found.
    """
    if img is None:
        return None
    boxes = detect_faces_in_image(img, scaleFactor=1.22, minNeighbors=7, minSize=(120,120))
    if boxes:
        boxes_sorted = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        face_crop = crop_face(img, boxes_sorted[0], margin=0.25)
    else:
        # center crop fallback
        h, w = img.shape[:2]
        side = min(h, w)
        cx = w // 2
        cy = h // 2
        x1 = max(0, cx - side//2)
        y1 = max(0, cy - side//2)
        face_crop = img[y1:y1+side, x1:x1+side]

    if face_crop is None:
        return None

    gray = to_gray(face_crop)
    if gray is None:
        return None
    try:
        resized = cv2.resize(gray, STANDARD_FACE_SIZE, interpolation=cv2.INTER_AREA)
        return resized
    except Exception:
        return None

# -------------------- Dataset builder --------------------
def load_face_dataset():
    """
    Reads data/faces/<person>/*(.jpg|.png) and returns:
        images: list of numpy arrays (grayscale)
        labels: list of ints
        label_names: list mapping index -> person name
    """
    images = []
    labels = []
    label_names = []
    if not FACES_DIR.exists():
        return images, labels, label_names
    people = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])
    for idx, person_dir in enumerate(people):
        person_name = person_dir.name
        label_names.append(person_name)
        # accept jpg/png
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_file in sorted(person_dir.glob(pattern)):
                img = read_image(str(img_file))
                face = prepare_face_for_training(img)
                if face is not None:
                    images.append(face)
                    labels.append(idx)
    return images, labels, label_names

# -------------------- Train LBPH --------------------
def train_lbph_model() -> bool:
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        print("❌ Cannot train — OpenCV or numpy missing.")
        return False

    images, labels, label_names = load_face_dataset()
    if len(images) == 0:
        print("❌ No face images available in data/faces/. Enroll first.")
        return False

    # ensure face module exists (opencv-contrib)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("❌ LBPH recognizer unavailable. Ensure 'opencv-contrib-python' is installed.", e)
        return False

    try:
        recognizer.train(images, np.array(labels))
        recognizer.write(str(LBPH_MODEL_FILE))
        mapping = {str(i): name for i, name in enumerate(label_names)}
        save_json(LABELS_JSON, mapping)
        print("✓ LBPH model trained and saved.")
        return True
    except Exception as e:
        print("❌ Training failed:", e)
        return False

# -------------------- Load LBPH --------------------
def load_lbph():
    if not LBPH_MODEL_FILE.exists():
        return None, None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(LBPH_MODEL_FILE))
    except Exception as e:
        print("❌ Could not load LBPH model:", e)
        return None, None
    labels = load_json(LABELS_JSON, {})
    return recognizer, labels

# -------------------- Enrollment --------------------
def enroll_image(name: str, image_path: str) -> bool:
    """
    Enrolls face by saving BOTH:
    - raw image (full resolution)
    - processed 200x200 cropped face image
    """
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available.")
        return False

    image_path = normalize_path(image_path)

    if not os.path.exists(image_path):
        print(f"❌ enroll_image: file not found: {image_path}")
        return False

    img = read_image(image_path)
    if img is None:
        print("❌ Could not read image.")
        return False

    face = prepare_face_for_training(img)
    if face is None:
        print("❌ No detectable face — try a clearer image.")
        return False

    save_dir = FACES_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)

    # Save RAW original image (helps model learn real variations)
    raw_path = save_dir / f"{name}_RAW_{timestamp}.jpg"
    cv2.imwrite(str(raw_path), img)

    # Save PROCESSED cropped face for LBPH training
    proc_path = save_dir / f"{name}_PROC_{timestamp}.jpg"
    cv2.imwrite(str(proc_path), face)

    print(f"✓ Enrolled RAW: {raw_path}")
    print(f"✓ Enrolled PROCESSED: {proc_path}")

    return True



# -------------------- Verify static image --------------------
def verify_face_image(name: str, img_path: str, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    """
    Returns (ok, confidence). Lower confidence = better match for LBPH.
    """
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available.")
        return False, 0.0
    recognizer, labels = load_lbph()
    if recognizer is None:
        print("❌ LBPH model not loaded. run: python -m core.face_auth train")
        return False, 0.0
    img = read_image(img_path)
    if img is None:
        print("❌ Could not read image:", img_path)
        return False, 0.0
    face = prepare_face_for_training(img)
    if face is None:
        print("❌ Could not detect face in image.")
        return False, 0.0
    try:
        predicted_label, confidence = recognizer.predict(face)
    except Exception as e:
        print("❌ Prediction failed:", e)
        return False, 0.0
    # check label mapping
    if str(predicted_label) in labels and labels[str(predicted_label)] == name:
        ok = confidence <= threshold
        return ok, float(confidence)
    return False, float(confidence)

# -------------------- Live verification (webcam) --------------------
def live_verify(name: str, attempts: int = 20, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available — live verify disabled.")
        return False, 0.0
    recognizer, labels = load_lbph()
    if recognizer is None:
        print("❌ LBPH model not loaded.")
        return False, 0.0
    # find target label
    target_label = None
    for k, v in (labels or {}).items():
        if v == name:
            try:
                target_label = int(k)
            except Exception:
                continue
            break
    if target_label is None:
        print(f"❌ No trained images found for {name}")
        return False, 0.0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return False, 0.0
    print("📷 Live verification started. Look at the camera...")
    best_conf = 9999.0
    try:
        for i in range(attempts):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ensure cascade exists
            if not ensure_cascade():
                print("❌ cascade not found.")
                break
            cascade = cv2.CascadeClassifier(str(CASCADE_FILE))
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
            if len(faces) == 0:
                # no face, continue
                time.sleep(0.05)
                continue
            # largest face
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face = gray[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, STANDARD_FACE_SIZE)
            except Exception:
                continue
            try:
                predicted, confidence = recognizer.predict(face)
            except Exception as e:
                print("❌ predict error:", e)
                continue
            best_conf = min(best_conf, float(confidence))
            if predicted == target_label and confidence <= threshold:
                print(f"✓ VERIFIED: {name} (confidence={confidence:.2f})")
                cap.release()
                return True, float(confidence)
            # small sleep
            time.sleep(0.06)
    except KeyboardInterrupt:
        print("\n❌ Verification aborted by user.")
    finally:
        cap.release()
    print(f"FAILED — best confidence={best_conf:.2f}")
    return False, best_conf

# -------------------- CLI glue --------------------
def enroll_image_cli(name: str, image: str) -> bool:
    return enroll_image(name, image)

def train_lbph() -> bool:
    return train_lbph_model()

def verify_image_cli(name: str, image: str, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    return verify_face_image(name, image, threshold)

def live_verify_cli(name: str, attempts: int = 20, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float]:
    return live_verify(name, attempts=attempts, threshold=threshold)

# -------------------- Main --------------------
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser(description="Simple LBPH face authentication")
    sub = parser.add_subparsers(dest="command")

    enroll_p = sub.add_parser("enroll", help="Enroll a new face")
    enroll_p.add_argument("--name", required=True, help="User name to enroll")
    enroll_p.add_argument("image", help="Path to face image")

    train_p = sub.add_parser("train", help="Train LBPH model from enrolled images")

    verify_p = sub.add_parser("verify_image", help="Verify using a static image")
    verify_p.add_argument("--name", required=True, help="User name")
    verify_p.add_argument("image", help="Path to image")
    verify_p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    live_p = sub.add_parser("live_verify", help="Verify using webcam")
    live_p.add_argument("--name", required=True, help="User name")
    live_p.add_argument("--attempts", type=int, default=20)
    live_p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    args = parser.parse_args(argv)

    # basic pre-check
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        print("ERROR: OpenCV and numpy required. Install: pip install opencv-contrib-python numpy")
        return

    # ensure cascade present (best-effort)
    if not ensure_cascade():
        print("⚠️ Warning: Haar cascade not found and couldn't be copied. Some face detection may fail.")
    else:
        print("✓ Haar cascade ready.")

    if args.command == "enroll":
        ok = enroll_image_cli(args.name, args.image)
        print(f"ENROLL: {ok}")
        return
    elif args.command == "train":
        ok = train_lbph()
        print(f"TRAIN: {ok}")
        return
    elif args.command == "verify_image":
        ok, score = verify_image_cli(args.name, args.image, threshold=args.threshold)
        print(f"VERIFY_IMAGE: ok={ok} score={score}")
        return
    elif args.command == "live_verify":
        ok, score = live_verify_cli(args.name, attempts=args.attempts, threshold=args.threshold)
        print(f"LIVE_VERIFY: ok={ok} score={score}")
        return
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


##############################################################################################################
# FILE: Jarvis\core\face_emotion.py
##############################################################################################################

# core/face_emotion.py
"""
Face Emotion Analyzer for Jarvis
SAFE VERSION — No TensorFlow required.
Uses DeepFace only if available; otherwise falls back to OpenCV Haarcascade.

Fully compatible with:
- shared memory engine
- brain mood fusion
- emotion reflection
"""

import cv2
import time

from core.memory_engine import shared_memory
from core.speech_engine import speak
from core.voice_effects import JarvisEffects

jarvis_fx = JarvisEffects()

# Try loading DeepFace (optional)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = False
except Exception:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available — using fallback face analyzer.")

# Load Haarcascade (fallback)
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class FaceEmotionAnalyzer:
    """Detects user emotion from camera with safe fallback."""

    def __init__(self):
        print("📸 Face Emotion Analyzer Ready (Safe Mode)")
    
    # ---------------------------------------------------------
    def capture_emotion(self):
        """Capture one frame and analyze emotion in safe mode."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None

        time.sleep(0.4)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # ----------------------------------------------------
        # If DeepFace exists → use it (max accuracy)
        # ----------------------------------------------------
        if DEEPFACE_AVAILABLE:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
                detected = result.get("dominant_emotion", "neutral").lower()
                return self._apply_mood(detected)

            except Exception as e:
                print("⚠️ DeepFace failed, switching to fallback:", e)

        # ----------------------------------------------------
        # FALLBACK → Basic haar detection + neutral guess
        # ----------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = HAAR.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return self._apply_mood("neutral")

        # fallback cannot classify → treat as neutral
        return self._apply_mood("neutral")

    # ---------------------------------------------------------
    # MOOD MAP + CINEMATIC RESPONSE
    # ---------------------------------------------------------
    def _apply_mood(self, emotion):
        """Maps detected emotion to Jarvis mood system."""

        emotion = emotion.lower()

        mood_map = {
            "happy": "happy",
            "surprise": "happy",
            "sad": "serious",
            "fear": "alert",
            "angry": "alert",
            "disgust": "serious",
            "neutral": "neutral"
        }

        jarvis_mood = mood_map.get(emotion, "neutral")

        # update global memory
        shared_memory.set_mood(jarvis_mood)
        jarvis_fx.mood_tone(jarvis_mood)

        # CINEMATIC REPLIES (trigger only occasionally)
        if emotion == "happy":
            speak("You look bright today, Yash.", mood="happy")

        elif emotion in ["sad", "fear"]:
            speak("You look a bit low… I'm here with you.", mood="serious")

        elif emotion == "angry":
            speak("Your expression seems tense — breathe with me.", mood="alert")

        return emotion


##############################################################################################################
# FILE: Jarvis\core\intent_parser.py
##############################################################################################################

# core/intent_parser.py
"""
Intent parser for Jarvis — lightweight, robust, and natural-language friendly.

Usage:
    from core.intent_parser import parse_intent
    intent = parse_intent("it's too bright in here")
    # intent -> {"intent": "adjust_brightness", "action": "decrease", "reason": "too bright", "confidence": 0.9}

Design:
- Pattern + keyword based (fast and offline)
- Returns a dict: { intent, confidence, params }
- Defensive: never throws; returns {"intent":"unknown", ...} on edge cases
"""

import re
from typing import Dict, Any, Optional

# small helper synonyms
_BRIGHTNESS_UP = ["increase brightness", "brightness up", "brighten", "more brightness", "it's dark", "i can't see", "not able to see", "low brightness", "too dark", "hard to see"]
_BRIGHTNESS_DOWN = ["decrease brightness", "brightness down", "dim", "dark", "too bright", "it's too bright", "reduce brightness", "bright is too much", "blinding"]
_VOLUME_UP = ["volume up", "increase volume", "louder", "sound up", "turn it up", "too low"]
_VOLUME_DOWN = ["volume down", "decrease volume", "lower volume", "dheere", "turn it down", "too loud"]
_MUTE = ["mute", "silence", "shut up", "quiet"]
_UNMUTE = ["unmute", "turn on sound"]
_SCREENSHOT = ["screenshot", "take screenshot", "save screen", "grab screen", "capture screen"]
_SEARCH = ["search", "find", "look up", "dhund", "search kar", "google", "on youtube"]
_OPEN = ["open", "launch", "start", "take me to"]
_CLOSE = ["close", "exit", "quit", "shutdown window"]
_PLAY = ["play", "pause", "resume", "stop", "play music", "pause music"]
_TYPE = ["type", "type this", "type message", "type that"]
_REMEMBER = ["remember that", "remember"]
_FORGET = ["forget", "forget that"]
_HELP = ["help", "explain", "how to", "what is", "why"]

def _contains_any(text: str, keywords):
    t = text.lower()
    for k in keywords:
        if k in t:
            return True
    return False

def _match_regex(text: str, patterns):
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m
    return None

def parse_intent(text: Optional[str]) -> Dict[str, Any]:
    """Parse user utterance into an intent dict.

    Returns:
      {
        "intent": str,
        "confidence": float (0.0-1.0),
        "params": { ... }  # intent-specific
      }
    """
    if not text or not text.strip():
        return {"intent": "none", "confidence": 0.0, "params": {}}

    t = text.lower().strip()

    # 1) Explicit memory patterns
    try:
        if "remember that" in t or t.startswith("remember "):
            # try "remember that key is value"
            m = re.search(r"remember (?:that )?(?P<k>[\w\s]+?) (?:is|=|to be) (?P<v>.+)", t)
            if m:
                return {
                    "intent": "remember_fact",
                    "confidence": 0.95,
                    "params": {"key": m.group("k").strip(), "value": m.group("v").strip()}
                }
            else:
                # <remember> with free text
                rest = t.replace("remember that", "").replace("remember", "").strip()
                return {"intent": "remember_prompt", "confidence": 0.8, "params": {"text": rest}}
    except Exception:
        pass

    # 2) Forget pattern
    try:
        if t.startswith("forget ") or " forget " in t:
            key = t.replace("forget", "").strip()
            return {"intent": "forget_fact", "confidence": 0.9, "params": {"key": key}}
    except Exception:
        pass

    # 3) Screenshot
    if _contains_any(t, _SCREENSHOT) or re.search(r"\b(screen shot|screen-shot|screen shot to)\b", t):
        # optional folder/file name
        m = re.search(r"(?:as|named|called)\s+([^\s]+(?:\.[a-zA-Z0-9]{2,4})?)", text, flags=re.I)
        filename = m.group(1) if m else None
        return {"intent": "screenshot", "confidence": 0.95, "params": {"filename": filename}}

    # 4) Brightness adjustments — include natural phrasing (user complaints)
    try:
        # direct increase/decrease commands
        if _contains_any(t, _BRIGHTNESS_UP):
            # if user says "i can't see" -> increase
            return {"intent": "adjust_brightness", "confidence": 0.95, "params": {"action": "increase", "reason": t}}
        if _contains_any(t, _BRIGHTNESS_DOWN):
            return {"intent": "adjust_brightness", "confidence": 0.95, "params": {"action": "decrease", "reason": t}}

        # complaint + qualifier mapping (eg: "too bright" => decrease)
        if re.search(r"\btoo bright\b|\bblinding\b|\bso bright\b", t):
            return {"intent": "adjust_brightness", "confidence": 0.98, "params": {"action": "decrease", "reason": "too bright"}}
        if re.search(r"\btoo dark\b|\bcan't see\b|\bnot able to see\b|\blow brightness\b", t):
            return {"intent": "adjust_brightness", "confidence": 0.98, "params": {"action": "increase", "reason": "too dark"}}
    except Exception:
        pass

    # 5) Volume adjustments & mute
    try:
        if _contains_any(t, _VOLUME_UP):
            return {"intent": "adjust_volume", "confidence": 0.95, "params": {"action": "up"}}
        if _contains_any(t, _VOLUME_DOWN):
            return {"intent": "adjust_volume", "confidence": 0.95, "params": {"action": "down"}}
        if "mute" in t and "unmute" not in t:
            return {"intent": "mute", "confidence": 0.98, "params": {}}
        if "unmute" in t:
            return {"intent": "unmute", "confidence": 0.98, "params": {}}
    except Exception:
        pass

    # 6) Open / launch / app-specific
    try:
        if any(k in t for k in _OPEN):
            # capture app/site name
            m = re.search(r"(?:open|launch|start|take me to)\s+(?P<app>[\w\.\-/ ]+)", t)
            app = m.group("app").strip() if m else None
            return {"intent": "open_app", "confidence": 0.9, "params": {"app": app}}
    except Exception:
        pass

    # 7) Search queries (explicit)
    try:
        if any(k in t for k in _SEARCH):
            # extract "search for X" or fallback whole phrase minus the verb
            m = re.search(r"(?:search (?:for|about)?|find|look up)\s+(?P<q>.+)", t)
            q = m.group("q").strip() if m else re.sub(r"\b(search|find|look up|dhund|search kar|on youtube|on google)\b", "", t).strip()
            return {"intent": "search", "confidence": 0.9, "params": {"query": q}}
    except Exception:
        pass

    # 8) Typing commands
    try:
        if any(k in t for k in _TYPE):
            # get content after 'type' keywords
            s = re.sub(r"(type this|type message|type that|type kar|type)\s*", "", t)
            if s:
                return {"intent": "type_text", "confidence": 0.92, "params": {"text": s}}
            else:
                return {"intent": "type_text_prompt", "confidence": 0.7, "params": {}}
    except Exception:
        pass

    # 9) Media controls
    try:
        if any(k in t for k in _PLAY):
            if "pause" in t:
                return {"intent": "media_pause", "confidence": 0.9, "params": {}}
            if "play" in t or "resume" in t:
                return {"intent": "media_play", "confidence": 0.9, "params": {}}
            if "stop" in t:
                return {"intent": "media_stop", "confidence": 0.9, "params": {}}
    except Exception:
        pass

    # 10) Explicit direct questions / help
    try:
        if any(k in t for k in _HELP):
            # return generic help intent + original text
            return {"intent": "ask_question", "confidence": 0.8, "params": {"text": text}}
    except Exception:
        pass

    # 11) Short responses / confirmations / negatives
    try:
        if re.fullmatch(r"\b(yes|yeah|yup|y|sure|ok|okay)\b", t):
            return {"intent": "confirm", "confidence": 0.95, "params": {}}
        if re.fullmatch(r"\b(no|nah|nope|don't|dont|stop)\b", t):
            return {"intent": "deny", "confidence": 0.95, "params": {}}
    except Exception:
        pass

    # 12) Memory recall: "what is my X" -> recall
    try:
        m = re.match(r"(?:what is|what's|tell me) (?:my )?(?P<k>[\w\s]+)\??", t)
        if m and ("what is" in t or "what's" in t):
            return {"intent": "recall_fact", "confidence": 0.8, "params": {"key": m.group("k").strip()}}
    except Exception:
        pass

    # 13) Fallback: try to detect simple "visibility complaint" to map to brightness
    try:
        if re.search(r"\b(i can(')?t see|can't see|not able to see|blur|too dark|can't read)\b", t):
            return {"intent": "adjust_brightness", "confidence": 0.88, "params": {"action": "increase", "reason": t}}
    except Exception:
        pass

    # 14) Unknown intent: return raw text for conversation fallback
    return {"intent": "unknown", "confidence": 0.35, "params": {"text": text}}


##############################################################################################################
# FILE: Jarvis\core\interface.py
##############################################################################################################

# core/interface.py — UPGRADED PART 1
import sys
import math
import time
import threading
import numpy as np
import sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets

# Keep the same class name and public API (run, stop, set_status, set_mood, react_to_audio)

class InterfaceOverlay(QtWidgets.QWidget):
    """Floating Siri-style circular overlay — mic-reactive + Jarvis mood reactive."""
    # Signals for thread-safe updates from non-Qt threads (sounddevice callback, listener threads)
    sig_set_status = QtCore.pyqtSignal(str)
    sig_set_mood = QtCore.pyqtSignal(str)
    sig_react_audio = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()

        # Frameless, always-on-top, translucent
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.resize(220, 220)

        # Visual state (only mutated on the Qt thread)
        self._pulse_angle = 0.0
        self._reactive_boost = 0.0
        self._mic_intensity = 0.0
        self._status_text = "Booting..."
        self._mood = "neutral"
        self.running = True

        # Mic stream control
        self._mic_stream = None
        self._mic_thread = None
        self._mic_enabled = True  # try to keep same feature; turns False if device busy

        # Timers (Qt timers run on the Qt main loop and won't block paintEvent)
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(16)   # ~60 FPS
        self._anim_timer.timeout.connect(self._on_anim_tick)

        # Fade animation (Qt property animation avoids time.sleep)
        self._fade_anim = QtCore.QPropertyAnimation(self, b"windowOpacity", self)

        # Mood color sets (used in paint)
        self.mood_colors = {
            "happy": [QtGui.QColor(60, 220, 200), QtGui.QColor(0, 160, 255)],
            "serious": [QtGui.QColor(255, 140, 0), QtGui.QColor(255, 80, 0)],
            "alert": [QtGui.QColor(255, 60, 90), QtGui.QColor(255, 0, 0)],
            "neutral": [QtGui.QColor(0, 200, 255), QtGui.QColor(120, 80, 255)]
        }

        # Connect signals to slots (thread-safe)
        self.sig_set_status.connect(self._slot_set_status)
        self.sig_set_mood.connect(self._slot_set_mood)
        self.sig_react_audio.connect(self._slot_react_audio)

        # Precompute common painter objects used every frame to reduce allocations
        self._cached_font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold)
        self._cached_pen = QtGui.QPen(QtCore.Qt.NoPen)

        # Ensure a clean initial opacity
        self.setWindowOpacity(0.0)

    # ---------------- Public Controls ----------------
    def react_to_audio(self, intensity=1.0):
        """Public method (thread-safe) to indicate audio activity."""
        # If called from non-Qt thread, forward via signal
        if QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread():
            self.sig_react_audio.emit(float(intensity))
        else:
            self._slot_react_audio(float(intensity))

    def set_status(self, text):
        """Thread-safe setter for status text."""
        if QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread():
            self.sig_set_status.emit(str(text))
        else:
            self._slot_set_status(str(text))

    def set_mood(self, mood):
        """Thread-safe setter for mood."""
        if QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread():
            self.sig_set_mood.emit(str(mood))
        else:
            self._slot_set_mood(str(mood))

    # ---------------- Slots (Qt thread) ----------------
    @QtCore.pyqtSlot(str)
    def _slot_set_status(self, text):
        self._status_text = text or ""
        # minimal repaint (only bottom area) would be better, but call update() for simplicity
        self.update()

    @QtCore.pyqtSlot(str)
    def _slot_set_mood(self, mood):
        if mood in self.mood_colors:
            self._mood = mood
        else:
            self._mood = "neutral"
        self.update()

    @QtCore.pyqtSlot(float)
    def _slot_react_audio(self, intensity):
        # clamp and smooth reactive boost
        intensity = float(intensity or 0.0)
        target = max(0.3, min(1.5, intensity))
        # immediate uplift (keeps natural feel) but avoid sudden huge jumps
        self._reactive_boost = max(self._reactive_boost, target)
        # store mic intensity separately for possible visual mapping
        self._mic_intensity = min(max(intensity, 0.0), 1.0)
        self.update()

    # ---------------- Mic Listener (background) ----------------
    def _mic_audio_callback(self, indata, frames, time_info, status):
        """
        This callback runs in the sounddevice thread. Keep it minimal:
        - Compute a small energy value and emit a Qt signal to update UI.
        """
        try:
            # compute RMS-ish energy scaled to small range
            vol = np.linalg.norm(indata) / (frames**0.5 + 1e-9)
            # scale appropriately; keep within 0..1
            scaled = min(max(vol * 10.0, 0.0), 1.0)
            # Emit to Qt thread for safe UI update
            self.sig_react_audio.emit(float(scaled))
        except Exception:
            # ignore errors inside callback (must not raise)
            pass

    def _start_mic_stream(self):
        """Start the sounddevice InputStream in a dedicated thread and handle exceptions gracefully."""
        if not self._mic_enabled:
            return

        def _mic_worker():
            try:
                # blocksize set to match earlier code (1024) and rate 16000
                with sd.InputStream(callback=self._mic_audio_callback, channels=1, samplerate=16000, blocksize=1024):
                    # keep the context alive until stopped
                    while self.running and self._mic_enabled:
                        time.sleep(0.1)
            except Exception as e:
                # If device busy or other PortAudio issues happen, disable mic listener to avoid crashes
                print(f"⚠️ Mic listener failed to start or run: {e}")
                self._mic_enabled = False
                # ensure UI knows mic is inactive
                self.sig_react_audio.emit(0.0)

        # spawn background thread for the InputStream to avoid blocking Qt main thread
        self._mic_thread = threading.Thread(target=_mic_worker, daemon=True, name="OverlayMicThread")
        self._mic_thread.start()

    # ---------------- Animation tick (Qt thread) ----------------
    def _on_anim_tick(self):
        # update pulse and decay reactive boost
        self._pulse_angle = (self._pulse_angle + 3.0) % 360.0
        # decay reactive boost smoothly (multiplicative decay)
        self._reactive_boost = max(self._reactive_boost * 0.92, 0.0)
        # trigger repaint
        self.update()
# core/interface.py — UPGRADED PART 2 (continuation)

    # ---------------- Fade-in ----------------
    def _start_fade_in(self, duration_ms=600):
        # Use QPropertyAnimation to animate windowOpacity safely on the Qt thread
        self._fade_anim.stop()
        self._fade_anim.setDuration(int(duration_ms))
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    # ---------------- Paint ----------------
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center = QtCore.QPointF(w / 2.0, h / 2.0)

        # mood colors (cached locally)
        colors = self.mood_colors.get(self._mood, self.mood_colors["neutral"])
        col_inner, col_outer = colors

        # compute animation parameters
        glow = (math.sin(math.radians(self._pulse_angle)) + 1.0) * 0.5 * (1.0 + self._reactive_boost)
        outer_radius = 70.0 + 25.0 * self._reactive_boost

        # radial gradient
        gradient = QtGui.QRadialGradient(center, outer_radius)
        inner_alpha = int(200 * (0.6 + glow * 0.4))
        mid_alpha = int(120 * (0.6 + glow * 0.4))
        gradient.setColorAt(0.0, QtGui.QColor(col_inner.red(), col_inner.green(), col_inner.blue(), max(0, min(255, inner_alpha))))
        gradient.setColorAt(0.6, QtGui.QColor(col_outer.red(), col_outer.green(), col_outer.blue(), max(0, min(255, mid_alpha))))
        gradient.setColorAt(1.0, QtCore.Qt.transparent)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawEllipse(center, outer_radius, outer_radius)

        # concentric rings
        for i, scale in enumerate([0.85, 0.65, 0.45, 0.25]):
            alpha = int(80 * (1 + 0.5 * self._reactive_boost) * (1 - i * 0.15))
            pen = QtGui.QPen(QtGui.QColor(col_outer.red(), col_outer.green(), col_outer.blue(), max(0, min(255, alpha))), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(center, outer_radius * scale, outer_radius * scale)

        # core circle
        core_radius = 35.0 + 8.0 * self._reactive_boost
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(col_inner)
        painter.drawEllipse(center, core_radius, core_radius)

        # status text
        if self._status_text:
            painter.setPen(QtGui.QColor(255, 255, 255, 220))
            painter.setFont(self._cached_font)
            rect = self.rect()
            painter.drawText(rect, QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, self._status_text)

    # ---------------- Run ----------------
    def run(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication instance missing. Create it before calling overlay.run().")

        try:
            screen = app.primaryScreen()
            geometry = screen.availableGeometry()

            # FIXED POSITIONING — PERFECT FOR ALL SCALINGS
            screen_center_x = geometry.x() + int((geometry.width() - self.width()) / 2)
            webcam_offset_y = geometry.y() + int(geometry.height() * 0.02)  # 2% from top

            self.move(screen_center_x, webcam_offset_y)

            print(f"📍 Overlay fixed at: X={screen_center_x}, Y={webcam_offset_y} (exact under webcam)")

        except Exception as e:
            print(f"⚠️ Overlay positioning failed: {e}")

        # start animation timer and mic stream (if available)
        self._anim_timer.start()
        # try to start mic stream — if portaudio busy, it will gracefully disable itself
        try:
            if self._mic_enabled:
                self._start_mic_stream()
        except Exception as e:
            print("⚠️ _start_mic_stream error:", e)
            self._mic_enabled = False

        # fade in smoothly
        self._start_fade_in()
        self.show()

    def stop(self):
        # Gracefully stop everything
        print("🛑 Siri-style Overlay stopping...")
        self.running = False

        # stop timers & animations on Qt thread
        try:
            self._anim_timer.stop()
        except Exception:
            pass

        try:
            self._fade_anim.stop()
        except Exception:
            pass

        # stop mic stream worker
        self._mic_enabled = False
        # allow mic thread to exit; join briefly if alive (non-blocking overall)
        try:
            if self._mic_thread and self._mic_thread.is_alive():
                self._mic_thread.join(timeout=0.3)
        except Exception:
            pass

        # close widget
        try:
            self.close()
        except Exception:
            pass

        print("🛑 Siri-style Overlay stopped.")


##############################################################################################################
# FILE: Jarvis\core\listener.py
##############################################################################################################

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


##############################################################################################################
# FILE: Jarvis\core\logger.py
##############################################################################################################

import logging

logging.basicConfig(
    filename="jarvis_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("JARVIS")


##############################################################################################################
# FILE: Jarvis\core\memory_engine.py
##############################################################################################################

# core/memory_engine.py
import json
import os
import tempfile
import random
import time
from core.speech_engine import speak

# Module-level singleton holder
_INSTANCE = None


class JarvisMemory:
    """Stores Jarvis’s emotional context, facts, and conversational memory.

    This class uses a singleton pattern (via __new__) so repeated calls to
    JarvisMemory() across modules return the same shared instance. This
    prevents repeated initialization prints and duplicated loads.
    """
    listener_hook = None

    def register_listener_hook(fn):
       global listener_hook
       listener_hook = fn

    def __new__(cls, *args, **kwargs):
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = super(JarvisMemory, cls).__new__(cls)
        return _INSTANCE

    def __init__(self):
        # Avoid re-running __init__ on the singleton after first creation
        if getattr(self, "_initialized", False):
            return

        # Resolve config path reliably (project root relative)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cfg_dir = os.path.join(base_dir, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        self.file_path = os.path.join(cfg_dir, "memory.json")

        # Default structure
        self.memory = {
            "facts": {},
            "mood": "neutral",
            "last_topic": None,
            "emotion_history": []
        }

        self._load_memory()
        self._validate_structure()
        # runtime-only (not persisted)
        self.last_action = None

        # mark initialized (prevents repeated prints)
        self._initialized = True
        print("🧠 Memory Engine Initialized")   

    # -------------------------------------------------------
    def _validate_structure(self):
        """
        Ensures required keys always exist and caps history length.
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

        if "emotion_history" not in self.memory:
            self.memory["emotion_history"] = []
            changed = True

        # limit emotion history to a sensible size to avoid huge files
        if isinstance(self.memory.get("emotion_history"), list):
            self.memory["emotion_history"] = self.memory["emotion_history"][-200:]  # keep last 200
            changed = True

        if changed:
            self._save_memory()

    # -------------------- LOAD / SAVE --------------------
    def _load_memory(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
        except Exception:
            # on any error, reset to defaults (but do not raise)
            self.memory = {
                "facts": {},
                "mood": "neutral",
                "last_topic": None,
                "emotion_history": []
            }

    def _save_memory(self):
        """Atomic save to avoid corrupting the file if interrupted."""
        try:
            dirpath = os.path.dirname(self.file_path)
            os.makedirs(dirpath, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.memory, f, indent=2, ensure_ascii=False)
                # atomic replace
                os.replace(tmp, self.file_path)
            finally:
                # if tmp still exists, try removing
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except:
                        pass
        except Exception:
            # best-effort save; ignore errors to avoid crashing Jarvis
            pass

    # -------------------- FACT MEMORY --------------------
    def remember_fact(self, key, value):
        if not key:
            speak("I need a key to remember that.", mood="alert")
            return
        try:
            self.memory.setdefault("facts", {})[key.lower()] = value
            self._save_memory()
            speak(f"Got it, Yash. I'll remember that {key} is {value}.", mood="happy")
        except Exception:
            speak("Couldn't save that right now.", mood="alert")

    def recall_fact(self, key):
        if not key:
            return None
        return self.memory.get("facts", {}).get(key.lower())

    def forget_fact(self, key):
        if not key:
            speak("Tell me what to forget.", mood="alert")
            return
        k = key.lower()
        if k in self.memory.get("facts", {}):
            try:
                del self.memory["facts"][k]
                self._save_memory()
                speak(f"Alright, I’ll forget about {key}.", mood="serious")
            except Exception:
                speak("Couldn't forget that right now.", mood="alert")
        else:
            speak(f"I don’t think you ever told me about {key}.", mood="alert")

    # -------------------- MOOD SYSTEM --------------------
    def set_mood(self, mood):
        """Stores the Jarvis internal mood and persists it."""
        try:
            if not mood:
                mood = "neutral"
            self.memory["mood"] = mood
            self._save_memory()
        except Exception:
            pass

    def get_mood(self):
        return self.memory.get("mood", "neutral")

    def emotional_response(self, mood):
        """Responds based on current emotional state (helper function)."""
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
        try:
            speak(random.choice(responses.get(mood, ["I’m feeling neutral right now."])), mood=mood)
        except Exception:
            pass

    # -------------------- EMOTION HISTORY --------------------
    def add_emotion_history(self, mood):
        """Append mood to emotion_history safely (keeps last 100)."""
        try:
            if mood not in ["happy", "serious", "neutral", "alert"]:
                mood = "neutral"
            entry = {
                "mood": mood,
                "time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            hist = self.memory.get("emotion_history", [])
            hist.append(entry)
            # keep last 100 entries
            self.memory["emotion_history"] = hist[-100:]
            self._save_memory()
        except Exception:
            pass

    # -------------------- TOPIC MEMORY --------------------
    def update_topic(self, topic):
        """Remember last topic user talked about (used for context)."""
        try:
            self.memory["last_topic"] = topic
            self._save_memory()
        except Exception:
            pass

    def get_last_topic(self):
        return self.memory.get("last_topic", None)

    # -------------------- ACTION OUTCOME MEMORY --------------------
    def record_action(self, intent, command, status, reason=None):
        """
        Store the outcome of the last executed action.
        Runtime-only (not persisted to disk).
        """
        try:
            self.last_action = {
                "intent": intent,
                "command": command,
                "status": status,
                "reason": reason,
                "timestamp": time.time()
            }
        except Exception:
            pass

    def get_last_action(self):
        """Return last recorded action outcome."""
        return self.last_action


    # -------------------- COMPAT HELPERS --------------------
    def update_mood_from_text(self, text):
        """
        Backwards-compatible helper used elsewhere.
        Simple text mood estimation (keeps this lightweight).
        """
        try:
            if not text:
                return
            t = text.lower()
            if any(w in t for w in ["sad", "low", "depressed", "hurt", "broken"]):
                self.set_mood("serious")
                self.add_emotion_history("serious")
            elif any(w in t for w in ["happy", "great", "awesome", "good", "nice"]):
                self.set_mood("happy")
                self.add_emotion_history("happy")
            elif any(w in t for w in ["angry", "mad", "hate", "furious"]):
                self.set_mood("alert")
                self.add_emotion_history("alert")
            else:
                # keep neutral for weak signals
                self.set_mood("neutral")
                self.add_emotion_history("neutral")
        except Exception:
            pass


# Expose a shared memory instance for older imports that expect object creation
memory = JarvisMemory()


##############################################################################################################
# FILE: Jarvis\core\music_player.py
##############################################################################################################

# core/music_player.py
"""
High-tech local music player for Jarvis.

Features:
- Play / pause / resume / stop.
- Playlist support and simple next/prev.
- Crossfade (soft) using pygame mixer fadeout.
- Integrates with overlay_instance for UI react.
- Emits short TTS confirmations.
"""

import os
import time
import threading
import pygame
from typing import Optional, List
from core.speech_engine import speak
from core.voice_effects import overlay_instance

# init mixer safely
try:
    pygame.mixer.init()
except Exception:
    try:
        pygame.mixer.quit()
        pygame.mixer.init()
    except:
        pass

class MusicPlayer:
    def __init__(self):
        self.playlist: List[str] = []
        self.index = 0
        self.lock = threading.Lock()
        self._paused = False

    def load(self, paths: List[str]):
        self.playlist = [p for p in paths if os.path.exists(p)]
        self.index = 0

    def play(self, path: Optional[str] = None):
        with self.lock:
            if path:
                if not os.path.exists(path):
                    speak("I couldn't find that song.", mood="alert")
                    return
                self.playlist = [path]
                self.index = 0
            if not self.playlist:
                speak("Your playlist is empty.", mood="alert")
                return
            current = self.playlist[self.index]
            try:
                pygame.mixer.music.load(current)
                pygame.mixer.music.play()
                self._paused = False
                speak(f"Playing {os.path.basename(current)}", mood="happy")
                if overlay_instance:
                    overlay_instance.react_to_audio(1.0)
            except Exception as e:
                print("Music play error:", e)
                speak("Couldn't play the track.", mood="alert")

    def pause(self):
        try:
            pygame.mixer.music.pause()
            self._paused = True
            speak("Paused.", mood="neutral")
        except Exception:
            pass

    def resume(self):
        try:
            pygame.mixer.music.unpause()
            self._paused = False
            speak("Resuming.", mood="happy")
        except Exception:
            pass

    def stop(self, fade_ms: int = 300):
        try:
            pygame.mixer.music.fadeout(fade_ms)
            time.sleep(fade_ms / 1000.0)
            self._paused = False
            speak("Stopped playback.", mood="neutral")
        except Exception:
            pass

    def next(self):
        with self.lock:
            if not self.playlist:
                return
            self.index = (self.index + 1) % len(self.playlist)
            self.play()

    def previous(self):
        with self.lock:
            if not self.playlist:
                return
            self.index = (self.index - 1) % len(self.playlist)
            self.play()

    def set_volume(self, v: float):
        try:
            v = max(0.0, min(1.0, v))
            pygame.mixer.music.set_volume(v)
            speak(f"Volume set to {int(v*100)} percent.", mood="neutral")
        except Exception:
            pass


# singleton
music_player = MusicPlayer()


##############################################################################################################
# FILE: Jarvis\core\music_stream.py
##############################################################################################################

# core/music_stream.py
"""
Music streaming helper for Jarvis.

Design:
- Lightweight default: open YouTube search in browser.
- If yt-dlp + ffmpeg available, can stream audio directly (experimental).
- Uses webbrowser for guaranteed cross-platform behavior.
"""

import os
import webbrowser
import threading
from core.speech_engine import speak
from core.voice_effects import overlay_instance

# try yt-dlp streaming (optional)
try:
    import yt_dlp
    _YTDLP = True
except Exception:
    _YTDLP = False

class MusicStream:
    def __init__(self):
        self._thread = None

    def play(self, query: str, open_in_browser: bool = True):
        if not query:
            speak("Which song should I play?", mood="neutral")
            return

        speak(f"Searching YouTube for {query}.", mood="happy")

        if open_in_browser or not _YTDLP:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            webbrowser.open_new_tab(url)
            return

        # else try yt-dlp direct stream (best-effort)
        def _stream_worker(q):
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'noplaylist': True,
                    'quiet': True,
                    'no_warnings': True,
                    'outtmpl': '-',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"ytsearch:{q}", download=False)['entries'][0]
                    url = info['webpage_url']
                    webbrowser.open_new_tab(url)
            except Exception as e:
                print("yt-dlp stream failed:", e)
                webbrowser.open_new_tab(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")

        t = threading.Thread(target=_stream_worker, args=(query,), daemon=True)
        t.start()
        self._thread = t

    def play_direct(self, url: str):
        if not url:
            speak("Give me a stream URL.", mood="neutral")
            return
        speak("Opening music stream.", mood="happy")
        webbrowser.open_new_tab(url)


# singleton
music_stream = MusicStream()


##############################################################################################################
# FILE: Jarvis\core\nlp_engine.py
##############################################################################################################

# core/nlp_engine.py — Upgraded Hybrid NLP Engine (SAFE & Compatible)
"""
This version is fully backward-compatible with all your existing files.
✓ Zero breaking changes
✓ Faster learning
✓ Safer Markov chain
✓ Cleaner history handling
✓ Higher-quality wake/ack lines
"""

import os
import random
import threading

# ------------------------------------------------------------
# HISTORY LOAD
# ------------------------------------------------------------
HISTORY_PATH = os.path.join("config", "nlp_history.txt")
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

try:
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        _HISTORY = [line.strip() for line in f.readlines() if line.strip()]
except:
    _HISTORY = []

# ------------------------------------------------------------
# BUILD MARKOV (safer version)
# ------------------------------------------------------------
def _build_markov(history):
    M = {}
    for line in history:
        words = line.split()
        if len(words) < 2:
            continue
        for i in range(len(words) - 1):
            a = words[i].lower()
            b = words[i + 1].lower()
            if a.isalpha() and b.isalpha():
                M.setdefault(a, []).append(b)
    return M

_MARKOV = _build_markov(_HISTORY)
_LOCK = threading.Lock()

# ------------------------------------------------------------
# LEARN (safe, fast)
# ------------------------------------------------------------
def learn(phrase: str):
    """Safely append phrase & update Markov chain."""
    if not phrase or not phrase.strip():
        return

    phrase = phrase.strip()

    with _LOCK:
        try:
            # Write safely to history file
            with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(phrase + "\n")

            _HISTORY.append(phrase)

            # Update Markov quickly
            words = phrase.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    a = words[i].lower()
                    b = words[i + 1].lower()
                    if a.isalpha() and b.isalpha():
                        _MARKOV.setdefault(a, []).append(b)

        except Exception:
            pass

# async wrapper
def learn_async(phrase):
    threading.Thread(target=learn, args=(phrase,), daemon=True).start()


# ------------------------------------------------------------
# MARKOV GENERATOR (cleaned & safer)
# ------------------------------------------------------------
def _markov_generate(seed_word=None, length=8):
    """Generate short natural text — guaranteed no crashes."""
    if not _MARKOV:
        return None

    # Seed selection
    seed = seed_word.lower() if seed_word else random.choice(list(_MARKOV.keys()))
    if seed not in _MARKOV:
        seed = random.choice(list(_MARKOV.keys()))

    out = [seed.capitalize()]
    cur = seed

    for _ in range(length - 1):
        options = _MARKOV.get(cur)
        if not options:
            break
        nxt = random.choice(options)
        out.append(nxt)
        cur = nxt

    sentence = " ".join(out)
    # Make it cleaner (avoid trailing bad tokens)
    return sentence.strip().rstrip(",. ")


# ------------------------------------------------------------
# TEMPLATES (improved naturalness)
# ------------------------------------------------------------
_WAKE_TEMPLATES = [
    "Good {time_of_day}, Yash. I'm active and listening.",
    "Systems up — what’s our first task today?",
    "Fully online. How can I assist you?"
]

_LIMITED_TEMPLATES = [
    "Face not verified — limited mode enabled. You can still give searches and local commands.",
    "Limited mode active, but I'm still here for essential tasks."
]

_ACK_TEMPLATES = [
    "Done. What next?",
    "Finished — anything else?",
    "All set, Yash."
]

# ------------------------------------------------------------
# PUBLIC GENERATORS
# ------------------------------------------------------------
def generate_greeting(mood="neutral"):
    base = random.choice(_WAKE_TEMPLATES)
    extra = _markov_generate(length=6) or ""
    return f"{base} {extra}".strip()

def generate_wakeup(mood="neutral"):
    variants = [
        "Aye aye, I'm awake.",
        "Yes Yash, I'm right here.",
        "Boot complete — listening."
    ]
    extra = _markov_generate(length=6) or ""
    return random.choice(variants) + (" " + extra if extra else "")

def generate_limited_mode_line(mood="neutral"):
    return random.choice(_LIMITED_TEMPLATES)

def generate_ack(mood="neutral"):
    return random.choice(_ACK_TEMPLATES)

# ------------------------------------------------------------
# Backward compatibility (do NOT remove)
# ------------------------------------------------------------
learn = learn


##############################################################################################################
# FILE: Jarvis\core\runtime.py
##############################################################################################################

# core/runtime.py
import os
import time
import threading
import random
import tempfile

from core import voice_effects  # overlay attach helper
import core.state as state
import core.sleep_manager as sleep_manager  # manage sleep/wake with overlay

# DeepFace toggle — disabled to avoid TensorFlow DLL errors on this system
USE_DEEPFACE = False
try:
    if USE_DEEPFACE:
        from deepface import DeepFace
    else:
        DeepFace = None
except Exception as e:
    print("⚠️ DeepFace not available on this system, using fallback-only face auth.")
    DeepFace = None
    USE_DEEPFACE = False


# ======================================================================
#  FACE AUTHENTICATION MODULE
# ======================================================================
class FaceAuth:
    """
    Stable & cinematic face verification using fallback OpenCV histogram.
    DeepFace is optional and safely handled.
    """

    def __init__(self):
        self.reference_path = os.path.join("config", "face_data", "yash_reference.jpg")
        os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
        print("📸 FaceAuth loaded")

    # ------------------------------------------------------
    def capture_reference(self):
        import cv2
        from core.speech_engine import speak

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera not accessible, Yash.", mood="alert")
            return

        speak("Look at the camera. Capturing your reference image.", mood="serious")
        time.sleep(1.2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            speak("Failed to capture your face clearly.", mood="alert")
            return

        cv2.imwrite(self.reference_path, frame)
        print("✅ Reference saved:", self.reference_path)
        speak("Reference image captured successfully.", mood="happy")

    # ------------------------------------------------------
    def _fallback_compare(self, ref_path, img2_path):
        """OpenCV histogram fallback."""
        import cv2
        try:
            ref = cv2.imread(ref_path)
            img = cv2.imread(img2_path)
            if ref is None or img is None:
                return False

            ref = cv2.resize(ref, (224, 224))
            img = cv2.resize(img, (224, 224))

            ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h1 = cv2.calcHist([ref_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            h2 = cv2.calcHist([img_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)

            score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            return score >= 0.55

        except Exception as e:
            print("⚠️ Fallback compare error:", e)
            return False

    # ------------------------------------------------------
    def verify_user(self):
        import cv2
        from core.speech_engine import speak, jarvis_fx

        # Ensure reference exists
        if not os.path.exists(self.reference_path):
            speak("No reference image found. Creating one.", mood="alert")
            self.capture_reference()
            if not os.path.exists(self.reference_path):
                return False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera not accessible for verification.", mood="alert")
            return False

        speak("Verifying your identity. Keep looking at the camera.", mood="serious")

        # Ambient sound ON
        try:
            threading.Thread(target=jarvis_fx.play_ambient, daemon=True).start()
        except:
            pass

        # Overlay scanning animation
        if getattr(voice_effects, "overlay_instance", None):
            try:
                voice_effects.overlay_instance.set_status("🔍 Scanning your face…")
                voice_effects.overlay_instance.set_mood("neutral")
            except:
                pass

        def scan_anim():
            if not getattr(voice_effects, "overlay_instance", None):
                return
            for _ in range(8):
                try:
                    voice_effects.overlay_instance.react_to_audio(1.0)
                    time.sleep(0.22)
                    voice_effects.overlay_instance.react_to_audio(0.2)
                    time.sleep(0.22)
                except:
                    break

        threading.Thread(target=scan_anim, daemon=True).start()

        time.sleep(1.4)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            speak("Couldn't capture a clear image.", mood="alert")
            try:
                jarvis_fx.fade_out_ambient(800)
            except:
                pass
            return False

        # Save temp scan image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            scan_path = tmp.name
        import cv2
        cv2.imwrite(scan_path, frame)

                # Try DeepFace (if enabled), otherwise fallback directly
        verified = False
        if DeepFace is not None and USE_DEEPFACE:
            try:
                result = DeepFace.verify(
                    img1_path=self.reference_path,
                    img2_path=scan_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                verified = bool(result.get("verified", False))
            except Exception as e:
                print("⚠️ DeepFace verify failed — using fallback instead:", e)
                verified = self._fallback_compare(self.reference_path, scan_path)
        else:
            # DeepFace disabled or not available → always use fallback
            verified = self._fallback_compare(self.reference_path, scan_path)


        # Remove temp
        try:
            os.remove(scan_path)
        except:
            pass

        # Stop ambient
        try:
            jarvis_fx.fade_out_ambient(800)
        except:
            pass

        # Return result
        return verified


# ======================================================================
#   JARVIS STARTUP FLOW
# ======================================================================
def _time_greeting():
    import datetime
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 22:
        return "Good evening"
    return "Hello"


def jarvis_startup(overlay):
    print("\n🤖 Booting Yash’s JARVIS…\n")
    import core.state as state
    from core.speech_engine import speak, jarvis_fx
    from core.memory_engine import JarvisMemory
    from core.command_handler import JarvisCommandHandler
    from core.listener import JarvisListener
    from core.context import memory


    handler = JarvisCommandHandler()

    # Link overlay to effects (preferred API: attach_overlay)
    try:
        if hasattr(voice_effects, "attach_overlay"):
            voice_effects.attach_overlay(overlay)
        else:
            # fallback: set attribute directly
            voice_effects.overlay_instance = overlay
        print("🌀 Overlay attached.")
    except Exception as e:
        print("⚠️ Overlay attach failed:", e)

    # Start sleep manager with overlay so it can dim/wake the UI
    try:
        sleep_manager.start_manager(overlay)
    except Exception:
        try:
            # fallback: start without overlay
            sleep_manager.start_manager()
        except Exception as e:
            print("⚠️ Sleep manager failed to start:", e)

    try:
        overlay.set_status("Booting systems…")
    except:
        pass

    # Boot animation
    for _ in range(3):
        try:
            overlay.react_to_audio(1.2)
            time.sleep(0.28)
            overlay.react_to_audio(0.2)
            time.sleep(0.28)
        except:
            time.sleep(0.5)

    # Startup sound
    try:
        jarvis_fx.play_startup()
    except Exception as e:
        print("⚠️ Startup sound:", e)
    time.sleep(5)

    speak("System booting up. Initializing cognition and neural modules.", mute_ambient=True)
    time.sleep(0.6)

    # ----------------------- FACE VERIFICATION ------------------------
    face = FaceAuth()
    verified = face.verify_user()

    # Sync to global state
    state.FACE_VERIFIED = bool(verified)

    # SUCCESS / FAILURE SOUNDS
    from core.speech_engine import speak, jarvis_fx

    if verified:
        try:
            jarvis_fx.play_success()
        except:
            pass
        try:
            overlay.set_status("Identity verified ✅")
            overlay.set_mood("happy")
            overlay.react_to_audio(1.3)
        except:
            pass
        speak("Identity verified. Welcome back, Yash.", mood="happy", mute_ambient=True)
    else:
        try:
            jarvis_fx.play_alert()
        except:
            pass
        try:
            overlay.set_status("Identity not recognized ❌")
            overlay.set_mood("alert")
            overlay.react_to_audio(0.6)
        except:
            pass
        speak("I couldn't recognize you. Limited mode enabled.", mood="alert", mute_ambient=True)

    # ----------------------- GREETING ------------------------
    greet = _time_greeting()
    mood = memory.get_mood()

    speak(f"{greet}, Yash.", mute_ambient=True)
    time.sleep(0.3)
    speak("Say 'Hey Jarvis' when you're ready.", mute_ambient=True)

    # ----------------------- LISTENER ------------------------
    try:
        overlay.set_status("Listening…")
    except:
        pass

    # Boot finished — allow sleep manager to operate normally
    sleep_manager.SYSTEM_BOOTING = False


    # Initialize LAST_INTERACTION so sleep manager has a baseline
    try:
        state.LAST_INTERACTION = time.time()
    except:
        pass

    print("\n🎤 Listener online — say: Hey Jarvis\n")
    try:
        # instantiate listener (it starts its own continuous thread)
        JarvisListener()
    except Exception as e:
        print("⚠️ Listener failed:", e)

    # main thread keeps running to keep Qt app alive and threads working
    while True:
        time.sleep(6)


##############################################################################################################
# FILE: Jarvis\core\sleep_manager.py
##############################################################################################################

# core/sleep_manager.py

import threading
import time
import random
import core.state as state
from core.speech_engine import speak
import core.voice_effects as fx
from core.brain import brain
SYSTEM_BOOTING = True


# Optional emotion module
try:
    from core.face_emotion import FaceEmotionAnalyzer
except:
    FaceEmotionAnalyzer = None


# ---------------------------------------------------------
# FRIENDLY SLEEP LINES
# ---------------------------------------------------------
SLEEP_LINES = [
    "I’m feeling a little tired… going to sleep now, Yash. Call me if you need me.",
    "Resting for a bit. Just say ‘Hey Jarvis’ to wake me.",
    "Going into sleep mode. I’ll be right here.",
    "Powering down softly… wake me anytime.",
]

# two-minute timeout
SLEEP_TIMEOUT = 120  
def _do_sleep_procedure(overlay=None):

    if state.MODE == "sleep":
        return
    
    state.MODE = "sleep"
    print("💤 Jarvis entering sleep mode...")

    # Soft friendly line
    try:
        speak(random.choice(SLEEP_LINES), mood="neutral", mute_ambient=True)
    except:
        pass

    # Fade out ambient audio
    try:
        fx.jarvis_fx.fade_out_ambient(800)
    except:
        pass

    # UI dim
    try:
        if overlay:
            overlay.set_status("Sleeping…")
            overlay.set_mood("neutral")
            overlay.setWindowOpacity(0.35)
    except:
        pass

    # Stop SFX
    try:
        fx.jarvis_fx.stop_all()
    except:
        pass
def _do_wake_procedure(overlay=None):
    print("⚡ Jarvis waking up...")
    state.MODE = "wake_transition"

    # wake chime
    try:
        fx.jarvis_fx.play_success()
        fx.jarvis_fx.play_ambient()
    except:
        pass

    time.sleep(0.2)

    # Optional face-emotion
    face_mood = None
    if FaceEmotionAnalyzer is not None:
        try:
            fe = FaceEmotionAnalyzer()
            face_mood = fe.capture_emotion()
        except:
            face_mood = None

    # Fuse emotions
    try:
        mood = brain.fuse_emotions(face=face_mood)
    except:
        mood = "neutral"

    state.JARVIS_MOOD = mood

    # Cinematic wake line
    try:
        line = brain.generate_wakeup_line(
            mood=mood,
            last_topic=state.LAST_TOPIC
        )
        speak(line, mood=mood, mute_ambient=True)
    except:
        speak("I'm awake now, Yash.", mood="neutral")

    # Restore UI brightness
    try:
        if overlay:
            overlay.setWindowOpacity(1.0)
            overlay.set_status("Ready")
            overlay.set_mood(mood)
    except:
        pass

    state.MODE = "active"
    state.LAST_INTERACTION = time.time()
class SleepManager:
    def __init__(self):
        self.running = False
        self.overlay = None     # UI reference

    def attach_overlay(self, overlay):
        self.overlay = overlay

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                last = state.LAST_INTERACTION
                mode = state.MODE

                if last is None:  # no interaction yet
                    time.sleep(1)
                    continue

                if mode == "sleep":
                    time.sleep(1)
                    continue

                idle = time.time() - last

                if not SYSTEM_BOOTING and idle >= SLEEP_TIMEOUT:
                 _do_sleep_procedure(self.overlay)


                time.sleep(1)

            except:
                time.sleep(1)


manager = SleepManager()


def start_manager(overlay=None):
    try:
        if overlay:
            manager.attach_overlay(overlay)
        manager.start()
    except:
        pass


##############################################################################################################
# FILE: Jarvis\core\speech_engine.py
##############################################################################################################

# core/speech_engine.py — PART 1/4
import os
import asyncio
import tempfile
import threading
import time
import traceback

# audio libs
import pygame
import pyttsx3
edge_tts = None

# optional neural TTS (Edge)
try:
    import edge_tts
except Exception:
    edge_tts = None

# local modules (effects + state)
from core.voice_effects import JarvisEffects
import core.voice_effects as fx
import core.state as state     # mic-mute integration

# singleton effects instance (best-effort)
try:
    jarvis_fx = JarvisEffects()
except Exception:
    jarvis_fx = None

# Listener hook (set by listener.register_listener_hook)
LISTENER_HOOK = None


def register_listener_hook(fn):
    """
    Listener should call register_listener_hook(self.set_speaking)
    so speech engine can mute/unmute mic properly.
    """
    global LISTENER_HOOK
    LISTENER_HOOK = fn
# core/speech_engine.py — PART 2/4
# Stable mixer initialization with dedicated channels to avoid contention.

class StableMixer:
    VOICE = 0
    SFX = 1
    AMBIENT = 2

    @staticmethod
    def init():
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        # Init with safe buffer; smaller buffer reduces latency on playback
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(8)
        StableMixer.voice = pygame.mixer.Channel(StableMixer.VOICE)
        StableMixer.sfx = pygame.mixer.Channel(StableMixer.SFX)
        StableMixer.ambient = pygame.mixer.Channel(StableMixer.AMBIENT)
        print("🔊 StableMixer initialized (3-channel mode)")


# initialize mixer at import
try:
    StableMixer.init()
except Exception as e:
    print("⚠️ Mixer init failed:", e)


# ---------------- TTS engine ----------------
class JarvisVoice:
    """
    Combines Edge-TTS (async, high quality) + pyttsx3 fallback.
    Edge-TTS playback runs in a dedicated thread/event-loop so speak() returns once playback completes
    but without blocking the main thread or causing mic context issues.
    """
    def __init__(self):
        # offline fallback engine (pyttsx3)
        try:
            self.offline_engine = pyttsx3.init()
            self.offline_engine.setProperty("rate", 175)
            self.offline_engine.setProperty("volume", 1.0)
            self.offline_engine.setProperty("voice", self._select_voice("male"))
        except Exception as e:
            print("⚠️ pyttsx3 init failed:", e)
            self.offline_engine = None

        self.online_enabled = edge_tts is not None
        self._lock = threading.RLock()
        self._play_thread = None
        self._stop_flag = threading.Event()
        print("🎧 JarvisVoice ready (edge_tts enabled: {})".format(bool(self.online_enabled)))

    def _select_voice(self, gender):
        try:
            voices = self.offline_engine.getProperty("voices")
            for v in voices:
                if gender.lower() in v.name.lower():
                    return v.id
            return voices[0].id
        except Exception:
            return None
# core/speech_engine.py — PART 3/4
    # ---------- Edge-TTS async playback (runs in separate thread) ----------
    def _edge_tts_worker(self, text):
        """
        Worker runs an asyncio loop inside a dedicated thread to create TTS file and play it.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._play_edge_tts_async(text))
        except Exception as e:
            print("⚠️ Edge-TTS worker error:", e)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def _play_edge_tts_async(self, text):
        tmp_path = None
        try:
            communicator = edge_tts.Communicate(text, "en-US-GuyNeural")
            # save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
            await communicator.save(tmp_path)

            # play via pygame mixer on voice channel
            try:
                StableMixer.voice.stop()
                sound = pygame.mixer.Sound(tmp_path)
                StableMixer.voice.play(sound)
            except Exception as e:
                print("⚠️ pygame play failed for Edge-TTS:", e)
                return False

            # overlay react if available
            try:
                if fx.overlay_instance:
                    fx.overlay_instance.react_to_audio(1.0)
            except:
                pass

            # block here until playback completes (non-blocking to main thread, since inside worker)
            while StableMixer.voice.get_busy() and not self._stop_flag.is_set():
                time.sleep(0.05)

            try:
                if fx.overlay_instance:
                    fx.overlay_instance.react_to_audio(0.2)
            except:
                pass

            return True
        except Exception as e:
            print("⚠️ Edge-TTS async error:", e)
            traceback.print_exc()
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ---------- offline TTS (pyttsx3) ----------
    def _speak_offline(self, text):
        try:
            if fx.overlay_instance:
                fx.overlay_instance.react_to_audio(1.0)
            if self.offline_engine:
                self.offline_engine.say(text)
                self.offline_engine.runAndWait()
            if fx.overlay_instance:
                fx.overlay_instance.react_to_audio(0.2)
            return True
        except Exception as e:
            print("⚠️ Offline TTS error:", e)
            return False
# core/speech_engine.py — PART 4/4
    # ---------- public speak entrypoint ----------
    def speak(self, text, allow_fallback=True):
        if not text or not text.strip():
            return

        # ensure only one speak at a time
        with self._lock:
            # stop any previous play
            self._stop_flag.clear()
            StableMixer.voice.stop()
            StableMixer.sfx.stop()

            # tell listener to stop listening (best-effort)
            try:
                state.SYSTEM_SPEAKING = True
            except:
                pass
            try:
                if LISTENER_HOOK:
                    LISTENER_HOOK(True)
            except Exception:
                pass

            # start playback in worker thread (so main thread isn't blocked)
            success = False
            try:
                if self.online_enabled:
                    # spawn worker and wait for it to finish (non-blocking main thread)
                    self._play_thread = threading.Thread(target=self._edge_tts_worker, args=(text,), daemon=True)
                    self._play_thread.start()
                    # Wait for thread to finish but release GIL occasionally — short join loops
                    while self._play_thread.is_alive():
                        time.sleep(0.05)
                    success = True  # if playback completed without raising we've succeeded (Edge playback logs errors)
                else:
                    success = self._speak_offline(text)

            except Exception as e:
                print("⚠️ speak() error (primary):", e)
                traceback.print_exc()
                success = False

            # fallback if needed
            if not success and allow_fallback:
                try:
                    self._speak_offline(text)
                except Exception:
                    pass

            # small cooldown
            time.sleep(0.05)

            # re-enable microphone
            try:
                state.SYSTEM_SPEAKING = False
            except:
                pass
            try:
                if LISTENER_HOOK:
                    LISTENER_HOOK(False)
            except Exception:
                pass


# global instance
jarvis_voice = JarvisVoice()

# ---------------- public speak wrapper (kept same API) ----------------
def speak(text, mood="neutral", mute_ambient=True):
    if not text or not text.strip():
        return
    try:
        # optionally mute ambient music
        if mute_ambient:
            try:
                StableMixer.ambient.stop()
            except:
                pass

        # mood audio react
        try:
            jarvis_fx.mood_tone(mood)
        except:
            pass

        if fx.overlay_instance:
            try:
                fx.overlay_instance.react_to_audio(0.8)
            except:
                pass

        # small lead-in for overlay
        time.sleep(0.10)

        # perform TTS (blocking until finished, but inside engine threads)
        jarvis_voice.speak(text)

        # after speech calm-down
        if fx.overlay_instance:
            try:
                fx.overlay_instance.react_to_audio(0.15)
            except:
                pass

    except Exception as e:
        print("⚠️ speak wrapper error:", e)
        traceback.print_exc()


# cinematic boot sequence (keeps same API)
def play_boot_sequence():
    try:
        if jarvis_fx:
            jarvis_fx.stop_all()
        try:
            StableMixer.sfx.stop()
        except:
            pass

        boot_path = os.path.join("assets", "audio", "startup_long.wav")
        if os.path.exists(boot_path):
            boot_sound = pygame.mixer.Sound(boot_path)
            StableMixer.sfx.play(boot_sound)
            # overlay react during boot
            try:
                if fx.overlay_instance:
                    for _ in range(6):
                        fx.overlay_instance.react_to_audio(1.3)
                        time.sleep(0.4)
                        fx.overlay_instance.react_to_audio(0.3)
                        time.sleep(0.3)
            except:
                pass
        else:
            print("⚠️ Boot sound missing:", boot_path)
    except Exception as e:
        print("⚠️ Boot sequence failed:", e)
        traceback.print_exc()


# If a module wants to know the engine default: expose jarvis_voice
__all__ = ["register_listener_hook", "speak", "play_boot_sequence", "jarvis_voice"]


##############################################################################################################
# FILE: Jarvis\core\state.py
##############################################################################################################

"""
Global runtime state for Jarvis.
Only variables — no logic, no functions.
Shared across all modules.
"""

# -------------------------------------------------------------
# FACE AUTH
# -------------------------------------------------------------
FACE_VERIFIED = False

# -------------------------------------------------------------
# OPERATION MODES
# -------------------------------------------------------------
#  "active"          → fully awake, listening normally
#  "sleep_wait"      → inactivity countdown running
#  "sleep"           → soft-sleep mode, only wake-word allowed
#  "wake_transition" → waking animation + dialog
#  "processing"      → busy executing command
MODE = "active"

# -------------------------------------------------------------
# LISTENING & SPEAKING FLAGS
# -------------------------------------------------------------
# Public flags used across listener, command handler, speech engine
SYSTEM_LISTENING = False        # microphone actively recording speech
SYSTEM_SPEAKING = False         # TTS speaking (listener should pause)

# Backward compatibility for older modules
LISTENING = False               # alias → avoid breaking older imports

# Wake-word availability
WAKE_WORD_ENABLED = True

# Duration (in seconds) before entering sleep mode
INACTIVITY_TIMEOUT = 120

# -------------------------------------------------------------
# EMOTION + CONTEXT
# -------------------------------------------------------------
USER_TONE = "neutral"       # user emotional tone detected by audio/text
JARVIS_MOOD = "neutral"     # internal mood used by brain & speech_engine
LAST_TOPIC = None           # used for topic continuation in brain

# Continuous conversation flag (Jarvis stays active)
CONVERSATION_ACTIVE = False

LAST_APP_CONTEXT = None
LAST_YOUTUBE_SEARCH = False
LAST_INTERACTION = 0

##############################################################################################################
# FILE: Jarvis\core\video_reader.py
##############################################################################################################

# core/video_reader.py
"""
Jarvis Ultra Video Reader & Summarizer (High Tech).

Capabilities:
- Extract audio from video robustly (moviepy / ffmpeg)
- Voice activity detection (VAD) to split into meaningful speech segments (uses webrtcvad if available)
- Transcribe segments using:
    - whisper (if installed)
    - openai whisper (if OPENAI_API_KEY available)
    - speech_recognition/google as fallback
- Optional visual OCR for slide detection (uses pytesseract + OpenCV if available)
- Chunk-aware summarization (uses same orchestrator as document_reader summarizer)
- Returns structured summary: Title / Key points / Timestamps
- Reads summary via core.speech_engine.speak
"""

import os
import tempfile
import threading
import time
from typing import Optional, List, Tuple

# moviepy for audio extraction
try:
    from moviepy import editor
    _MOVIEPY = True
except Exception:
    _MOVIEPY = False

# transcription backends
try:
    import whisper
    _WHISPER = True
except Exception:
    _WHISPER = False

try:
    import openai
    _OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
except Exception:
    _OPENAI = False

# fallback STT
import speech_recognition as sr

# optional VAD
try:
    import webrtcvad
    _VAD = True
except Exception:
    _VAD = False

# optional OCR for slides
try:
    import cv2
    import pytesseract
    _OCR = True
except Exception:
    _OCR = False

from core.speech_engine import speak
import core.nlp_engine as nlp
from core.document_reader import _summarize_text  # reuse summarizer
from core.voice_effects import overlay_instance
from core.memory_engine import JarvisMemory
from core.context import memory



# -------------------------
# Helpers: write audio
# -------------------------
def _extract_audio(video_path: str, target_sample_rate=16000) -> Optional[str]:
    if not _MOVIEPY:
        return None
    try:
        clip = editor.VideoFileClip(video_path)
        aud = clip.audio
        if aud is None:
            return None
        # write mono wav
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        aud.write_audiofile(tmp.name, fps=target_sample_rate, nbytes=2, codec="pcm_s16le")
        return tmp.name
    except Exception as e:
        print("⚠️ audio extract error:", e)
        return None


# -------------------------
# VAD segmenter (optional)
# -------------------------
def _vad_segment_wav(wav_path: str, aggressiveness: int = 2) -> List[Tuple[int, int]]:
    """
    Returns list of (start_ms, end_ms) speech ranges.
    If webrtcvad not available, return full file as single segment.
    """
    if not _VAD:
        return [(0, -1)]
    try:
        import wave
        vad = webrtcvad.Vad(aggressiveness)
        wf = wave.open(wav_path, "rb")
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        wf.close()
        # fallback: return single range
        # full implementation requires framing; to keep safe, fallback to single chunk
        return [(0, -1)]
    except Exception:
        return [(0, -1)]


# -------------------------
# Transcription dispatcher
# -------------------------
def _transcribe_with_whisper_local(wav_path: str) -> str:
    try:
        model = whisper.load_model("small")
        result = model.transcribe(wav_path, language="en")
        return result.get("text", "").strip()
    except Exception as e:
        print("⚠️ whisper local failed:", e)
        return ""

def _transcribe_with_openai(wav_path: str) -> str:
    try:
        with open(wav_path, "rb") as f:
            resp = openai.Audio.transcriptions.create(file=f, model="gpt-4o-transcribe" if hasattr(openai, "gpt4o") else "whisper-1")
            text = resp.get("text") or resp.get("transcript") or ""
            return text.strip()
    except Exception as e:
        print("⚠️ openai transcription failed:", e)
        return ""

def _transcribe_with_google(wav_path: str) -> str:
    try:
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            audio = r.record(src)
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        print("⚠️ google stt failed:", e)
        return ""


# -------------------------
# OCR slides (optional)
# -------------------------
def _extract_slide_texts(video_path: str, nth_frame: int = 60) -> List[str]:
    if not _OCR:
        return []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        texts = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % nth_frame == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # simple threshold
                try:
                    txt = pytesseract.image_to_string(gray)
                    if txt and len(txt.strip()) > 10:
                        texts.append(txt.strip())
                except Exception:
                    pass
            i += 1
        cap.release()
        return texts
    except Exception:
        return []


# -------------------------
# Core Summarizer pipeline
# -------------------------
class VideoReader:
    def __init__(self):
        self._thread = None

    def _transcribe(self, wav_path: str) -> str:
        # priority: whisper local -> openai -> google
        text = ""
        if _WHISPER:
            text = _transcribe_with_whisper_local(wav_path)
            if text:
                return text
        if _OPENAI:
            text = _transcribe_with_openai(wav_path)
            if text:
                return text
        # fallback google STT
        text = _transcribe_with_google(wav_path)
        return text or ""

    def summarize(self, video_path: str, prefer_summarizer: str = "auto", do_ocr: bool = True) -> Optional[str]:
        """
        Synchronous summarization. Returns final summary string or None on failure.
        For long videos, use summarize_async to avoid blocking.
        """
        if not os.path.exists(video_path):
            speak("I couldn't find that video.", mood="alert")
            return None

        speak("Processing video. This may take a bit...", mood="neutral")

        wav = _extract_audio(video_path)
        if not wav:
            speak("Couldn't extract audio from the video.", mood="alert")
            return None

        # optional OCR to enrich context
        slide_texts = []
        if do_ocr and _OCR:
            try:
                slide_texts = _extract_slide_texts(video_path)
            except Exception:
                slide_texts = []

        # transcribe whole audio
        transcript = self._transcribe(wav)
        # delete wav
        try:
            os.remove(wav)
        except:
            pass

        if not transcript:
            speak("I couldn't transcribe the audio reliably.", mood="alert")
            return None

        # enrich transcript with slide texts
        if slide_texts:
            transcript = "\n".join(slide_texts[:3]) + "\n\n" + transcript

        # chunk + summarize
        summary = _summarize_text(transcript, prefer=prefer_summarizer)

        # postprocess summary: create bullets if not too long
        bullets = []
        for line in summary.splitlines():
            if line.strip():
                bullets.append(line.strip())
        final = "\n".join(bullets[:8]) if bullets else summary

        # speak structured summary
        try:
            speak("Here is the video summary:", mood="happy")
            # break into smaller speak calls so TTS doesn't hit limits
            for part in final.split("\n\n"):
                speak(part.strip(), mood="neutral")
        except Exception:
            pass

        return final

    def summarize_async(self, video_path: str, prefer_summarizer: str = "auto", do_ocr: bool = True):
        t = threading.Thread(target=self.summarize, args=(video_path, prefer_summarizer, do_ocr), daemon=True)
        t.start()
        self._thread = t
        return t


# singleton
video_reader = VideoReader()


##############################################################################################################
# FILE: Jarvis\core\voice_effects.py
##############################################################################################################

# core/voice_effects.py
# ============================================================
#   JARVIS CINEMATIC SOUND ENGINE — STABLE & ENHANCED EDITION
# ============================================================
import os
import pygame
import threading
import time
import random
import traceback

# Ensure SDL uses a reasonable audio driver on Windows
os.environ.setdefault("SDL_AUDIODRIVER", "directsound")

# Overlay reference (InterfaceOverlay instance)
overlay_instance = None


def attach_overlay(overlay):
    """Attach the InterfaceOverlay instance safely."""
    global overlay_instance
    overlay_instance = overlay
    print("🌀 Overlay successfully linked with Jarvis voice system.")


class JarvisEffects:
    """Cinematic Jarvis sound system (smooth typewriter + ambient)."""

    CHANNEL_SFX = 1
    CHANNEL_AMBIENT = 2
    CHANNEL_UI = 3  # typing / UI pings

    def __init__(self):
        # Sounds folder expected at core/sounds/
        self.sounds_path = os.path.join(os.path.dirname(__file__), "sounds")
        self._init_mixer_safely()
        self._ambient_sound = None
        self._ambient_lock = threading.Lock()
        print("🎵 Jarvis Cinematic Sound Engine Ready — sounds_path:", self.sounds_path)

    # --------------------------------------------------------
    # MIXER INITIALIZATION (robust)
    # --------------------------------------------------------
    def _init_mixer_safely(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(12)
            else:
                pygame.mixer.set_num_channels(max(12, pygame.mixer.get_num_channels()))
        except Exception as e:
            print(f"⚠️ Mixer init failed: {e}")
            try:
                pygame.mixer.quit()
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(12)
            except Exception as e2:
                print(f"❌ Mixer recovery failed: {e2}")

    # --------------------------------------------------------
    # FILE LOADER
    # --------------------------------------------------------
    def _load_sound(self, filename):
        path = os.path.join(self.sounds_path, filename)
        if not os.path.exists(path):
            print(f"⚠️ Missing sound file: {path}")
            return None
        try:
            return pygame.mixer.Sound(path)
        except Exception as e:
            print(f"⚠️ Failed to load sound {path}: {e}")
            traceback.print_exc()
            return None

    def _get_channel(self, idx):
        # ensure mixer is initialized before getting channel
        self._init_mixer_safely()
        try:
            if idx >= pygame.mixer.get_num_channels():
                pygame.mixer.set_num_channels(idx + 4)
            return pygame.mixer.Channel(idx)
        except Exception as e:
            print(f"⚠️ _get_channel failed for {idx}: {e}")
            return None

    # --------------------------------------------------------
    # INTERNAL PLAYBACK
    # --------------------------------------------------------
    def _play_on_channel(self, channel_idx, sound_obj, loop=False, limit=None, volume=1.0):
        if sound_obj is None:
            return

        def runner():
            try:
                ch = self._get_channel(channel_idx)
                if not ch:
                    return

                ch.set_volume(volume)
                ch.play(sound_obj, loops=(-1 if loop else 0))

                if overlay_instance:
                    try:
                        overlay_instance.react_to_audio(volume)
                    except Exception:
                        pass

                if limit:
                    time.sleep(limit)
                    try:
                        ch.fadeout(600)
                    except:
                        ch.stop()

            except Exception as e:
                print(f"⚠️ Playback error: {e}")
                traceback.print_exc()

        threading.Thread(target=runner, daemon=True).start()

    # --------------------------------------------------------
    # PUBLIC SOUND FUNCTIONS
    # --------------------------------------------------------
    def _play_sound(self, name, channel=None, loop=False, limit=None, volume=1.0):
        snd = self._load_sound(name)
        if snd:
            self._play_on_channel(channel, snd, loop=loop, limit=limit, volume=volume)

    def play_ack(self):
        self._play_sound("ack.mp3", channel=self.CHANNEL_UI, limit=0.8, volume=0.9)

    def play_startup(self, short=False):
        dur = 2 if short else 5
        self._play_sound("startup_sequence.mp3", channel=self.CHANNEL_SFX, limit=dur)
        # Optional overlay animation during boot
        if overlay_instance:
            try:
                for i in range(5):
                    overlay_instance.react_to_audio(1.0)
                    time.sleep(0.4)
                    overlay_instance.react_to_audio(0.2)
                    time.sleep(0.25)
            except Exception:
                pass

    def play_alert(self):
        self._play_sound("alert_warning.mp3", channel=self.CHANNEL_SFX)

    def play_success(self):
        self._play_sound("task_complete.mp3", channel=self.CHANNEL_SFX)

    def play_listening(self):
        self._play_sound("listening_mode.mp3", channel=self.CHANNEL_UI)

    # --------------------------------------------------------
    # AMBIENT BACKGROUND
    # --------------------------------------------------------
    def play_ambient(self):
        with self._ambient_lock:
            if not self._ambient_sound:
                self._ambient_sound = self._load_sound("ambient_background.mp3")
            if not self._ambient_sound:
                return

            ch = self._get_channel(self.CHANNEL_AMBIENT)
            if not ch:
                return
            try:
                if not ch.get_busy():
                    ch.play(self._ambient_sound, loops=-1)
                    ch.set_volume(0.4)
            except Exception as e:
                print(f"⚠️ play_ambient error: {e}")

    def fade_out_ambient(self, ms=1000):
        try:
            ch = self._get_channel(self.CHANNEL_AMBIENT)
            if ch:
                ch.fadeout(ms)
        except Exception:
            pass

    # --------------------------------------------------------
    # TYPEWRITER EFFECT (Enhanced realism)
    # --------------------------------------------------------
    def typing_effect(self, duration=0.15):
        """Play a random-soft click per keystroke."""
        try:
            vol = random.uniform(0.35, 0.9)
            self._play_sound("type_click.mp3", channel=self.CHANNEL_UI, limit=duration, volume=vol)
        except Exception as e:
            print(f"⚠️ Typing effect failed: {e}")

    # --------------------------------------------------------
    # MOOD TONES (intentional small set)
    # --------------------------------------------------------
    def mood_tone(self, mood):
        mood = (mood or "").lower()
        try:
            if mood == "alert":
                self.play_alert()
            elif mood == "happy":
                self.play_success()
            elif mood == "listening":
                self.play_listening()
            # serious tone intentionally not used here
        except Exception:
            pass

    # --------------------------------------------------------
    # STOP ALL SOUNDS
    # --------------------------------------------------------
    def stop_all(self):
        try:
            for i in range(16):
                ch = self._get_channel(i)
                if ch:
                    try:
                        ch.stop()
                    except:
                        pass
        except Exception:
            pass


##############################################################################################################
# FILE: Jarvis\core\whatsapp_selenium.py
##############################################################################################################

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os


def send_whatsapp_message(contact: str, message: str):
    chrome_options = Options()

    # ✅ Dedicated WhatsApp profile (NO crash, NO conflict)
    profile_dir = os.path.abspath("jarvis_whatsapp_profile")
    chrome_options.add_argument(f"--user-data-dir={profile_dir}")
    chrome_options.add_argument("--profile-directory=Default")

    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    driver.get("https://web.whatsapp.com")
    print("🟢 WhatsApp Web opened. Scan QR once if needed.")
    time.sleep(15)

    # 🔍 Search box (stable selector)
    search_box = driver.find_element(
        By.XPATH,
        "//div[@contenteditable='true' and @data-tab='3']"
    )
    search_box.click()
    search_box.clear()
    search_box.send_keys(contact)
    time.sleep(2)

    # 🔎 Find matching contact
    chats = driver.find_elements(By.XPATH, "//span[@title]")
    target = None

    for chat in chats:
        if contact.lower() in chat.get_attribute("title").lower():
            target = chat
            break

    if not target:
        print(f"❌ Contact '{contact}' not found. Message NOT sent.")
        driver.quit()
        return

    # ✅ Open chat
    target.click()
    time.sleep(1)

    # ✉️ Message input box
    message_box = driver.find_elements(
        By.XPATH, "//div[@contenteditable='true']"
    )[-1]

    message_box.send_keys(message)
    time.sleep(0.3)
    message_box.send_keys(Keys.ENTER)

    print(f"✅ Message sent to {target.get_attribute('title')}")
    time.sleep(2)
    driver.quit()
