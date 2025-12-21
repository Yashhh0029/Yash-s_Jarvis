print("🔥 ACTIVE command_handler.py LOADED FROM:", __file__)
import os
import webbrowser
import psutil
import pyautogui
import subprocess
import time
import threading
import random
import datetime

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
from core.youtube_driver import get_youtube_driver
from core.whatsapp_selenium import send_whatsapp_message
from core.intent_parser import parse_intent
from core.youtube_driver import (
    open_youtube,
    play_nth_video,
    play_pause,
    fullscreen,
    mute,
    forward,
    backward,
    scroll_up,
    scroll_down
)
EXPECTING_FOLLOWUP = False

THINKING_DELAY = 0.5    # seconds before saying "thinking"
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

if not any("chrome.exe" in (p.name() or "").lower() for p in psutil.process_iter()):
    YOUTUBE_ACTIVE = False

def is_youtube_command(command: str) -> bool:
    return (
        "youtube" in command
        or (
            state.LAST_APP_CONTEXT == "youtube"
            and any(k in command for k in [
                "play", "pause", "resume",
                "fullscreen", "mute",
                "forward", "backward", "rewind",
                "scroll up", "scroll down"
            ])
        )
    )

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
        command = (
                command
                .replace("full screen", "fullscreen")
                .replace("short screen", "fullscreen")
                .replace("minimise", "minimize")
                .replace("collapse", "")
                )

        if not command:
            return

        log_command(command)
        raw_command = command
        command = command.lower().strip()

        from core.context import get_last_action, set_last_action

        print(f"🎤 Processing Command: {command}")
        
        # ==================================================
        # WHATSAPP MESSAGE — HARD INPUT CAPTURE (FIX)
        # ==================================================
        if (
            PENDING_INTENT
            and PENDING_INTENT.get("intent") == "whatsapp_message"
            and PENDING_INTENT.get("contact") is not None
            and PENDING_INTENT.get("message") is None
        ):
            # 👇 Treat this input ONLY as message text
            PENDING_INTENT["message"] = raw_command.strip()

            contact = PENDING_INTENT["contact"]
            message = PENDING_INTENT["message"]

            speak(f"Sending message to {contact}.", mood="happy")

            threading.Thread(
                target=send_whatsapp_message,
                args=(contact, message),
                daemon=True
            ).start()

            PENDING_INTENT = None
            return

        # ==================================================
        # YOUTUBE — SINGLE CLEAN CONTROL BLOCK (SELENIUM ONLY)
        # ==================================================
        if is_youtube_command(command):

            # -------------------------------
            # OPEN YOUTUBE
            # -------------------------------
            if command == "open youtube":
                speak("Opening YouTube.", mood="happy")
                open_youtube()
                state.LAST_APP_CONTEXT = "youtube"
                state.LAST_YOUTUBE_SEARCH = False
                YOUTUBE_ACTIVE = True
                return

            # -------------------------------
            # SEARCH YOUTUBE (LETTER BY LETTER)
            # -------------------------------
            if (
                command.startswith("search youtube")
                or command.startswith("search on youtube")
                or ("search" in command and "youtube" in command)
                or (state.LAST_APP_CONTEXT == "youtube" and command.startswith("search"))
            ):
                query = (
                    command
                    .replace("search youtube", "")
                    .replace("search on youtube", "")
                    .replace("search", "")
                    .replace("youtube", "")
                    .strip()
                )

                if not query:
                    speak("What should I search on YouTube?")
                    return

                speak(f"Searching YouTube for {query}.", mood="happy")

                # Real typing via Selenium (letter by letter)
                from selenium.webdriver.common.by import By
                from selenium.webdriver.common.keys import Keys
                driver = get_youtube_driver()

                # Open YouTube ONLY if not already active
                if state.LAST_APP_CONTEXT != "youtube":
                    open_youtube()
                    time.sleep(2)

                box = driver.find_element(By.NAME, "search_query")
                box.clear()

                for ch in query:
                    box.send_keys(ch)
                    try:
                       jarvis_fx.typing_effect()
                    except:
                       pass
                    time.sleep(0.07)


                box.send_keys(Keys.RETURN)

                state.LAST_APP_CONTEXT = "youtube"
                state.LAST_YOUTUBE_SEARCH = True
                YOUTUBE_ACTIVE = True
                return

            # -------------------------------
            # PLAY Nth VIDEO (ANY NUMBER)
            # -------------------------------
            if "play" in command and "video" in command:

                number_map = {
                    "one": 1, "first": 1, "1st": 1,
                    "two": 2, "second": 2, "2nd": 2,
                    "three": 3, "third": 3, "3rd": 3,
                    "four": 4, "fourth": 4, "4th": 4,
                    "five": 5, "fifth": 5, "5th": 5
                }

                index = None
                for word in command.split():
                    if word.isdigit():
                        index = int(word)
                        break
                    if word in number_map:
                        index = number_map[word]
                        break

                if not index:
                    speak("Which video number should I play?")
                    return

                speak(f"Playing video number {index}.", mood="happy")
                play_nth_video(index)
                state.LAST_YOUTUBE_SEARCH = False
                return

            # -------------------------------
            # PLAYER CONTROLS
            # -------------------------------
            if command in ["play", "pause", "resume"]:
                play_pause()
                return

            if "fullscreen" in command:
                fullscreen()
                return

            if "mute" in command:
                mute()
                return

            if "forward" in command:
                forward(10)
                return

            if "backward" in command or "rewind" in command:
                backward(10)
                return

            if "scroll down" in command:
                scroll_down()
                return

            if "scroll up" in command:
                scroll_up()
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

            # STEP 3 — capture message & SEND
            if PENDING_INTENT["message"] is None:
               PENDING_INTENT["message"] = raw_command.strip()

               contact = PENDING_INTENT["contact"]
               message = PENDING_INTENT["message"]

               speak(f"Sending message to {contact}.", mood="happy")
               threading.Thread(
                  target=send_whatsapp_message,
                  args=(contact, message),
                  daemon=True
                  ).start()

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
        # INTENT THINKING GATE (NO ACTION, NO AUTOMATION)
        # --------------------------------------------------------------
        intent = parse_intent(raw_command)

        intent_name = intent.get("intent")
        params = intent.get("params", {})
        confidence = intent.get("confidence", 0.0)

        print("🧠 INTENT PARSED:", intent)
        
        # --------------------------------------------------------------
        # IGNORE VERY SHORT / BROKEN SPEECH FRAGMENTS
        # (does NOT affect logic, only prevents AI spam)
        # --------------------------------------------------------------
        if len(raw_command.strip()) < 4:
         return
        
        # --------------------------------------------------------------
        # FOLLOW-UP CONTINUATION (DO NOT RE-INTERPRET)
        # --------------------------------------------------------------
        global EXPECTING_FOLLOWUP
        if EXPECTING_FOLLOWUP and not command.startswith(("open", "search", "play")):
            EXPECTING_FOLLOWUP = False
            threading.Thread(
                target=self._ai_pipeline_worker,
                args=(raw_command,),
                daemon=True
            ).start()
            return

        # --------------------------------------------------------------
        # LOW-CONFIDENCE INTENT → SEND DIRECTLY TO AI
        # --------------------------------------------------------------
        if (
            intent_name == "unknown"
            or confidence < 0.5
        ) and not PENDING_INTENT:
            threading.Thread(
                target=self._ai_pipeline_worker,
                args=(raw_command,),
                daemon=True
            ).start()
            return

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
                if desktop:
                    desktop.increase_brightness()
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
        if (
           "mute" in command
           and "unmute" not in command
           and state.LAST_APP_CONTEXT != "youtube"
           ):

            try:
                if desktop: desktop.mute()
                speak_enhanced("Muted.", mood="neutral")
            except:
                speak("Failed to mute.", mood="alert")
            return

        if "unmute" in command and state.LAST_APP_CONTEXT != "youtube":
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
            and intent_name not in ["search", "open_app"]
            and any(command.startswith(pref) for pref in ["open ", "launch ", "search ", "type "])
            and "tab" not in command
            and "window" not in command
            and "youtube" not in command      
            and "whatsapp" not in command
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
            global EXPECTING_FOLLOWUP
            if enhanced.strip().endswith("?"):
                EXPECTING_FOLLOWUP = True

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

