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
