# core/voice_effects.py
import os
import pygame
import threading
import time

# Prevent invalid audio device issues on Windows
os.environ.setdefault("SDL_AUDIODRIVER", "directsound")

# -----------------------------------------------------------------
# ✅ Lazy overlay attach (no QWidget creation before QApplication)
# -----------------------------------------------------------------
overlay_instance = None


def attach_overlay(overlay):
    """
    Attach the InterfaceOverlay instance after QApplication is created.
    Called from main.py once GUI is running.
    """
    global overlay_instance
    overlay_instance = overlay
    print("🌀 Overlay successfully linked with Jarvis voice system.")
# -----------------------------------------------------------------


class JarvisEffects:
    """Handles Jarvis sound effects and transitions — clean, cinematic, and stable."""

    # Channel indexes (keep channel 0 reserved for voice)
    CHANNEL_SFX = 1
    CHANNEL_AMBIENT = 2

    def __init__(self):
        self.sounds_path = os.path.join(os.path.dirname(__file__), "sounds")
        self._safe_init_mixer()
        self._ambient_sound = None
        self._ambient_lock = threading.Lock()
        print("🎵 Jarvis Sound Effects Module Ready")

    # ---------------- SAFE MIXER INIT ----------------
    def _safe_init_mixer(self):
        """Ensure pygame mixer initializes correctly (even after audio errors)."""
        try:
            if pygame.mixer.get_init():
                # do not quit completely if other modules depend on mixer;
                # ensure channels are present
                pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
            else:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(8)
        except Exception as e:
            print(f"⚠️ Mixer init failed: {e} — retrying simple init.")
            try:
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
                pygame.mixer.init()
                pygame.mixer.set_num_channels(8)
            except Exception as e2:
                print(f"❌ Mixer recovery failed: {e2}")

    # ---------------- LOW-LEVEL PLAY ----------------
    def _load_sound(self, filename):
        path = os.path.join(self.sounds_path, filename)
        if not os.path.exists(path):
            print(f"⚠️ Missing sound file: {path}")
            return None
        try:
            return pygame.mixer.Sound(path)
        except Exception as e:
            print(f"⚠️ Failed to load sound {path}: {e}")
            return None

    def _get_channel(self, idx):
        try:
            self._safe_init_mixer()
            # if channel index is larger than configured, ensure num_channels
            if idx >= pygame.mixer.get_num_channels():
                pygame.mixer.set_num_channels(idx + 4)
            return pygame.mixer.Channel(idx)
        except Exception as e:
            print(f"⚠️ Could not get mixer channel {idx}: {e}")
            return None

    def _play_on_channel(self, channel_idx, sound_obj, loop=False, limit_seconds=None):
        """Play a Sound object on a given channel in a background thread."""
        if sound_obj is None:
            return

        def runner():
            try:
                ch = self._get_channel(channel_idx)
                if not ch:
                    return
                loops = -1 if loop else 0
                ch.play(sound_obj, loops=loops)
                # trigger overlay once at start
                try:
                    if overlay_instance:
                        overlay_instance.react_to_audio(1.0)
                except Exception:
                    pass

                if limit_seconds and limit_seconds > 0:
                    start = time.time()
                    while ch.get_busy() and (time.time() - start) < limit_seconds:
                        time.sleep(0.1)
                    # fade out gracefully
                    try:
                        ch.fadeout(600)
                    except Exception:
                        ch.stop()
                else:
                    # if looping indefinitely, just return control to mixer
                    while ch.get_busy():
                        time.sleep(0.2)
            except Exception as e:
                print(f"⚠️ Playback thread error: {e}")

        threading.Thread(target=runner, daemon=True).start()

    # ---------------- HIGH-LEVEL SOUND PLAYBACK ----------------
    def _play_sound(self, filename, sfx=True, loop=False, limit_seconds=None):
        """Public helper to play a sound file."""
        sound = self._load_sound(filename)
        if not sound:
            return

        channel_idx = self.CHANNEL_SFX if sfx else self.CHANNEL_AMBIENT
        self._play_on_channel(channel_idx, sound, loop=loop, limit_seconds=limit_seconds)

    # ---------------- TONE & MOOD ----------------
    def play_startup(self, short=False):
        """
        Play cinematic startup tone.
        If short=True, play only ~2 seconds (used during quick reactivations).
        """
        duration = 2 if short else 5
        # prefer a WAV/OGG with quick fade capability; use SFX channel
        self._play_sound("startup_sequence.mp3", sfx=True, loop=False, limit_seconds=duration)

    def play_alert(self):
        """Play warning tone."""
        self._play_sound("alert_warning.mp3", sfx=True, loop=False)

    def play_success(self):
        """Play success tone."""
        self._play_sound("task_complete.mp3", sfx=True, loop=False)

    def play_listening(self):
        """Short ping tone when Jarvis starts listening."""
        self._play_sound("listening_mode.mp3", sfx=True, loop=False)

    def play_serious(self):
        """Soft tone for serious or calm modes."""
        self._play_sound("serious_tone.mp3", sfx=True, loop=False)

    # ---------------- AMBIENT CONTROL ----------------
    def play_ambient(self):
        """Play looping ambient background sound softly (ambient channel)."""
        with self._ambient_lock:
            if self._ambient_sound is None:
                self._ambient_sound = self._load_sound("ambient_background.mp3")
            if not self._ambient_sound:
                return
            # If ambient already playing on channel, don't restart
            ch = self._get_channel(self.CHANNEL_AMBIENT)
            try:
                if ch and ch.get_busy():
                    return
            except Exception:
                pass
            self._play_on_channel(self.CHANNEL_AMBIENT, self._ambient_sound, loop=True, limit_seconds=None)

    def fade_in_ambient(self, filename="ambient_background.mp3", duration=1000):
        """Start ambient loop with a small delay (simulated fade-in)."""
        self.play_ambient()
        # small sleep to simulate fade-in progress (actual volume fade would need set_volume calls)
        time.sleep(duration / 1000.0)

    # ---------------- FADE & STOP ----------------
    def fade_out_ambient(self, duration=1000):
        """Smooth fade-out for ambient channel."""
        try:
            ch = self._get_channel(self.CHANNEL_AMBIENT)
            if ch and ch.get_busy():
                ch.fadeout(duration)
                time.sleep(duration / 1000.0)
        except Exception as e:
            print(f"⚠️ Fade-out ambient error: {e}")

    def stop_all(self):
        """Immediately stop SFX and ambient sounds (but keep mixer alive)."""
        try:
            ch_sfx = self._get_channel(self.CHANNEL_SFX)
            ch_amb = self._get_channel(self.CHANNEL_AMBIENT)
            if ch_sfx:
                ch_sfx.stop()
            if ch_amb:
                ch_amb.stop()
        except Exception as e:
            print(f"⚠️ stop_all error: {e}")

    # ---------------- MOOD TONES ----------------
    def mood_tone(self, mood):
        """Play a tone or cue based on Jarvis’s current mood."""
        mood = (mood or "").lower().strip()
        if mood == "alert":
            self.play_alert()
        elif mood == "happy":
            self.play_success()
        elif mood == "listening":
            self.play_listening()
        elif mood == "serious":
            self.play_serious()
