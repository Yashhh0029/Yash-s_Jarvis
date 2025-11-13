# main.py
# ============================================================
#  JARVIS MAIN — safe startup, merged FaceAuth (lazy DeepFace)
# ============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # reduce TF spam if present

import sys
import time
import threading
import random
import tempfile
from PyQt5 import QtWidgets

# Core UI & voice system (used later inside startup)
from core.interface import InterfaceOverlay
from core import voice_effects  # to attach overlay safely

# Global verification flag (single source of truth)
FACE_VERIFIED = False


# ============================================================
#   FaceAuth (merged) — lazy-imports DeepFace, fallback to OpenCV
# ============================================================
class FaceAuth:
    """Cinematic + stable face authentication for Yash.

    Tries to use DeepFace (Facenet + opencv backend) if available.
    If DeepFace / TensorFlow can't be imported, falls back to
    an OpenCV histogram similarity comparison as a safe alternative.
    """

    def __init__(self):
        self.reference_path = os.path.join("config", "face_data", "yash_reference.jpg")
        os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
        print("📸 Face Authentication Module Loaded (merged)")

    # ---------------------
    def capture_reference(self):
        """Capture and save a reference image for Yash."""
        import cv2
        from core.speech_engine import speak

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera not accessible, Yash.", mood="alert")
            return

        speak("Please look at the camera. Capturing your reference image.", mood="serious")
        time.sleep(1.2)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            speak("I couldn't capture your face clearly. Try again.", mood="alert")
            return

        cv2.imwrite(self.reference_path, frame)
        speak("Reference image captured successfully.", mood="happy")
        print(f"✅ Reference saved at: {self.reference_path}")

    # ---------------------
    def _fallback_compare(self, img1_path, img2_path):
        """Fast OpenCV histogram comparison fallback. Returns True if similar."""
        import cv2
        try:
            a = cv2.imread(img1_path)
            b = cv2.imread(img2_path)
            if a is None or b is None:
                return False

            # Resize to same size for stable histogram comparison
            h, w = 224, 224
            a = cv2.resize(a, (w, h))
            b = cv2.resize(b, (w, h))

            # Convert to HSV and compute histograms
            a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
            hist_a = cv2.calcHist([a_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_b = cv2.calcHist([b_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)  # -1..1 (1 perfect)
            return score >= 0.55
        except Exception as e:
            print("⚠️ Fallback compare failed:", e)
            return False

    # ---------------------
    def verify_user(self):
        """Verify user using DeepFace if available, else fallback compare."""
        import cv2
        from core.speech_engine import speak, jarvis_fx

        # Ensure reference exists
        if not os.path.exists(self.reference_path):
            speak("No reference image found. Let me capture one.", mood="alert")
            self.capture_reference()
            if not os.path.exists(self.reference_path):
                return False  # capture failed

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera not accessible for verification.", mood="alert")
            return False

        speak("Verifying your identity. Keep looking at the camera.", mood="serious")

        # UI + ambient
        if voice_effects.overlay_instance:
            try:
                voice_effects.overlay_instance.set_status("🔍 Scanning your face…")
                voice_effects.overlay_instance.set_mood("neutral")
                voice_effects.overlay_instance.react_to_audio(0.8)
            except Exception:
                pass

        # start ambient hum non-blocking
        try:
            threading.Thread(target=jarvis_fx.play_ambient, daemon=True).start()
        except Exception:
            pass

        # small scanning animation
        def _scan_anim():
            if not voice_effects.overlay_instance:
                return
            for _ in range(8):
                try:
                    voice_effects.overlay_instance.react_to_audio(1.0)
                    time.sleep(0.22)
                    voice_effects.overlay_instance.react_to_audio(0.25)
                    time.sleep(0.22)
                except Exception:
                    break

        threading.Thread(target=_scan_anim, daemon=True).start()

        time.sleep(1.4)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            speak("Failed to capture a clear image for verification.", mood="alert")
            try:
                jarvis_fx.fade_out_ambient(800)
            except Exception:
                pass
            return False

        # save temp image
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)
        except Exception as e:
            print("⚠️ Temp write failed:", e)
            try:
                jarvis_fx.fade_out_ambient(800)
            except Exception:
                pass
            return False

        verified = False

        # Try DeepFace (lazy import) first — but handle failures gracefully
        try:
            from deepface import DeepFace  # lazy
            # prefer Facenet + opencv backend to avoid extra ML backend if possible
            result = DeepFace.verify(
                img1_path=self.reference_path,
                img2_path=tmp_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )
            verified = bool(result.get("verified", False))
        except Exception as e:
            # DeepFace/TensorFlow import or verify failed — fallback
            print("⚠️ DeepFace verify failed or unavailable:", e)
            verified = self._fallback_compare(self.reference_path, tmp_path)

        # cleanup temp
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        # stop ambient
        try:
            jarvis_fx.fade_out_ambient(800)
        except Exception:
            pass

        # UI + voice result
        if verified:
            if voice_effects.overlay_instance:
                try:
                    voice_effects.overlay_instance.set_status("✅ Identity Verified — Welcome Yash")
                    voice_effects.overlay_instance.set_mood("happy")
                    voice_effects.overlay_instance.react_to_audio(1.3)
                except Exception:
                    pass
            speak("Identity verified. Welcome back, Yash.", mood="happy")
            print("✅ Face Verified: Access Granted")
            return True
        else:
            if voice_effects.overlay_instance:
                try:
                    voice_effects.overlay_instance.set_status("❌ Face Not Recognized")
                    voice_effects.overlay_instance.set_mood("alert")
                    voice_effects.overlay_instance.react_to_audio(0.4)
                except Exception:
                    pass
            speak("I couldn't recognize you. Access denied.", mood="alert")
            print("❌ Verification Failed")
            return False


# ============================================================
#  Normal main startup logic (uses FaceAuth above)
# ============================================================
def _time_greeting():
    import datetime
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 22:
        return "Good evening"
    return "Hello"


def jarvis_startup(overlay):
    """Runs Jarvis startup flow in a background thread (non-Qt)."""
    global FACE_VERIFIED
    print("\n🤖 Initializing Yash’s JARVIS — Neural, Cinematic & Alive...\n")

    # late imports to avoid PyQt/Qt conflicts
    try:
        from core.memory_engine import JarvisMemory
        from core.command_handler import JarvisCommandHandler
        from core.speech_engine import speak, jarvis_fx
        from core.listener import JarvisListener
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return

    memory = JarvisMemory()
    command_handler = JarvisCommandHandler()

    # Attach overlay to voice_effects (safe API)
    try:
        if hasattr(voice_effects, "attach_overlay"):
            voice_effects.attach_overlay(overlay)
        else:
            # fallback
            voice_effects.overlay_instance = overlay
        print("🌀 Overlay successfully linked with Jarvis voice system.")
    except Exception as e:
        print(f"⚠️ Could not attach overlay: {e}")

    # UI boot status
    try:
        overlay.set_status("Booting systems…")
    except Exception:
        pass

    # Cinematic boot glow
    for _ in range(3):
        try:
            overlay.react_to_audio(1.0)
            time.sleep(0.28)
            overlay.react_to_audio(0.2)
            time.sleep(0.28)
        except Exception:
            time.sleep(0.6)

    # Startup sound + speech
    try:
        jarvis_fx.play_startup()  # 5s by default
    except Exception as e:
        print(f"⚠️ play_startup failed: {e}")

    # Wait for startup sound to mostly finish so scan happens after it
    try:
        # play_startup plays (limit 5 sec in your voice_effects); wait safely
        time.sleep(5.2)
    except Exception:
        time.sleep(1.2)

    try:
        speak("System booting up. Initializing cognition and neural modules.", mute_ambient=True)
    except Exception:
        pass

    # allow audio device a moment
    time.sleep(0.6)

    # ---------------- Face verification ----------------
    face_auth = FaceAuth()
    verified = False

    try:
        try:
            overlay.set_status("Verifying face…")
        except Exception:
            pass

        try:
            speak("Verifying your identity. Please look at the camera, Yash.", mood="serious", mute_ambient=True)
        except Exception:
            pass

        # run face verify in worker thread with timeout
        import queue
        result_q = queue.Queue()

        def _run_auth_thread():
            try:
                res = face_auth.verify_user()
                result_q.put(bool(res))
            except Exception as ex:
                print("⚠️ FaceAuth exception:", ex)
                result_q.put(False)

        t = threading.Thread(target=_run_auth_thread, daemon=True)
        t.start()

        start_t = time.time()
        TIMEOUT = 14.0
        while t.is_alive() and time.time() - start_t < TIMEOUT:
            try:
                overlay.react_to_audio(0.9)
            except Exception:
                pass
            time.sleep(0.35)
            try:
                overlay.react_to_audio(0.15)
            except Exception:
                pass
            time.sleep(0.35)

        t.join(timeout=0.5)
        verified = result_q.get() if not result_q.empty() else False

        try:
            jarvis_fx.fade_out_ambient(800)
        except Exception:
            pass

    except Exception as e:
        print("⚠️ Face verification flow error:", e)
        verified = False

    # Persist result to global flag so listeners/other modules can check
    FACE_VERIFIED = bool(verified)

    # UI + voice after verification
    try:
        if FACE_VERIFIED:
            overlay.set_status("Identity verified ✅")
            speak("Identity verified. Welcome back, Yash.", mood="happy", mute_ambient=True)
        else:
            overlay.set_status("Identity not recognized ❌")
            speak("I couldn't recognize you, Yash. Switching to limited mode.", mood="alert", mute_ambient=True)
    except Exception:
        pass

    # ---------------- Mood-aware greeting ----------------
    time.sleep(0.6)
    greet = _time_greeting()
    last_mood = memory.get_mood()
    mood_lines = {
        "happy": [
            f"{greet}, Yash. You left on a high note last time.",
            f"{greet}, Yash. You sounded cheerful previously."
        ],
        "serious": [
            f"{greet}, Yash. You were focused last time.",
            f"{greet}, Yash. Let's get things done."
        ],
        "alert": [
            f"{greet}, Yash. You seemed cautious previously.",
            f"{greet}, Yash. Systems are steady."
        ],
        "neutral": [
            f"{greet}, Yash. Everything’s stable.",
            f"{greet}, Yash. I’m online and ready."
        ]
    }

    try:
        speak(random.choice(mood_lines.get(last_mood, mood_lines["neutral"])), mute_ambient=True)
        time.sleep(0.3)
        if FACE_VERIFIED:
            speak("Say 'Hey Jarvis' when you're ready.", mute_ambient=True)
        else:
            speak("Say 'Hey Jarvis' to continue in limited mode.", mute_ambient=True)
    except Exception:
        pass

    # ---------------- Start Listener ----------------
    try:
        overlay.set_status("Listening…")
    except Exception:
        pass

    print("\n🎤 Hotword listener online — say: 'Hey Jarvis'\n")
    try:
        listener = JarvisListener()
    except Exception as e:
        print("⚠️ Failed to start JarvisListener:", e)

    # keep backend alive
    while True:
        time.sleep(1)


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("CURRENT DIR =", os.getcwd())
    print("FILES =", os.listdir())

    app = QtWidgets.QApplication(sys.argv)
    overlay = InterfaceOverlay()

    try:
        overlay.run()
    except Exception as e:
        print("⚠️ overlay.run() raised:", e)
        overlay.show()

    backend_thread = threading.Thread(target=jarvis_startup, args=(overlay,), daemon=True)
    backend_thread.start()

    sys.exit(app.exec_())
