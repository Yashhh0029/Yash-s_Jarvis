# ============================================================
#  JARVIS MAIN.PY — WITH MERGED FaceAuth CLASS AT THE TOP
# ============================================================

import cv2
from deepface import DeepFace
from core.speech_engine import speak, jarvis_fx
from core.voice_effects import overlay_instance
import os
import time
import tempfile
import threading
import sys
import random
import importlib.util
from PyQt5 import QtWidgets


# ============================================================
#                🔥 MERGED FACE AUTH MODULE 🔥
# ============================================================

class FaceAuth:
    """Cinematic + stable face authentication for Yash (TensorFlow-free Facenet mode)."""

    def __init__(self):
        # Save reference image under /config/face_data
        self.reference_path = os.path.join("config", "face_data", "yash_reference.jpg")
        os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
        print("📸 Face Authentication Module Loaded (Merged in main.py)")

    # ---------------------------------------------------------
    def capture_reference(self):
        """Capture Yash's face as the base reference image."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            speak("Camera not accessible, Yash.", mood="alert")
            return

        speak("Please look at the camera. Capturing your reference image.", mood="serious")
        time.sleep(1.3)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            speak("I couldn't capture your face clearly. Try again.", mood="alert")
            return

        cv2.imwrite(self.reference_path, frame)
        speak("Reference image captured successfully.", mood="happy")
        print(f"✅ Reference saved at: {self.reference_path}")

    # ---------------------------------------------------------
    def verify_user(self):
        """Verify Yash using DeepFace face comparison (TensorFlow-free)."""

        if not os.path.exists(self.reference_path):
            speak("No reference image found. Let me capture one.", mood="alert")
            self.capture_reference()
            return self.verify_user()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera not accessible for verification.", mood="alert")
            return False

        speak("Verifying your identity. Keep looking at the camera.", mood="serious")

        # UI Scan Glow
        if overlay_instance:
            try:
                overlay_instance.set_status("🔍 Scanning your face…")
                overlay_instance.set_mood("neutral")
                overlay_instance.react_to_audio(0.8)
            except:
                pass

        # Ambient hum
        try:
            threading.Thread(target=jarvis_fx.play_ambient, daemon=True).start()
        except:
            pass

        # Scan Animation
        def scanning_anim():
            if not overlay_instance:
                return
            for _ in range(8):
                overlay_instance.react_to_audio(1.0)
                time.sleep(0.22)
                overlay_instance.react_to_audio(0.25)
                time.sleep(0.22)

        threading.Thread(target=scanning_anim, daemon=True).start()

        time.sleep(1.6)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            speak("Failed to capture a clear image for verification.", mood="alert")
            jarvis_fx.fade_out_ambient(800)
            return False

        # TEMP IMAGE
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img_temp_path = tmp.name
                cv2.imwrite(img_temp_path, frame)

            # TensorFlow-free DeepFace
            result = DeepFace.verify(
                img1_path=self.reference_path,
                img2_path=img_temp_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )

            os.remove(img_temp_path)
            verified = result.get("verified", False)

        except Exception as e:
            print(f"⚠️ DeepFace Error: {e}")
            speak("Something went wrong with face verification.", mood="alert")
            jarvis_fx.fade_out_ambient(800)
            return False

        jarvis_fx.fade_out_ambient(800)

        # RESULT
        if verified:
            if overlay_instance:
                overlay_instance.set_status("✅ Identity Verified — Welcome Yash")
                overlay_instance.set_mood("happy")
                overlay_instance.react_to_audio(1.3)

            speak("Identity verified. Welcome back, Yash.", mood="happy")
            print("✅ Face Verified: Access Granted")
            return True

        else:
            if overlay_instance:
                overlay_instance.set_status("❌ Face Not Recognized")
                overlay_instance.set_mood("alert")
                overlay_instance.react_to_audio(0.4)

            speak("I couldn't recognize you. Access denied.", mood="alert")
            print("❌ Verification Failed")
            return False


# ============================================================
#               🔹 NORMAL MAIN.PY STARTS HERE 🔹
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
    print("\n🤖 Initializing Yash’s JARVIS — Neural, Cinematic & Alive...\n")

    # late imports
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

    # Attach overlay
    try:
        if hasattr(overlay_instance, "attach_overlay"):
            overlay_instance.attach_overlay(overlay)
        else:
            overlay_instance = overlay
        print("🌀 Overlay successfully linked with Jarvis voice system.")
    except:
        pass

    # Boot Glow
    try: overlay.set_status("Booting systems…")
    except: pass

    for _ in range(3):
        try:
            overlay.react_to_audio(1.0); time.sleep(0.28)
            overlay.react_to_audio(0.2); time.sleep(0.28)
        except:
            time.sleep(0.6)

    # Startup sound
    try:
        jarvis_fx.play_startup()
    except:
        pass

    try:
        speak("System booting up. Initializing cognition and neural modules.", mute_ambient=True)
    except:
        pass

    time.sleep(1.2)

    # ----------------------------------------------------------
    #               FACE VERIFICATION (Merged)
    # ----------------------------------------------------------
    verified = True

    overlay.set_status("Verifying face…")
    speak("Verifying your identity. Please look at the camera, Yash.", mood="serious", mute_ambient=True)

    try:
        jarvis_fx.play_ambient()
    except:
        pass

    import queue
    result_q = queue.Queue()

    def _run_auth():
        try:
            auth = FaceAuth()
            res = auth.verify_user()
            result_q.put(bool(res))
        except Exception as ex:
            print(f"⚠️ FaceAuth exception: {ex}")
            result_q.put(False)

    t = threading.Thread(target=_run_auth, daemon=True)
    t.start()

    start_t = time.time()
    TIMEOUT = 14.0

    while t.is_alive() and time.time() - start_t < TIMEOUT:
        try:
            overlay.react_to_audio(0.9); time.sleep(0.35)
            overlay.react_to_audio(0.15); time.sleep(0.35)
        except:
            time.sleep(0.35)

    t.join(timeout=0.5)

    verified = result_q.get() if not result_q.empty() else False

    try: jarvis_fx.fade_out_ambient(800)
    except: pass

    if verified:
        overlay.set_status("Identity verified ✅")
        speak("Identity verified. Welcome back, Yash.", mood="happy", mute_ambient=True)
    else:
        overlay.set_status("Identity not recognized ❌")
        speak("I couldn't recognize you, Yash. Switching to limited mode.", mood="alert", mute_ambient=True)

    # ----------------------------------------------------------
    #               Mood-aware greeting
    # ----------------------------------------------------------
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

    speak(random.choice(mood_lines.get(last_mood, mood_lines["neutral"])), mute_ambient=True)
    time.sleep(0.3)

    if verified:
        speak("Say 'Hey Jarvis' when you're ready.", mute_ambient=True)
    else:
        speak("Say 'Hey Jarvis' to continue in limited mode.", mute_ambient=True)

    # ----------------------------------------------------------
    #               Start Hotword Listener
    # ----------------------------------------------------------
    overlay.set_status("Listening…")
    print("\n🎤 Hotword listener online — say: 'Hey Jarvis'\n")

    try:
        listener = JarvisListener()
    except Exception as e:
        print(f"⚠️ Failed to start JarvisListener: {e}")

    while True:
        time.sleep(1)


# ============================================================
#                    ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("CURRENT DIR =", os.getcwd())
    print("FILES =", os.listdir())

    app = QtWidgets.QApplication(sys.argv)

    from core.interface import InterfaceOverlay
    overlay = InterfaceOverlay()

    try:
        overlay.run()
    except Exception as e:
        print(f"⚠️ overlay.run() raised: {e}")
        overlay.show()

    backend_thread = threading.Thread(target=jarvis_startup, args=(overlay,), daemon=True)
    backend_thread.start()

    sys.exit(app.exec_())
