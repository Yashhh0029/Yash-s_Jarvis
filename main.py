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
