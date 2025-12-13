# main.py — ENTRY POINT ONLY (UI + backend thread)

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
