import sys
import math
import time
import threading
import numpy as np
import sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets


class InterfaceOverlay(QtWidgets.QWidget):
    """Floating Siri-style circular overlay — mic-reactive + Jarvis mood reactive."""

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

        # Visual state
        self.pulse = 0
        self.reactive_boost = 0.0
        self.mic_intensity = 0.0
        self.status_text = "Booting..."
        self.mood = "neutral"
        self.running = True

        # Threads
        self._mic_thread = None
        self._anim_thread = None

        # Mood color sets (used in paint)
        self.mood_colors = {
            "happy": [QtGui.QColor(60, 220, 200), QtGui.QColor(0, 160, 255)],
            "serious": [QtGui.QColor(255, 140, 0), QtGui.QColor(255, 80, 0)],
            "alert": [QtGui.QColor(255, 60, 90), QtGui.QColor(255, 0, 0)],
            "neutral": [QtGui.QColor(0, 200, 255), QtGui.QColor(120, 80, 255)]
        }

    # ---------------- Public Controls ----------------
    def react_to_audio(self, intensity=1.0):
        self.reactive_boost = max(0.3, min(1.5, intensity))
        self.update()

    def set_status(self, text):
        self.status_text = text
        self.update()

    def set_mood(self, mood):
        self.mood = mood if mood in self.mood_colors else "neutral"
        self.update()

    # ---------------- Mic Listener ----------------
    def _mic_listener(self):
        try:
            def callback(indata, frames, time_, status):
                vol = np.linalg.norm(indata) / (frames**0.5 + 1e-9)
                self.mic_intensity = min(max(vol * 10.0, 0.0), 1.0)

            with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=1024):
                while self.running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"⚠️ Mic listener failed: {e}")
            self.mic_intensity = 0.0

    # ---------------- Animation ----------------
    def _animate_loop(self):
        while self.running:
            self.pulse = (self.pulse + 3) % 360
            self.reactive_boost = max(self.reactive_boost * 0.92, 0.0)
            self.update()
            time.sleep(0.016)

    # ---------------- Fade-in ----------------
    def _fade_in(self):
        for i in range(0, 21):
            self.setWindowOpacity(i / 20)
            time.sleep(0.03)

    # ---------------- Paint ----------------
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center = QtCore.QPointF(w / 2, h / 2)

        colors = self.mood_colors.get(self.mood, self.mood_colors["neutral"])
        col_inner, col_outer = colors

        glow = (math.sin(math.radians(self.pulse)) + 1) * 0.5 * (1 + self.reactive_boost)
        outer_radius = 70 + 25 * self.reactive_boost

        gradient = QtGui.QRadialGradient(center, outer_radius)
        gradient.setColorAt(0.0, QtGui.QColor(col_inner.red(), col_inner.green(), col_inner.blue(), int(200 * (0.6 + glow * 0.4))))
        gradient.setColorAt(0.6, QtGui.QColor(col_outer.red(), col_outer.green(), col_outer.blue(), int(120 * (0.6 + glow * 0.4))))
        gradient.setColorAt(1.0, QtCore.Qt.transparent)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(center, outer_radius, outer_radius)

        for i, scale in enumerate([0.85, 0.65, 0.45, 0.25]):
            alpha = int(80 * (1 + 0.5 * self.reactive_boost) * (1 - i * 0.15))
            pen = QtGui.QPen(QtGui.QColor(col_outer.red(), col_outer.green(), col_outer.blue(), alpha), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(center, outer_radius * scale, outer_radius * scale)

        core_radius = 35 + 8 * self.reactive_boost
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(col_inner)
        painter.drawEllipse(center, core_radius, core_radius)

        if self.status_text:
            painter.setPen(QtGui.QColor(255, 255, 255, 220))
            font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, self.status_text)

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

        self._anim_thread = threading.Thread(target=self._animate_loop, daemon=True)
        self._anim_thread.start()

        self._mic_thread = threading.Thread(target=self._mic_listener, daemon=True)
        self._mic_thread.start()

        threading.Thread(target=self._fade_in, daemon=True).start()
        self.show()

    def stop(self):
        self.running = False
        self.close()
        print("🛑 Siri-style Overlay stopped.")
