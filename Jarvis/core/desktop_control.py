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
