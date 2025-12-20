from core import background_listener
import time

listener = background_listener.BackgroundListener()
listener.start()

print("Jarvis running in background. Press Ctrl+C to stop.")

while True:
    time.sleep(10)
