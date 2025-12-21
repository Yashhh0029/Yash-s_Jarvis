from core.background_listener import BackgroundListener
import time

def main():
    print("JARVIS STARTED (BACKGROUND MODE)")  # TEMP DEBUG

    listener = BackgroundListener()
    listener.start()

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
