from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import os

_driver = None  # SINGLE DRIVER INSTANCE


def get_youtube_driver():
    global _driver
    if _driver:
        return _driver

    options = Options()

    # ✅ DEDICATED JARVIS YOUTUBE PROFILE (NO LOGIN LOOP, NO CRASH)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yt_profile = os.path.join(base_dir, "selenium_profiles", "youtube")
    os.makedirs(yt_profile, exist_ok=True)

    options.add_argument(f"--user-data-dir={yt_profile}")
    options.add_argument("--profile-directory=Default")

    # Stability flags
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service()  # chromedriver from PATH
    _driver = webdriver.Chrome(service=service, options=options)
    return _driver


def open_youtube():
    driver = get_youtube_driver()

    # ✅ DO NOT RELOAD if already on YouTube
    try:
        if "youtube.com" not in driver.current_url:
            driver.get("https://www.youtube.com")
            time.sleep(2)
    except:
        driver.get("https://www.youtube.com")
        time.sleep(2)


def search_youtube(query):
    driver = get_youtube_driver()

    box = driver.find_element(By.NAME, "search_query")
    box.clear()

    for ch in query:
        box.send_keys(ch)
        time.sleep(0.06)

    box.send_keys(Keys.RETURN)
    time.sleep(2)


def play_nth_video(n=1):
    driver = get_youtube_driver()

    videos = driver.find_elements(By.ID, "video-title")
    if len(videos) < n:
        return

    video = videos[n - 1]

    # ✅ FIX: make element interactable
    driver.execute_script(
        "arguments[0].scrollIntoView({block:'center'});", video
    )
    time.sleep(0.5)
    driver.execute_script("arguments[0].click();", video)


def play_pause():
    get_youtube_driver().find_element(By.TAG_NAME, "body").send_keys("k")


def fullscreen():
    get_youtube_driver().find_element(By.TAG_NAME, "body").send_keys("f")


def mute():
    get_youtube_driver().find_element(By.TAG_NAME, "body").send_keys("m")


def forward(seconds=10):
    body = get_youtube_driver().find_element(By.TAG_NAME, "body")
    for _ in range(max(1, seconds // 5)):
        body.send_keys(Keys.ARROW_RIGHT)


def backward(seconds=10):
    body = get_youtube_driver().find_element(By.TAG_NAME, "body")
    for _ in range(max(1, seconds // 5)):
        body.send_keys(Keys.ARROW_LEFT)


def scroll_down():
    get_youtube_driver().execute_script("window.scrollBy(0, 600);")


def scroll_up():
    get_youtube_driver().execute_script("window.scrollBy(0, -600);")
