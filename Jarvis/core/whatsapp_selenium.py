from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import threading

WHATSAPP_LOCK = threading.Lock()



def send_whatsapp_message(contact: str, message: str):
    with WHATSAPP_LOCK:   # 🔒 IMPORTANT — prevents Chrome crash

        chrome_options = Options()

        # ✅ Dedicated WhatsApp profile (NO crash, NO conflict)
        profile_dir = os.path.abspath("jarvis_whatsapp_profile")
        chrome_options.add_argument(f"--user-data-dir={profile_dir}")
        chrome_options.add_argument("--profile-directory=Default")

        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-notifications")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

        driver.get("https://web.whatsapp.com")
        print("🟢 WhatsApp Web opened. Scan QR once if needed.")
        time.sleep(15)

        # 🔍 Search box
        search_box = driver.find_element(
            By.XPATH,
            "//div[@contenteditable='true' and @data-tab='3']"
        )
        search_box.click()
        search_box.send_keys(contact)
        time.sleep(2)

        # 🔎 Find matching contact
        chats = driver.find_elements(By.XPATH, "//span[@title]")
        target = None

        for chat in chats:
            if contact.lower() in chat.get_attribute("title").lower():
                target = chat
                break

        if not target:
            print(f"❌ Contact '{contact}' not found.")
            driver.quit()
            return

        # ✅ Open chat
        target.click()
        time.sleep(1)

        # ✉️ Message box
        message_box = driver.find_elements(
            By.XPATH, "//div[@contenteditable='true']"
        )[-1]

        message_box.send_keys(message)
        message_box.send_keys(Keys.ENTER)

        print(f"✅ Message sent to {target.get_attribute('title')}")
        time.sleep(2)
        driver.quit()
