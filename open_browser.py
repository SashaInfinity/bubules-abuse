import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--window-size=1296,742")
chrome_options.add_argument("--window-position=0,0")
chrome_options.add_argument('--log-level=3')
chrome_options.add_argument("--mute-audio")

driver = webdriver.Chrome(options=chrome_options)
driver.set_window_position(-8, -1)
driver.get('https://rides.imaginaryones.com/b-rider')

while True:
    if driver.execute_script("return document.readyState") == "complete":
        break

while True:
    time.sleep(1)
    window_handles = driver.window_handles

    if not window_handles:
        quit()
