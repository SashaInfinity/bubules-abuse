import multiprocessing
import pyautogui
import time
import keyboard
import os
import cv2
import numpy as np
from PIL import Image
# import pytesseract


screenshots_path = './screenshots/'

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.common.exceptions import WebDriverException

# chrome_options = Options()
# chrome_options.add_argument("--window-size=1280,720")
# chrome_options.add_argument("--window-position=0,0")
# chrome_options.add_argument('--log-level=3')
# chrome_options.add_argument("--mute-audio")

# browser = webdriver.Chrome(chrome_options=chrome_options)
# browser.get('https://rides.imaginaryones.com/b-rider')


# while True:
#     if browser.execute_script("return document.readyState") == "complete":
#         break

# while True:
#     window_handles = browser.window_handles
    
#     if not window_handles:
#         quit()

def get_new_session():
    folders = [d for d in os.listdir(screenshots_path) if os.path.isdir(os.path.join(screenshots_path, d))]
    if (len(folders) == 0):
        return '1'
    
    sessions = sorted(folders,  key=lambda x: int(x))
    new_session_num = str(int(sessions[-1]) + 1)
    return new_session_num


def make_screenshot(path, index):
    isShiftPressed = True

    screenshot = pyautogui.screenshot(region=(0, 0, 1280,728))
    new_size = (int(screenshot.size[0] * 0.6), int(screenshot.size[1] * 0.6))
    screenshot = screenshot.resize(new_size, resample=Image.ANTIALIAS)

    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    red_pixels = cv2.findNonZero(cv2.inRange(frame[0:50, :], (0, 0, 200), (50, 50, 255)))
    if red_pixels is not None:
        isShiftPressed = False
    
    x, y, w, h = 0, 68, 768, 436 #0, 64, 768, 432  (1280/720)
    screenshot = screenshot.crop((x, y, w, h))

    # x, y, w, h = 36, 304, 160, 334
    # score_frame = screenshot.crop((x, y, w, h))
    # gray = cv2.cvtColor(np.array(score_frame), cv2.COLOR_BGR2GRAY)
    # _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # dights = pytesseract.image_to_string(threshold, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    # filtered_dights = "".join(c for c in dights if  c.isdecimal())
    # score = filtered_dights if filtered_dights else '0'
    # + 'score=' + score \


    filename = path + "/" \
    + str(index) + '_' \
    + 'shiftPos=' + ('1' if isShiftPressed else '0') \
    + '.jpg'

    screenshot.save(filename)

def start_making_screenshots():
    session_num = get_new_session()
    path_to_save = screenshots_path + session_num
    os.makedirs(path_to_save)

    index = 1
    while True:
        process = multiprocessing.Process(target=make_screenshot, args=(path_to_save, index,))
        process.start()
        index+=1
        time.sleep(0.03)


def stop_running():
    process.terminate()

def start_running():
    global process
    process = multiprocessing.Process(target=start_making_screenshots)
    process.start()

if __name__ == '__main__':
    keyboard.add_hotkey('alt+q', stop_running, suppress=True)
    keyboard.add_hotkey('alt+s', start_running, suppress=True)
    keyboard.wait('esc')

