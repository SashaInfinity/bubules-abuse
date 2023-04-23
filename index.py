import cv2
import numpy
import mss
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


bbox = {"top": 0, "left": 0, "width": 1280, "height": 728}

def press_space(driver):
    ActionChains(driver).send_keys(Keys.SPACE).key_down(Keys.SPACE).perform()

def up_space(driver):
    ActionChains(driver).key_up(Keys.SPACE).perform()

def connect_to_driver(address):
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", address)
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def analyze_screenshot(driver,screenshot):
    press_space(driver)
    time.sleep(0.5)
    up_space(driver)


def make_screenshot(sct):
        img = numpy.asarray(sct.grab(bbox))

        reduction_percentage = 0.6
        resized_img = cv2.resize(img, None, fx=reduction_percentage, fy=reduction_percentage, interpolation=cv2.INTER_AREA)
        
        x, y, w, h = 0, 69, 768, 436
        cropped_img = resized_img[y:y+h, x:x+w]

        return cropped_img


def screen_record_start() -> int:
    sct = mss.mss()

    driver_address = "127.0.0.1:63384"
    driver = connect_to_driver(driver_address)

    while True:
        screenshot = make_screenshot(sct)

        analyze_screenshot(driver, screenshot)

        # title = "[MSS] FPS benchmark"
        # cv2.imshow(title, screenshot)

        #Stop infinity loop if "Q" was pressed 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


screen_record_start()


