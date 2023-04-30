import cv2
import numpy
import mss
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from nn_scripts.model_test import predict_space_state

from selenium.webdriver.common.by import By

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
    state = predict_space_state(screenshot)
    print("predicted : " + str(time.time()))

    if (state == 1):
        press_space(driver)
    else:
        up_space(driver)


def make_screenshot(sct):
        img = numpy.asarray(sct.grab(bbox))

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        reduction_percentage = 0.6
        resized_img = cv2.resize(img, None, fx=reduction_percentage, fy=reduction_percentage, interpolation=cv2.INTER_AREA)
        
        x, y, w, h = 0, 69, 768, 436
        cropped_img = resized_img[y:y+h, x:x+w]

        resized = cv2.resize(cropped_img, (300, 150), interpolation=cv2.INTER_AREA)

        return resized


def screen_record_start() -> int:
    sct = mss.mss()

    driver_address = "127.0.0.1:51796"
    driver = connect_to_driver(driver_address)
    print("READY")

    while True:
        print("start: " + str(time.time()))

        screenshot = make_screenshot(sct)
        print("screenshot : " + str(time.time()))

        analyze_screenshot(driver, screenshot)
        print("action-done : " + str(time.time()))


        # title = "[MSS] FPS benchmark"
        # cv2.imshow(title, screenshot)

        #Stop infinity loop if "Q" was pressed 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


screen_record_start()


