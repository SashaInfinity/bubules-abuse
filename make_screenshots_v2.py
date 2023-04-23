
import cv2
import numpy
import time
import os
import mss
import keyboard

screenshots_path = './screenshots/'

def get_new_session():
    folders = [d for d in os.listdir(screenshots_path) if os.path.isdir(os.path.join(screenshots_path, d))]
    if (len(folders) == 0):
        return '1'
    
    sessions = sorted(folders,  key=lambda x: int(x))
    new_session_num = str(int(sessions[-1]) + 1)
    return new_session_num

def make_screenshot(index, path, sct):
        isShiftPressed = True

        print("start ["+ str(index) +"]: " + str(time.time()))

        bbox = {"top": 0, "left": 0, "width": 1280, "height": 728}
        img = numpy.array(sct.grab(bbox))
        resized_img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        
        x, y, w, h = 0, 69, 768, 436
        cropped_img = resized_img[y:y+h, x:x+w]

        #Зажат пробел или нет (использовать только с расширением)
        # frame = frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # red_pixels = cv2.findNonZero(cv2.inRange(frame[80:130, 0:50], (0, 0, 200), (50, 50, 255)))
        # if red_pixels is not None:
        #     isShiftPressed = False

        filename = path + "/" \
        + str(index) \
        + '.jpg'
        # + '_' + 'shiftPos=' + ('1' if isShiftPressed else '0') \


        print("finish ["+ str(index) +"]: " + str(time.time()))

        cv2.imwrite(filename, cropped_img)

def screen_record_efficient() -> int:
    index = 0
    session_num = get_new_session()
    path_to_save = screenshots_path + session_num
    os.makedirs(path_to_save)

    sct = mss.mss()

    while True:
        index += 1
        make_screenshot(index, path_to_save, sct)

        if cv2.waitKey(1000) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    keyboard.add_hotkey('alt+s', screen_record_efficient, suppress=True)
    keyboard.wait('esc')


#КОД ОПРЕДЕЛЯЕТ СКОР

# x, y, w, h = 36, 304, 160, 334
# score_frame = screenshot.crop((x, y, w, h))
# gray = cv2.cvtColor(np.array(score_frame), cv2.COLOR_BGR2GRAY)
# _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# dights = pytesseract.image_to_string(threshold, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
# filtered_dights = "".join(c for c in dights if  c.isdecimal())
# score = filtered_dights if filtered_dights else '0'
# + 'score=' + score \


# ПРИМЕР МУЛЬТИПРОЦЕССОВ

# process = multiprocessing.Process(target=make_screenshot, args=(path_to_save, index,))
# process.start()

# def stop_running():
#     process.terminate()

# def start_running():
#     global process
#     process = multiprocessing.Process(target=start_making_screenshots)
#     process.start()

