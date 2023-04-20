import cv2
import easygui
import os


sessions_path = './sessions/'
def get_new_session():
    folders = [d for d in os.listdir(sessions_path) if os.path.isdir(os.path.join(sessions_path, d))]
    if (len(folders) == 0):
        return '1'
    
    sessions = sorted(folders,  key=lambda x: int(x))
    new_session_num = str(int(sessions[-1]) + 1)
    return new_session_num

# Получить видео
file_path = easygui.fileopenbox()
video = cv2.VideoCapture(file_path)
if not video.isOpened():
    print("Ошибка при открытии файла")
    exit()

session_num = get_new_session()
path_to_save = sessions_path + session_num
os.makedirs(path_to_save)

# Разбить на кадры
success, frame = video.read()
while success:
    x, y, w, h = 0, 64, 768, 368 #432
    cropped_frame = frame[y:y+h, x:x+w]

    isSpacePressed = False
    red_pixels = cv2.findNonZero(cv2.inRange(cropped_frame[0:50, :], (0, 0, 200), (50, 50, 255)))
    if red_pixels is not None:
        isSpacePressed = True
    
    filename = path_to_save + "/" + str(int(video.get(cv2.CAP_PROP_POS_FRAMES))) \
    + '_' + 'shiftPos=' + ('0' if isSpacePressed else '1') + ".jpg"

    cv2.imwrite(filename, cropped_frame)

    success, frame = video.read()

# Освободить ресурсы
video.release()