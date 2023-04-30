import torch
import cv2
import time

from models.resnet import ResNet, ResidualBlock
MODEL_PATH_IN = "./models/ep_001_acc_0.951565.bin"
IMG_PATH = "./123.jpg"
img = cv2.imread(str(IMG_PATH))
reduction_percentage = 0.6
resized_img = cv2.resize(img, None, fx=reduction_percentage, fy=reduction_percentage, interpolation=cv2.INTER_AREA)

x, y, w, h = 0, 69, 768, 436
cropped_img = resized_img[y:y+h, x:x+w]

resized = cv2.resize(cropped_img, (300, 150), interpolation=cv2.INTER_AREA)

def predict_space_state(screenshot):
    print("start: " + str(time.time()))
    tensor = torch.from_numpy(screenshot).permute(2, 0, 1).unsqueeze(0).float()

    model = ResNet(ResidualBlock, [3, 2, 3, 1])
    model.load_state_dict(torch.load(MODEL_PATH_IN))
    outputs = model(tensor)

    space_up_probability = outputs[0][0].item()
    space_down_probability = outputs[0][1].item()
    print("finish: " + str(time.time()))


    if space_up_probability > space_down_probability:
        return 0
    else:
        return 1

predict_space_state(resized)