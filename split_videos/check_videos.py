from pathlib import Path
from tqdm import tqdm
import cv2


path_to_data = Path("./sessions")
all_folders = path_to_data.glob("*")
file_paths = sorted([x for x in all_folders if x.name != ".DS_Store"], key=lambda x: int(x.name.split()[0]))

broken_frames = []
for file_path in file_paths:
    iters = list(file_path.glob("*.jpg"))
    broken_frames = []
    
    for img_path in tqdm(iters):
        img = cv2.imread(str(img_path))

        height, width = img.shape[:2]
        crop_img = img[int(0.75 * height):, :]

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(thresh) == 0:
            broken_frames.append(img_path)

    print(file_path, len(broken_frames)>0)
    if (len(broken_frames)>0):
        broken_frames.append(file_path)

print(broken_frames)

