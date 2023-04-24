import cv2
from pathlib import Path

from tqdm import tqdm

path_to_data = Path("/media/kirrog/data/data/bubble/raw")
path_out_data = Path("/media/kirrog/data/data/bubble/clear")
path_out_data.mkdir(exist_ok=True, parents=True)
games = list(path_to_data.glob("*"))
print(len(games))
for game_path in games:
    out_path = path_out_data / game_path.name
    out_path.mkdir(exist_ok=True, parents=True)
    iters = list(game_path.glob("*.jpg"))
    with open(str(game_path / "range.txt"), "r") as f:
        start, stop = ata = f.read().split(",")
        start = int(start)
        stop = int(stop)
    for img_path in tqdm(iters):
        spl = img_path.name.split("_")
        img_num = int(spl[0])
        if start > img_num or stop < img_num:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (300, 150), interpolation=cv2.INTER_AREA)
        out_img_path = out_path / f"{img_num - start:09d}_{spl[1][:-4]}.jpg"
        cv2.imwrite(str(out_img_path), resized)
