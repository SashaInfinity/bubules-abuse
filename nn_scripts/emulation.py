from pathlib import Path

import cv2
from tqdm import tqdm

games_path_data = Path("/media/kirrog/data/data/bubble/clear")


def create_datasets():
    games_paths = list(games_path_data.glob("*"))
    games = []
    for game in tqdm(games_paths):
        game_data = []
        for img_path in game.glob("*.jpg"):
            spl = img_path.name.split("_")
            img_data = {
                "img_array": cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED),
                "num": int(spl[0]),
                "space": int(spl[1].split("=")[1].split(".")[0])
            }
            game_data.append(img_data)
        games.append(game_data)
    return games


res = create_datasets()
print(len(res))
