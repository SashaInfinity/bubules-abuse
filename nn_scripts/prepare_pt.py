from pathlib import Path

import torch
from tqdm import tqdm

from nn_scripts.emulation import load_image_array

path_to_data = Path("/media/kirrog/ssd_cache/data/bubble_abuse/data/clear")
path_out_data = Path("/media/kirrog/ssd_cache/data/bubble_abuse/data/pytorch_tensors")
path_out_data.mkdir(exist_ok=True, parents=True)
games = list(sorted(list(path_to_data.glob("*")), key=lambda x: int(x.name)))
print(len(games))
count = 0
for game_path in games:
    out_path = path_out_data / game_path.name
    out_path.mkdir(exist_ok=True, parents=True)
    iters = list(game_path.glob("*.jpg"))
    size = ""
    for img_path in tqdm(iters, desc="game_pictures"):
        count += 1
        spl = img_path.name.split("_")
        img_num = int(spl[0])
        out_img_path = out_path / f"{img_num:09d}_{spl[1][:-4]}.pt"
        if out_img_path.exists():
            continue
        torch_data = load_image_array(img_path)
        size = torch_data.size()
        torch.save(torch_data, str(out_img_path))
    torch_data = load_image_array(iters[0])
    size = torch_data.size()
    print(f"Game: {game_path.name} Size: {size} Num: {len(iters)} Mean: {torch.mean(torch_data)}")
print(f"Num of files: {count}")
