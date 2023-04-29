import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image_array(path: Path):
    return np.array(np.transpose(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), (2, 0, 1)),
                    dtype=np.float32) / 255


def load_img(path: Path):
    spl = path.name.split("_")
    z = torch.zeros(2)
    z[int(spl[1].split("=")[1].split(".")[0])] = 1.0
    # z = int(spl[1].split("=")[1].split(".")[0])
    img_data = {
        "img_array": path,
        "num": int(spl[0]),
        "game_num": int(path.parent.name),
        "space": z
    }
    return img_data


def collect_datasets(games_path: Path):
    games_paths = list(games_path.glob("*"))
    games = []
    for game in tqdm(games_paths):
        imgs_paths = list(game.glob("*.jpg"))
        game_data = [load_img(x) for x in imgs_paths]
        sorted_game_data = list(sorted(game_data, key=lambda x: x["num"]))
        for i in range(len(sorted_game_data) - 1):
            sorted_game_data[i]["next_pred"] = sorted_game_data[i + 1]["space"]
        games.append(list(filter(lambda x: "next_pred" in x.keys(), sorted_game_data)))
    return games


def transform_dataset2list(games):
    imgs = []
    for game in games:
        for img in game:
            imgs.append(img)
    random.shuffle(imgs)
    return imgs


class CustomImageDataset(Dataset):
    def __init__(self, data_elements):
        self.img_labels = data_elements

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = load_image_array((self.img_labels[idx]["img_array"]))
        label = self.img_labels[idx]["next_pred"]
        return image, label


def split_dataset(dataset, prop_train, prop_test, batch_size, shuffle=False):
    train_count = int(len(dataset) * prop_train)
    test_count = int(len(dataset) * prop_test)
    train_data = dataset[:train_count]
    valid_data = dataset[train_count:-test_count]
    test_data = dataset[-test_count:]
    train_dataset = CustomImageDataset(train_data)
    valid_dataset = CustomImageDataset(valid_data)
    test_dataset = CustomImageDataset(test_data)
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    return data_loader_train, data_loader_valid, data_loader_test


if __name__ == "__main__":
    games_path_data = Path("../data/clear")
    res = collect_datasets(games_path_data)
    dataset = transform_dataset2list(res)
    train_loader, valid_loader, test_loader = split_dataset(dataset, 0.7, 0.15, 256)
    print(len(res))
    print(sum([len(x) for x in res]))
    print(train_loader)
