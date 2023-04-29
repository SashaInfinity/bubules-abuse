import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from nn_scripts.emulation import collect_datasets, transform_dataset2list, split_dataset
from nn_scripts.models.resnet import ResNet, ResidualBlock

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 208

games_path_data = Path("/media/kirrog/ssd_cache/data/bubble_abuse/data/clear")
games = collect_datasets(games_path_data)
dataset = transform_dataset2list(games)
print("Shuffeled")
train_loader, valid_loader, test_loader = split_dataset(dataset, 0.8, 0.1, batch_size)
print("Loaders created")
num_classes = 2
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-8

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(resnet_groups_optimizer, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

# Train the model
total_step = len(train_loader)

experiment = "base"
experiment_path = Path(f"../model/unique_feature/{experiment}")
experiment_path.mkdir(exist_ok=True, parents=True)

counter = 0
start_time = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="training")):
        # counter += 1
        if counter > 10:
            print("--- %s seconds --- Load batch" % (time.time() - start_time))
            start_time = time.time()
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        if counter > 10:
            print("--- %s seconds --- Copy to GPU" % (time.time() - start_time))
            start_time = time.time()
        # Forward pass
        outputs = model(images)
        if counter > 10:
            print("--- %s seconds --- Infer model" % (time.time() - start_time))
            start_time = time.time()
        loss = criterion(outputs, labels)
        if counter > 10:
            print("--- %s seconds --- Calc loss" % (time.time() - start_time))
            start_time = time.time()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        if counter > 10:
            print("--- %s seconds --- Backward" % (time.time() - start_time))
            start_time = time.time()
        optimizer.step()
        if counter > 10:
            print("--- %s seconds --- Step" % (time.time() - start_time))
            start_time = time.time()
        # del images, labels, outputs
        if counter > 10:
            print("--- %s seconds --- Del" % (time.time() - start_time))
            start_time = time.time()
        # torch.cuda.empty_cache()
        # gc.collect()
        if counter > 10:
            print("--- %s seconds --- GC" % (time.time() - start_time))
            start_time = time.time()

        if counter > 20:
            exit(0)
    ep = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} Weight value: {sum([float(torch.sum(x)) for x in model.parameters()]):0.4f}'
    print(ep)

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            outputs[outputs > 0.5] = 1.0
            outputs[outputs <= 0.5] = 0.0
            correct += (outputs == labels).sum().item() / 2
            del images, labels, outputs
        acc = correct / total
        ac = 'Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * acc)
        print(ac)
    with open(str(experiment_path / "stats.txt"), "a") as f:
        f.write(f"{ep}\n")
        f.write(f"{ac}\n")
    torch.save(model.state_dict(), str(experiment_path / f"ep_{epoch:03d}_acc_{acc:04f}.bin"))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs
    acc = correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * acc))
    torch.save(model.state_dict(), str(experiment_path / f"result_acc_{acc:04f}.bin"))
del model
torch.cuda.empty_cache()
gc.collect()
