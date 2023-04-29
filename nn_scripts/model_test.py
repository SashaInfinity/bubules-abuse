import torch

from nn_scripts.models.resnet import ResNet, ResidualBlock

images = torch.zeros((1,3, 150, 300))
model = ResNet(ResidualBlock, [3, 4, 6, 3])
outputs = model(images)
print(outputs.size())