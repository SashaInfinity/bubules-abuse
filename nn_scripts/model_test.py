import torch

from nn_scripts.models.resnet import ResNet, ResidualBlock
MODEL_PATH_IN = "../model/unique_feature/base/ep_010_acc_0.953290.bin"
images = torch.zeros((1, 3, 150, 300))
model = ResNet(ResidualBlock, [3, 4, 6, 3])
model.load_state_dict(torch.load(MODEL_PATH_IN))
outputs = model(images)
print(outputs.size())
