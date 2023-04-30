import torch

from nn_scripts.models.resnet import ResNet, ResidualBlock
MODEL_PATH_IN = "./models/ep_010_acc_0.953290.bin"

def predict_space_state(screenshot):
    tensor = torch.from_numpy(screenshot).permute(2, 0, 1).unsqueeze(0).float()

    model = ResNet(ResidualBlock, [3, 4, 6, 3])
    model.load_state_dict(torch.load(MODEL_PATH_IN))
    outputs = model(tensor)

    space_up_probability = outputs[0][0].item()
    space_down_probability = outputs[0][1].item()

    if space_up_probability > space_down_probability:
        return 0
    else:
        return 1
