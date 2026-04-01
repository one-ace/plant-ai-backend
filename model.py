import torch.nn as nn
from torchvision.models import resnet18

def get_model():
    model = resnet18(weights=None)  # no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model