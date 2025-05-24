import torch
import torch.nn as nn
import torch.optim as optimfrom torchvision import datasets, transforms
from torch.utils.data import DataLoader
import maplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(16,32,kenel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    self.fc_Layers = nn.Seqential(
        nn.Linear(32*7*7,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1,32 * 7 * 7)
        x = self.fc_layers(x)
        return xi

    def test(model, loader, criterion)
    