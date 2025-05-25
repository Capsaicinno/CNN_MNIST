import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

class SimpleCNN(nn.Module):
    def__init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc_layers(x)
        return x
def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
def load_data(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader
def visualize_predictions(model, data           _loader):
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            plt.imshow(data[0].squeeze(), cmap='gray')
            plt.title(f'Predicted: {pred.item()}, Actual: {target.item()}')
            plt.show()
            break  # Show only one example
def main():
    model_path = 'model.pth'  # Path to your trained model
    model = load_model(model_path)
    data_loader = load_data(batch_size=1)
    visualize_predictions(model, data_loader)
if __name__ == "__main__":
    main()
#     )
#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)         