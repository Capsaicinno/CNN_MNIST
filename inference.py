import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_Layers = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self, x):
        x =self.conv_layers(x)
        x = x.view(-1, 32*7*7)
        x = self.fc_Layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

random_idx = random.randint(0, len(test_dataset) - 1)
image, true_label = test_dataset[random_idx]

input_image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_image)
    predicted_label = output.argmax(dim=1).item()


image_unnorm =image * 0.3081 + 0.1307

image_np = image_unnorm.squeeze().cpu().numpy()

plt.figure(figsize=(4,4))
plt.imshow(image_np, cmap='gray')
plt.axis('off')
plt.text(0.95,0.95, f'{predicted_label}',
        transform =plt.gca().transAxes,
        color='red', fontsize=20, fontweight='bold',
        horizontalalignment='right', verticalalignment='top')
plt.show()
print("show")