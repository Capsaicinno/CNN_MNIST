import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module): #nn.module is the base class for all neural network modules in PyTorch
    #define init and forward methods
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(16,32,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
        self.fc_Layers = nn.Sequential(
        nn.Linear(32*7*7,128),
        nn.ReLU(),
        nn.Linear(128,10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1,32 * 7 * 7) #Flatten the tensor
        x = self.fc_Layers(x)
        return x
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx *len(data)}/{len(loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, loader, criterion):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.10f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n')
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ))
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        test(model, test_loader, criterion)
        
    torch.save(model.state_dict(), 'mnist_cnn.pth')