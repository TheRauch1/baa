import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x


class MNIST:
    def __init__(self, batch_size, num_epochs, learning_rate, device="cpu"):
        self.device = device
        self.model = Net().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train(self):
        print("Training the model...")
        for epoch in range(self.num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(
                        f"Train Epoch: {epoch+1} [{batch_idx*len(data)}/{len(self.train_loader.dataset)}]"
                        f"\tLoss: {loss.item():.6f}"
                    )

    def evaluate(self, model=None):
        if model is not None:
            self.model = model
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # if i == 0:
                #     print(f"Output: {output}")
                test_loss += criterion(output, target).item()  # Sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        print(
            f"\nModel Test set: Average loss: {test_loss:.6f}, Accuracy: {correct}/{len(self.test_loader.dataset)}"
            f" ({accuracy:.2f}%)\n"
        )
