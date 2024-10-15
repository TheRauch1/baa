# implementing a mnist model trainer with pytorch using fixed point quantization and STE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MNIST dataset
train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000, shuffle=True)

def activation_quant(x, num_bits=8):
    scale = 2 ** num_bits - 1 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)   
    y = (x * scale).round().clamp_(-2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1)
    return y

def weight_quant(w, num_bits=8):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    # u = (w * scale).round().clamp_(-1, 1) 
    return u

class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        rmsnorm = nn.RMSNorm(x.shape[1]).to(x.device)
        x_norm = rmsnorm(x)
        # Trick to implement STE using .detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        # Print device of x_quant and w_quant
        return F.linear(x_quant, w_quant)

# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.hidden = BitLinear(784, 128)
        self.output = BitLinear(128, 10)

    def forward(self, x):
        assert x.device == self.hidden.weight.device
        x = x.view(-1, 784)
        x = F.relu(self.hidden(x))
        return self.output(x)
    
model = MNISTModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Define the training function
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Define the test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Train the model
for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# Print distinct weight values to check there should only be 3 distinct values
print(model.hidden.weight)

# Save the model
torch.save(model.state_dict(), 'mnist_ste.pth')
print('Saved the model to mnist_ste.pth')