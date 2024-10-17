import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_epochs = 1  # Adjust as needed
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()


def train():
    print("Training the model...")
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}]"
                    f"\tLoss: {loss.item():.6f}"
                )


# check if 'mnist_model.pth' exists in the current directory
if not os.path.exists("mnist_model.pth"):
    train()
    torch.save(model.state_dict(), "mnist_model.pth")
else:
    model.load_state_dict(torch.load("mnist_model.pth"))
    print("Model loaded from disk")


def evaluate(model):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # if i == 0:
            #     print(f"Output: {output}")
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\Model Test set: Average loss: {test_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({accuracy:.2f}%)\n"
    )


# Evaluate the model
print("Evaluating the model...")
evaluate(model)


def quant_w(weights: torch.Tensor) -> torch.Tensor:
    """
    Quantize the weights to 8-bit integers
    """
    assert isinstance(weights, torch.Tensor)
    max_val = torch.max(torch.abs(weights))
    min_val = torch.min(torch.abs(weights))
    # Calculate the scale factor
    scale = (max_val - min_val) / 255.0
    # Quantize the weights
    weights = torch.round(weights / scale)  # Quantize the weights to 8-bit integers
    # Clip the weights to the range [-128, 127]
    weights = torch.clamp(weights, -128, 127)

    # weights = weights.to(torch.int8)
    return weights


def quant_a(module, input, output):
    """
    Quantize the activations to 8-bit integers
    """
    assert isinstance(output, torch.Tensor)
    max_val = torch.max(torch.abs(output))
    min_val = torch.min(torch.abs(output))
    # Calculate the scale factor
    scale = (max_val - min_val) / 255.0
    # Quantize the activations
    output = torch.round(output / scale)  # Quantize the weights to 8-bit integers
    # Clip the activations to the range [-128, 127]
    output = torch.clamp(output, -128, 127)

    # output = output.to(torch.int8)
    return output


with torch.no_grad():
    for param in model.parameters():
        # param.requires_grad_(False)
        param.data = quant_w(param.data)
        # quantize activation before passing to the next layer by injecting with register_pre_forward_hook
        # param.register_hook(quant_a)

    # Evaluate the quantized model
    print("Evaluating the quantized model...")
    # print head of fc1 weights
    print(model.fc1.weight[0][:10])
    evaluate(model)
