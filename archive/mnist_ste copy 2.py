import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Import quantization modules
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import prepare_qat, convert

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
num_epochs = 1  # Adjust as needed
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the model with linear layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        # Quantize the input
        x = self.quant(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Dequantize the output
        x = self.dequant(x)
        return x

# Instantiate the model
model = Net().to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Prepare the model for quantization-aware training
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
prepare_qat(model, inplace=True)

print("Training the quantization-aware model...")
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'QAT Train Epoch: {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}]'
                  f'\tLoss: {loss.item():.6f}')

# Convert the model to a quantized version
model = model.to('cpu')
model.eval()
convert(model, inplace=True)

# Function to evaluate the model
def evaluate(model):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nQuantized Model Test set: Average loss: {test_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')

# Evaluate the quantized model
print("Evaluating the quantized model...")
device = 'cpu'
evaluate(model)
print("Quantized model evaluation complete.")

