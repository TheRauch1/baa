import torch

from baa import print_memory_usage

# matrixA = torch.tensor([[1, 2], [3, 4]])
matrixA = torch.randint(10, (2, 2))
# scaleA = torch.tensor([1.5])
scaleA = torch.randint(10, (1, 1))

# matrixB = torch.tensor([[5, 6], [7, 8]])
matrixB = torch.randint(10, (2, 2))
# scaleB = torch.tensor([2.5])
scaleB = torch.randint(10, (1, 1))

print(matrixA * scaleA)
print(matrixB * scaleB)
print((matrixA * scaleA) @ (matrixB * scaleB))
print((matrixA @ matrixB) * scaleA * scaleB)

from baa.mnist import MNIST
from baa.quantizer import QuantizedLinearLayer, replace_linear_layer

mnist = MNIST(device="cuda", batch_size=256, num_epochs=1, learning_rate=0.001)
mnist.train()
mnist.evaluate()

print_memory_usage(mnist.model)

replace_linear_layer(mnist.model, QuantizedLinearLayer, [], quantized=True)

print_memory_usage(mnist.model)

mnist.evaluate()
