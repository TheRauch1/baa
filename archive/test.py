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

# from baa.mnist import MNIST
# from baa.quantizer import QuantizedLinearLayer, replace_linear_layer
#
# mnist = MNIST(device="cuda", batch_size=256, num_epochs=1, learning_rate=0.001)
# mnist.train()
# mnist.evaluate()
#
# print_memory_usage(mnist.model)
#
# replace_linear_layer(mnist.model, QuantizedLinearLayer, [], quantized=True)
#
# print_memory_usage(mnist.model)
#
# mnist.evaluate()


import torch


def quantize_weights_to_int_with_bitshifts(weights, num_bits=8):
    # Get the maximum absolute value of the weights
    max_val = torch.max(torch.abs(weights))

    # Compute the scale factor such that the max value is represented by an integer
    scale = max_val / (2 ** (num_bits - 1) - 1)

    # Calculate the bit shifts needed (i.e., log2 of the scale)
    bit_shifts = torch.round(torch.log2(scale)).to(torch.int)

    # Quantize weights by dividing with 2^bit_shifts and rounding to integers
    quantized_weights = torch.round(weights / (2.0**bit_shifts)).to(torch.int)

    return quantized_weights, bit_shifts


def dequantize_weights_with_bitshifts(quantized_weights, bit_shifts):
    # Dequantize weights by bit shifting
    # Bit shift left (i.e., multiply by 2^bit_shifts)
    # dequantized_weights = quantized_weights * (2.0 ** bit_shifts)
    if bit_shifts < 0:

        def shift(weights, bit_shifts):
            return torch.bitwise_left_shift(weights, bit_shifts, dtype=torch.float)
    else:

        def shift(weights, bit_shifts):
            return torch.bitwise_right_shift(weights, bit_shifts, dtype=torch.float)

    dequantized_weights = shift(quantized_weights, bit_shifts)
    return dequantized_weights


# Example weights tensor
weights = torch.tensor([0.3, 0.5, 1.2, -2.6, 4.1])

# Quantize the weights and get the number of bit shifts
quantized_weights, bit_shifts = quantize_weights_to_int_with_bitshifts(weights)

# Dequantize the weights using bit shifts
dequantized_weights = dequantize_weights_with_bitshifts(quantized_weights, bit_shifts)

print("Original weights:", weights)
print("Quantized weights (integers):", quantized_weights)
print("Number of bit shifts:", bit_shifts)
print("Dequantized weights:", dequantized_weights)
