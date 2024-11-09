import copy

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import AccuracyBenchmark, device_map
from baa.mnist import MNIST, Net
from baa.quantizer import Quantizer

load_dotenv()

mnist = MNIST(batch_size=128, num_epochs=1, learning_rate=0.001, device="cuda")
mnist.train()

evaluation_fn = mnist.evaluate
quantizer = Quantizer(evaluation_fn=evaluation_fn)


quantization_levels = [16, 12, 10, 8, 6, 5, 4, 3, 2]

error_threshold = 10

quantized_model, layer_quantization_info = quantizer.quantize_layer_independently(
    mnist.model, error_threshold, quantization_levels
)

print("\nLayerwise quantization info:")
for layer_name, (bit_width, error) in layer_quantization_info.items():
    print(f"Layer: {layer_name}, Bit width: {bit_width}, Error: {error} dB")

mnist.model = quantized_model
mnist.evaluate()
# benchmark.model = quantized_model
# acc = benchmark.evaluate(sample_size=100)
# print(f"Accuracy of quantized model: {acc}")
