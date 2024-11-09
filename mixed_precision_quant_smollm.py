import copy

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import AccuracyBenchmark, device_map
from baa.mnist import MNIST, Net
from baa.quantizer import Quantizer

load_dotenv()

model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
benchmark = AccuracyBenchmark(model, tokenizer, dataset)


def evaluation_fn(model):
    benchmark.model = model
    benchmark.evaluate(sample_size=200)


quantizer = Quantizer(evaluation_fn=evaluation_fn)


quantization_levels = [16, 12, 10, 8, 6, 5, 4, 3, 2]

error_threshold = 15

quantized_model, layer_quantization_info = quantizer.quantize_layer_independently(
    model, error_threshold, quantization_levels
)

print("\nLayerwise quantization info:")
for layer_name, (bit_width, error) in layer_quantization_info.items():
    print(f"Layer: {layer_name}, Bit width: {bit_width}, Error: {error} dB")

average_bit_width = sum(
    [bit_width for bit_width, _ in layer_quantization_info.values()]
) / len(layer_quantization_info)

print(f"Average bit width: {average_bit_width}")
benchmark.model = quantized_model
acc = benchmark.evaluate(sample_size=200)
print(f"Accuracy of quantized model: {acc}")
