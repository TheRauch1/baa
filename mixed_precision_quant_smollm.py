import copy
import gc
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import LLMAccuracyBenchmark, device_map
from baa.mnist import MNIST, Net
from baa.quantizer import Quantizer

load_dotenv()

# model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
original_device = model.device
tokenizer = AutoTokenizer.from_pretrained(model_name)


def evaluation_fn(model):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    benchmark = LLMAccuracyBenchmark(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        sequence_length=512,
        num_samples=100,
        batch_size=1,
    )
    print(benchmark.evaluate())


quantizer = Quantizer(evaluation_fn=evaluation_fn)

quantization_levels = [16, 12, 10, 8, 6, 5, 4, 3, 2]
# quantization_levels = [5, 4]

error_threshold = 20

layer_quantization_info = quantizer.quantize_layer_independently(
    model, error_threshold, quantization_levels
)

model.save_pretrained("/tmp/quantized_model")

print("\nLayerwise quantization info:")
for layer_name, (bit_width, error) in layer_quantization_info.items():
    print(f"Layer: {layer_name}, Bit width: {bit_width}, Error: {error} dB")

average_bit_width = sum(
    [bit_width for bit_width, _ in layer_quantization_info.values()]
) / len(layer_quantization_info)

print(f"Average bit width: {average_bit_width}")
# benchmark.model = quantized_model
# evaluation_fn(model.to(original_device))
# model.model.embed_tokens.to("cuda")
# reapply original devices
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained("/tmp/quantized_model", device_map="auto")
evaluation_fn(model)
# acc = benchmark.evaluate(sample_size=100)
# print(f"Accuracy of quantized model: {acc}")
