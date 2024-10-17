import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from baa import AccuracyBenchmark, PerplexityBenchmark, device_map, get_llm_memory_usage

# model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model memory usage: {get_llm_memory_usage(model) / 1024 ** 2:.2f} MB")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

benchmark = AccuracyBenchmark(model=model, tokenizer=tokenizer, dataset=dataset)

print(f"Original model accuracy: {benchmark.evaluate(sample_size=200):.2f}")


def quant_w(weights: torch.Tensor, bits: int = 16) -> torch.Tensor:
    """
    Quantize the weights to 8-bit integers
    """
    assert isinstance(weights, torch.Tensor)
    assert isinstance(bits, int) and bits > 0

    max_val = torch.max(torch.abs(weights))
    min_val = torch.min(torch.abs(weights))

    # Calculate the scale factor
    scale = (max_val - min_val) / (2**bits - 1)

    # Quantize the weights
    weights = torch.round(weights / scale)
    # weights = torch.round(weights / scale, decimals=4)

    # Calculate the clamping range
    clamp_min = -(2 ** (bits - 1))
    clamp_max = (2 ** (bits - 1)) - 1

    # Clip the weights to the range [clamp_min, clamp_max]
    weights = torch.clamp(weights, clamp_min, clamp_max)
    return weights


for param in model.model.layers[0].self_attn.q_proj.parameters():
    param.data = quant_w(param.data)


print(f"Quantized model accuracy: {benchmark.evaluate(sample_size=200):.2f}")
