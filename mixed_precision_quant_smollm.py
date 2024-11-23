import copy
import datetime
import gc
import json
import os
import shutil

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from baa import LLMAccuracyBenchmark, device_map
from baa.mnist import MNIST, Net
from baa.quantizer import Quantizer

load_dotenv()

# Initialize WandB
wandb.init()
# Retrieve configuration
config = wandb.config
min_quantile, max_quantile = config.quantile_range

# Load model and tokenizer
model_name = config.model_name
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
        num_samples=300,
        batch_size=1,
    )
    return benchmark.evaluate()


# Initialize quantizer with WandB parameters
quantizer = Quantizer(
    evaluation_fn=evaluation_fn,
    min_quantile=min_quantile,
    max_quantile=max_quantile,
)

quantization_levels = [16, 12, 10, 8, 6, 5, 4, 3, 2]
error_threshold = config.error_threshold

layer_quantization_info, original_model_accuracy = (
    quantizer.quantize_layer_independently(model, error_threshold, quantization_levels)
)

# if dir already exists, delete it
if os.path.exists("/tmp/quantized_model"):
    shutil.rmtree("/tmp/quantized_model")
model.save_pretrained("/tmp/quantized_model")

# Log layerwise quantization info
print("\nLayerwise quantization info:")
for layer_name, (bit_width, error) in layer_quantization_info.items():
    print(f"Layer: {layer_name}, Bit width: {bit_width}, Error: {error} dB")

average_bit_width = sum(
    [bit_width for bit_width, _ in layer_quantization_info.values()]
) / len(layer_quantization_info)

print(f"Average bit width: {average_bit_width}")

# Clear resources and reload model
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained("/tmp/quantized_model", device_map="auto")
quantized_model_accuracy = evaluation_fn(model)

# delete the temporary model
shutil.rmtree("/tmp/quantized_model")

# Write log
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_name = os.path.join(
    "logs",
    (
        f"{model_name.replace('/', '_')}"
        + f"_quantization_{error_threshold}dB"
        + f"_min-{quantizer.min_quantile}_max-{quantizer.max_quantile}"
        + f"_{date_time}.json"
    ),
)

log = {
    "model_name": model_name,
    "original_model_accuracy": original_model_accuracy,
    "layerwise_quantization_info": {
        layer_name: {"bit_width": bit_width, "error": error}
        for layer_name, (bit_width, error) in layer_quantization_info.items()
    },
    "average_bit_width": average_bit_width,
    "error_threshold": error_threshold,
    "min_quantile": quantizer.min_quantile,
    "max_quantile": quantizer.max_quantile,
    "quantized_model_accuracy": quantized_model_accuracy,
}

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
with open(log_name, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=4)

# Log results to WandB
wandb.log(log)

# Save the log as an artifact in WandB
artifact = wandb.Artifact(f"quantization_log_{date_time}", type="evaluation")
artifact.add_file(log_name)
wandb.log_artifact(artifact)

wandb.finish()
