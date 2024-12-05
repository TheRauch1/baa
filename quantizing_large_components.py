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
from baa import (
    LLMAccuracyBenchmark,
    QuantizedLinearLayerWithActivation,
    add_custom_name_to_linear_layers,
    remove_all_hooks,
    replace_linear_layer_with_activation,
)

load_dotenv()

# Initialize WandB
wandb.init()
# Retrieve configuration
config = wandb.config
include_component = config.include_component

# Load model and tokenizer
model_name = config.model_name
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
original_device = model.device
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
benchmark = LLMAccuracyBenchmark(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    sequence_length=512,
    num_samples=300,
    batch_size=1,
)


with torch.no_grad():
    add_custom_name_to_linear_layers(model)
    original_model_accuracy = benchmark.evaluate()
    include_list = {
        "Full_Model": [],
        "Self_Attention": [
            layer.custom_name
            for layer in model.named_modudes
            if "self_attn" not in layer.custom_name
        ],
        "MLP": [
            layer.custom_name
            for layer in model.named_modules
            if "mlp" not in layer.custom_name
        ],
        "LM_Head": [
            layer.custom_name
            for layer in model.named_modules
            if "lm_head" not in layer.custom_name
        ],
    }
    exclude_list = include_list[include_component]
    replace_linear_layer_with_activation(
        base_model=model,
        quantizer_class=QuantizedLinearLayerWithActivation,
        weight_bits=config.weight_bits,
        activation_bits=16,
        exclude_list=exclude_list,
        quantized=True,
    )
    remove_all_hooks(model)

    quantized_model_accuracy = benchmark.evaluate()

# Write log
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_name = os.path.join(
    "logs",
    "first_experiment",
    (
        f"{model_name.replace('/', '_')}"
        + f"_quantization_{include_component}"
        + f"{config.weight_bits}bits"
        + f"_{date_time}.json"
    ),
)

log = {
    "model_name": model_name,
    "original_model_accuracy": original_model_accuracy,
    "include_component": include_component,
    "weight_bits": config.weight_bits,
    "quantized_model_accuracy": quantized_model_accuracy,
}

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_name), exist_ok=True)
with open(log_name, "w", encoding="utf-8") as f:
    json.dump(log, f, indent=4)

# Log results to WandB
wandb.log(log)

# Save the log as an artifact in WandB
artifact = wandb.Artifact(f"quantization_log_{date_time}", type="evaluation")
artifact.add_file(log_name)
wandb.log_artifact(artifact)

wandb.finish()
