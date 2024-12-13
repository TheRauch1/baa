import copy
import datetime
import gc
import json
import os
import shutil

import torch
import transformers
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from baa import (
    LLMAccuracyBenchmark,
    MMLUBenchmark,
    SanityTextBenchmark,
    seed,
)

transformers.set_seed(seed)
load_dotenv()


# Initialize WandB
wandb.init()
# Retrieve configuration

# Load model and tokenizer
model_name = config.model_name
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
original_device = model.device
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")


def evaluation_fn(model):
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-v1",
        split="test",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    )
    wikitext_benchmark = LLMAccuracyBenchmark(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        sequence_length=512,
        num_samples=300,
        batch_size=1,
    )
    wikitext_accuracy = wikitext_benchmark.evaluate()
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    mmlu_benchmark = MMLUBenchmark(
        model=model, tokenizer=tokenizer, model_name=model_name
    )
    mmlu_results = mmlu_benchmark.evaluate()
    del mmlu_benchmark
    gc.collect()
    torch.cuda.empty_cache()

    sanity_check_benchmark = SanityTextBenchmark(model, tokenizer)
    sanity_check_string = sanity_check_benchmark.evaluate()
    del sanity_check_benchmark
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "wikitext_accuracy": wikitext_accuracy,
        "mmlu_results": mmlu_results,
        "sanity_check_string": sanity_check_string,
    }


with torch.no_grad():
    original_model_benchmarks = evaluation_fn(model)

# Write log
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_name = os.path.join(
    "logs",
    "original_model_benchmarks",
    (
        f"{model_name.replace('/', '_')}"
        + f"_original_model_benchmarks"
        + f"_{date_time}.json"
    ),
)

log = {
    "model_name": model_name,
    "original_model_benchmarks": original_model_benchmarks,
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
