# %%
import json
from typing import List, Literal

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from baa import SanityTextBenchmark

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
)

# %%
# model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
# model_name = "HuggingFaceTB/SmolLM-135M"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sanity_benchmark = SanityTextBenchmark(model, tokenizer)

print(sanity_benchmark.evaluate())
