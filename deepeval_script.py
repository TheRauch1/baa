# %%
import asyncio
import json
from typing import List, Literal

import torch
import transformers
from deepeval.benchmarks import MMLU, HellaSwag, TruthfulQA
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.benchmarks.tasks import HellaSwagTask, MMLUTask, TruthfulQATask
from deepeval.models.base_model import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %%
seed = 42
transformers.set_seed(seed)

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# %%
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# %%
from typing import List


class CustomMultipleChoiceSchema(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class Llama(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()

        with torch.no_grad():
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                use_cache=True,
                device_map="auto",
                max_new_tokens=100,
                do_sample=True,
                top_k=5,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )

            output_dict = pipeline(
                prompt,
                prefix_allowed_tokens_fn=prefix_function,
            )
            output = output_dict[0]["generated_text"][len(prompt) :]
            # print(output)
            try:
                json_result = json.loads(output)
            except json.JSONDecodeError:
                print("Error decoding JSON")
                return schema(**{"answer": ""})

            return schema(**json_result)

    def batch_generate(self, prompt: List[str]) -> List[str]:
        model = self.load_model()

        with torch.no_grad():
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                use_cache=True,
                device_map="auto",
                max_new_tokens=100,
                # do_sample=True,
                # top_k=5,
                num_return_sequences=1,
                batch_size=len(prompt),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            try:
                pipeline.tokenizer.pad_token_id = model.config.eos_token_id[0]
            except:
                pipeline.tokenizer.pad_token_id = model.config.eos_token_id

            schema = CustomMultipleChoiceSchema

            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )

            output_dict = pipeline(
                prompt,
                prefix_allowed_tokens_fn=prefix_function,
            )

            output = []
            for i in range(len(prompt)):
                output_text = output_dict[i][0]["generated_text"][len(prompt[i]) :]
                try:
                    json_result = json.loads(output_text)
                except json.JSONDecodeError:
                    print("Error decoding JSON")
                    output.append("")
                    continue

                output.append(json_result["answer"])

            return output

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3.2-1B-Instruct"


# %%
llama = Llama(model, tokenizer)
llama.batch_generate(["Hi there", "How are you", "What is your name"])
# print(llama.generate("Tell a short story of humanity with happy ending"))

# %%
benchmark = MMLU(
    tasks=[
        MMLUTask.BUSINESS_ETHICS,
        MMLUTask.MEDICAL_GENETICS,
        MMLUTask.FORMAL_LOGIC,
    ],
    n_shots=5,
)
# benchmark = MMLU(n_shots=5)
# benchmark = MMLU(n_shots=2)
# benchmark = TruthfulQA(tasks=[TruthfulQATask.ADVERTISING], mode=TruthfulQAMode.MC2)
# benchmark = HellaSwag(n_shots=3)
results = benchmark.evaluate(llama, batch_size=16)
preds = benchmark.predictions
print(results)

preds_correct = preds[preds["Correct"] == True]
preds_incorrect = preds[preds["Correct"] == False]

print(f"accuracy: {len(preds_correct) / len(preds)}")

# %%
preds_empty = preds[preds["Prediction"] == ""]
print(f"empty: {len(preds_empty)}")

# benchmark = HellaSwag(tasks=[HellaSwagTask.], n_shots=5)
# results = benchmark.evaluate(llama, batch_size=32)
# preds = benchmark.predictions
# print(results)

# preds_correct = preds[preds["Correct"] == True]
# preds_incorrect = preds[preds["Correct"] == False]

# print(f"accuracy: {len(preds_correct) / len(preds)}")

# %%
