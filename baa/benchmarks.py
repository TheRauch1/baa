# class AccuracyBenchmark:
#     def __init__(self, model, tokenizer, dataset):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.dataset = dataset
#     def evaluate(self, max_length=2048, stride=512, sample_size=None):
#         # Optionally sample a subset of the dataset
#         if sample_size:
#             self.dataset = self.dataset.select(range(sample_size))
#         # Tokenize the dataset
#         encodings = self.tokenizer(
#             "\n\n".join(self.dataset["text"]),
#             return_tensors="pt",
#         )
#         seq_len = encodings["input_ids"].size(1)
#         total_correct = 0
#         total_tokens = 0
#         prev_end_idx = 0
#         # Loop over the dataset with the specified stride
#         for begin_idx in tqdm(range(0, seq_len, stride)):
#             end_idx = min(begin_idx + max_length, seq_len)
#             target_len = end_idx - prev_end_idx
#             input_ids = encodings["input_ids"][:, begin_idx:end_idx].to(
#                 self.model.device
#             )
#             # Shift inputs and labels for causal language modeling
#             inputs = input_ids[:, :-1]
#             labels = input_ids[:, 1:].clone()
#             # Mask labels to ignore overlapping tokens
#             labels[:, :-target_len] = -100
#             with torch.no_grad():
#                 outputs = self.model(inputs)
#                 logits = outputs.logits
#             # Get predictions and compute accuracy
#             predictions = torch.argmax(logits, dim=-1)
#             mask = labels != -100
#             correct = (predictions == labels) & mask
#             total_correct += correct.sum().item()
#             total_tokens += mask.sum().item()
#             del input_ids, inputs, labels, outputs, logits, predictions, mask, correct
#             gc.collect()
#             torch.cuda.empty_cache()
#             prev_end_idx = end_idx
#             if end_idx == seq_len:
#                 break
#         accuracy = total_correct / total_tokens if total_tokens > 0 else 0
#         return accuracy
import json

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .constants import seed
from .utils import chat_with_model

transformers.set_seed(seed)


class LLMAccuracyBenchmark:
    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        sequence_length: int = 512,
        num_samples: int = 1000,
        batch_size: int = 8,
    ):
        """
        Initializes the LLMAccuracyBenchmark class.

        Args:
            model (PreTrainedModel): Model to evaluate.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            dataset (Dataset): Dataset to evaluate on.
            sequence_length (int): The desired sequence length.
            num_samples (int): The number of samples to use.
            batch_size (int): Batch size for evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.input_ids = None

        # Load and prepare data
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares the data by tokenizing and chunking the text into sequences.
        """
        # Load the dataset
        # Select a subset of the dataset
        texts = self.dataset["text"][: self.num_samples]
        # Tokenize and chunk the texts into sequences of the specified length
        encodings = self.tokenizer("\n\n".join(texts), return_tensors="pt")
        input_ids = encodings["input_ids"][0]
        total_length = input_ids.size(0)
        num_sequences = total_length // self.sequence_length
        input_ids = input_ids[: num_sequences * self.sequence_length]
        input_ids = input_ids.view(num_sequences, self.sequence_length)
        self.input_ids = input_ids

    def evaluate(self):
        """
        Evaluates the model on the prepared data and prints the token-level accuracy.
        """
        if self.input_ids is None:
            raise ValueError("Data not prepared.")
        # self.model.eval()
        dataset = TensorDataset(self.input_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                try:
                    input_ids = batch[0].to(self.model.device)
                except:
                    input_ids = batch[0]
                # Prepare inputs and labels by shifting the input_ids
                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                outputs = self.model(inputs)
                logits = (
                    outputs.logits
                )  # shape: (batch_size, seq_length - 1, vocab_size)
                predictions = torch.argmax(logits, dim=-1)
                # Compare predictions with labels
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        accuracy = correct / total
        print(f"Token-level accuracy: {accuracy:.4f}")
        return accuracy


from typing import List, Literal

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel


class CustomMultipleChoiceSchema(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class CustomDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

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

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        with torch.no_grad():
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                use_cache=True,
                device_map="auto",
                max_new_tokens=100,
                num_return_sequences=1,
                batch_size=len(prompts),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            try:
                pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id[0]
            except:
                pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

            schema = CustomMultipleChoiceSchema

            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )

            output_dict = pipeline(
                prompts,
                prefix_allowed_tokens_fn=prefix_function,
            )

            output = []
            for i in range(len(prompts)):
                output_text = output_dict[i][0]["generated_text"][len(prompts[i]) :]
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
        return self.model_name


class MMLUBenchmark:
    def __init__(self, model, tokenizer, model_name):
        model.eval()
        self.model = CustomDeepEvalModel(model, tokenizer, model_name)
        self.tokenizer = tokenizer

    def evaluate(self):
        benchmark = MMLU(
            tasks=[
                MMLUTask.BUSINESS_ETHICS,
                MMLUTask.MEDICAL_GENETICS,
                MMLUTask.FORMAL_LOGIC,
            ],
            n_shots=5,
        )
        overall_score = benchmark.evaluate(self.model, batch_size=16)
        results = {
            "overall_score": overall_score,
            "task_scores": benchmark.task_scores.to_json(),
        }
        return results


class SanityTextBenchmark:
    def __init__(self, model: AutoModelForCausalLM, tokenizer, tokens_to_generate=100):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.tokens_to_generate = tokens_to_generate

        self.prompt = "Tell a short story of humanity with happy ending"

    def evaluate(self):
        with torch.no_grad():
            messages = [{"role": "user", "content": self.prompt}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.tokens_to_generate,
                top_p=0.9,
                do_sample=True,
            )
            text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text_output