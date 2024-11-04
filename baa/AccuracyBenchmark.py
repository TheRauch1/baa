import gc

import torch
from tqdm import tqdm


class AccuracyBenchmark:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def evaluate(self, max_length=2048, stride=512, sample_size=None):
        # Optionally sample a subset of the dataset
        if sample_size:
            self.dataset = self.dataset.select(range(sample_size))

        # Tokenize the dataset
        encodings = self.tokenizer(
            "\n\n".join(self.dataset["text"]),
            return_tensors="pt",
        )
        seq_len = encodings["input_ids"].size(1)
        total_correct = 0
        total_tokens = 0
        prev_end_idx = 0

        # Loop over the dataset with the specified stride
        for begin_idx in tqdm(range(0, seq_len, stride)):
            end_idx = min(begin_idx + max_length, seq_len)
            target_len = end_idx - prev_end_idx
            input_ids = encodings["input_ids"][:, begin_idx:end_idx].to(
                self.model.device
            )

            # Shift inputs and labels for causal language modeling
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:].clone()

            # Mask labels to ignore overlapping tokens
            labels[:, :-target_len] = -100

            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits

            # Get predictions and compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            del input_ids, inputs, labels, outputs, logits, predictions, mask, correct
            gc.collect()
            torch.cuda.empty_cache()

            prev_end_idx = end_idx
            if end_idx == seq_len:
                break

        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        return accuracy
