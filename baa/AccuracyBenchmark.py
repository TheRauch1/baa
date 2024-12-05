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


import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


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
                input_ids = batch[0].to(self.model.device)
                # input_ids = batch[0]
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
