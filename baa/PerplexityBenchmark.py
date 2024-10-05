from tqdm import tqdm
import torch


class PerplexityBenchmark:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def evaluate(self, max_length=2048, stride=512, sample_size=None):
        self.dataset = (
            self.dataset.select(range(sample_size)) if sample_size else self.dataset
        )
        encodings = self.tokenizer(
            "\n\n".join(self.dataset["text"]),
            return_tensors="pt",
        )
        seq_len = encodings["input_ids"].shape[1]
        negative_log_likelihoods = []
        prev_end_idx = 0
        for begin_idx in tqdm(range(0, seq_len, stride)):
            end_idx = min(begin_idx + max_length, seq_len)
            target_len = end_idx - prev_end_idx
            input_ids = encodings["input_ids"][:, begin_idx:end_idx].to(
                self.model.device
            )
            target_ids = input_ids.clone()
            target_ids[:, :-target_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                neg_log_likelihood = outputs.loss

            negative_log_likelihoods.append(neg_log_likelihood.item())

            prev_end_idx = end_idx
            if end_idx == seq_len:
                break

        perplexity = torch.exp(torch.tensor(negative_log_likelihoods).mean()).item()
        return perplexity
