import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_prompt(example, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        for fs_example in few_shot_examples:
            fs_question = fs_example["question"]
            fs_choices = fs_example["choices"]
            fs_answer = fs_example["answer"]
            prompt += f"Question: {fs_question}\nChoices: {', '.join(fs_choices)}\nAnswer: {fs_answer}\n\n"
    question = example["question"]
    choices = example["choices"]
    prompt += f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    return prompt


def evaluate_mmlu(
    model_name, task_name, split="validation", max_tokens=1024, few_shot=0, batch_size=8
):
    device = torch.device("mps")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Load the MMLU dataset
    dataset = load_dataset("cais/mmlu", task_name)
    train_dataset = dataset["auxiliary_train"] if "auxiliary_train" in dataset else None
    eval_dataset = dataset[split]

    # Prepare few-shot examples
    few_shot_examples = None
    if train_dataset and few_shot > 0:
        few_shot_examples = random.sample(
            list(train_dataset), min(few_shot, len(train_dataset))
        )

    # Prepare DataLoader for batching
    def collate_fn(batch):
        prompts = [prepare_prompt(ex, few_shot_examples) for ex in batch]
        return tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

    correct = 0
    total = 0

    # Evaluation loop with progress bar
    for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
        print(batch)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens
            )

        generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, example in enumerate(batch):
            generated_answer = generated_answers[i].strip()
            model_answer = generated_answer.split("\n")[-1].strip()
            print(f"Example: {example}")
            correct_answer = example["answer"]
            if model_answer == correct_answer:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Task: {task_name} | Accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MMLU with few-shot prompting"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="The name or path of the Hugging Face model to use.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="anatomy",
        help="The MMLU task to evaluate (e.g., 'high_school_geometry').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The dataset split to evaluate on (default: validation).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20,
        help="Maximum number of tokens for input/output.",
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=5,
        help="Number of few-shot examples to include in the prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for evaluation.",
    )

    args = parser.parse_args()

    evaluate_mmlu(
        model_name=args.model_name,
        task_name=args.task_name,
        split=args.split,
        max_tokens=args.max_tokens,
        few_shot=args.few_shot,
        batch_size=args.batch_size,
    )
