import textwrap


def get_memory_usage(model):
    """Calculates the memory usage of a PyTorch model."""
    total_memory = 0
    for param in model.parameters():
        total_memory += param.element_size() * param.numel()
    return total_memory


def print_memory_usage(model):
    """Prints the memory usage of a PyTorch model."""
    # Define the memory units and their thresholds
    memory_units = {1024**3: "GB", 1024**2: "MB", 1024: "KB", 1: "B"}

    # Get the model's memory usage
    memory_usage = get_memory_usage(model)

    # Loop through the memory_units dictionary to find the appropriate unit
    for threshold, unit in memory_units.items():
        if memory_usage >= threshold:
            print(f"Model memory usage: {memory_usage / threshold:.2f} {unit}")
            break


def _beautify_text(text):
    print("Generated Output:\n")
    for i, sentence in enumerate(text, 1):
        wrapped_sentence = textwrap.fill(sentence, width=80)
        print(f"Output {i}:\n{wrapped_sentence}\n")


def chat_with_model(model, tokenizer, prompt, max_new_tokens=100, beautify=True):
    """Chat with a model using a given prompt."""
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate a response from the model
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Decode the response
    response = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Beautify the response
    if beautify:
        _beautify_text(response)

    return response
