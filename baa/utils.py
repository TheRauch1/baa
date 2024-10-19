def get_memory_usage(model):
    """Calculates the memory usage of a PyTorch model."""
    total_memory = 0
    for param in model.parameters():
        total_memory += param.element_size() * param.numel()
    return total_memory

def print_memory_usage(model):
    """Prints the memory usage of a PyTorch model."""
    # Define the memory units and their thresholds
    memory_units = {
        1024 ** 3: "GB",
        1024 ** 2: "MB",
        1024: "KB",
        1: "B"
    }

    # Get the model's memory usage
    memory_usage = get_memory_usage(model)

    # Loop through the memory_units dictionary to find the appropriate unit
    for threshold, unit in memory_units.items():
        if memory_usage >= threshold:
            print(f"Model memory usage: {memory_usage / threshold:.2f} {unit}")
            break
