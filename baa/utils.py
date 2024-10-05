def get_llm_memory_usage(model):
    """Calculates the memory usage of a PyTorch model."""
    total_memory = 0
    for param in model.parameters():
        total_memory += param.element_size() * param.numel()
    return total_memory
