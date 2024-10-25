import copy

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import AccuracyBenchmark, device_map
from baa.mnist import MNIST, Net

load_dotenv()

mnist = MNIST(device="cpu", batch_size=128, num_epochs=1, learning_rate=0.001)
mnist.train()
# model_name = "HuggingFaceTB/SmolLM-135M"
# llm = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# benchmark = AccuracyBenchmark(llm, tokenizer, dataset)
# print(f"Accuracy of model: {benchmark.evaluate(sample_size=100)}")


def quantize_tensor(tensor, bit_width):
    qmin = -(2 ** (bit_width - 1))
    qmax = 2 ** (bit_width - 1) - 1

    scale = tensor.abs().max() / qmax
    tensor_q = (tensor / scale).round().clamp(qmin, qmax)
    tensor_q = tensor_q * scale
    return tensor_q


def quantize_linear_layer(layer, bit_width):
    quantized_layer = copy.deepcopy(layer)

    quantized_weight = quantize_tensor(layer.weight.data, bit_width)
    quantized_layer.weight.data = quantized_weight

    if layer.bias is not None:
        quantized_bias = quantize_tensor(layer.bias.data, bit_width)
        quantized_layer.bias.data = quantized_bias

    return quantized_layer


def compute_layer_error(original_output, quantized_output):
    error = torch.nn.functional.mse_loss(
        original_output, quantized_output, reduction="mean"
    )
    return error


def replace_layer_in_model(model, layer_name, new_layer):
    modules = layer_name.split(".")
    parent_module = model
    for name in modules[:-1]:
        parent_module = getattr(parent_module, name)
    setattr(parent_module, modules[-1], new_layer)


def quantize_layer_independently(model: Net, error_threshold, quantization_levels):
    quantized_model = copy.deepcopy(model)
    layer_quantization_info = {}

    layer_name_to_module = {name: module for name, module in model.named_modules()}
    layer_inputs_outputs = {}

    hooks = []

    def register_hooks():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(
                module, torch.nn.Conv2d
            ):

                def create_hook(name):
                    def hook(module, input, output):
                        layer_inputs_outputs[name] = (
                            input[0].detach(),
                            output.detach(),
                        )

                    return hook

                hooks.append(module.register_forward_hook(create_hook(name)))

    register_hooks()

    with torch.no_grad():
        mnist.model = model
        mnist.evaluate()
        # benchmark.model = model
        # benchmark.evaluate(sample_size=100)

    for h in hooks:
        h.remove()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            print(f"Quantizing layer: {name}")
            original_input, original_output = layer_inputs_outputs[name]

            min_error = float("inf")
            best_bit_width = None
            best_quantized_layer = None

            for bit_width in quantization_levels:
                quantized_layer = quantize_linear_layer(module, bit_width)
                quantized_output = quantized_layer(original_input)
                error = compute_layer_error(original_output, quantized_output)

                if error <= error_threshold:
                    min_error = error
                    best_bit_width = bit_width
                    best_quantized_layer = quantized_layer
                else:
                    continue

            if best_quantized_layer is not None:
                replace_layer_in_model(quantized_model, name, best_quantized_layer)
                layer_quantization_info[name] = (best_bit_width, min_error)

            else:
                print(f"Could not quantize layer {name} withing error threshold")

    return quantized_model, layer_quantization_info


quantization_levels = [16, 12, 10, 8, 6, 5, 4, 3, 2]

error_threshold = 1e-1

quantized_model, layer_quantization_info = quantize_layer_independently(
    mnist.model, error_threshold, quantization_levels
)

print("\nLayerwise quantization info:")
for layer_name, (bit_width, error) in layer_quantization_info.items():
    print(f"Layer: {layer_name}, Bit width: {bit_width}, Error: {error}")

mnist.model = quantized_model
mnist.evaluate()
# benchmark.model = quantized_model
# acc = benchmark.evaluate(sample_size=100)
# print(f"Accuracy of quantized model: {acc}")
