import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import AccuracyBenchmark, device_map, get_llm_memory_usage


class QuantizedLinearLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, dtype=torch.float32, bits=8
    ):
        super().__init__()
        self.bits = bits

        self.type_mapping = {
            8: torch.int8,
            16: torch.int16,
            32: torch.int32,
        }

        self.register_buffer(
            "weight",
            torch.randint(
                -(2 ** (bits - 1)),
                2 ** (bits - 1),
                (out_features, in_features),
            ).to(self.type_mapping[bits]),
        )

        self.register_buffer(
            "scale",
            torch.randn((out_features), dtype=dtype),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.randn((1, out_features), dtype=dtype),
            )
        else:
            self.bias = None

    def quantize(self, weight):
        weight_f32 = weight.clone().to(torch.float32)

        Qmin = torch.iinfo(self.type_mapping[self.bits]).min
        Qmax = torch.iinfo(self.type_mapping[self.bits]).max

        scale = weight_f32.abs().max(dim=-1).values / ((Qmax - Qmin) // 2)
        scale = scale.to(weight.dtype)

        quantized_weight = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)), Qmin, Qmax
        ).to(self.type_mapping[self.bits])

        self.weight = quantized_weight
        self.scale = scale

    def forward(self, x):
        output = F.linear(x, self.weight.to(x.dtype)) * self.scale
        if self.bias is not None:
            output += self.bias
        return output


def replace_linear_layer(base_model, quantizer_class, exclude_list, quantized=True):
    for name, child in base_model.named_children():
        if name in exclude_list:
            continue
        if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
            old_bias = child.bias
            old_weight = child.weight
            in_features = child.in_features
            out_features = child.out_features

            quantizer_layer = quantizer_class(
                in_features,
                out_features,
                old_bias is not None,
                old_weight.dtype,
                bits=8,
            )

            setattr(base_model, name, quantizer_layer)

            if quantized:
                getattr(base_model, name).quantize(old_weight)

            if old_bias is not None:
                getattr(base_model, name).bias = old_bias

        else:
            replace_linear_layer(
                child, quantizer_class, exclude_list, quantized=quantized
            )


# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model memory usage: {get_llm_memory_usage(model) / 1024 ** 2:.2f} MB")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

benchmark = AccuracyBenchmark(model=model, tokenizer=tokenizer, dataset=dataset)

print(f"Original model accuracy: {benchmark.evaluate(sample_size=1000):.2f}")

replace_linear_layer(model, QuantizedLinearLayer, [], quantized=True)

print(f"Quantized model memory usage: {get_llm_memory_usage(model) / 1024 ** 2:.2f} MB")
print(f"Quantized model accuracy: {benchmark.evaluate(sample_size=1000):.2f}")
