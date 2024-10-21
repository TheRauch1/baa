from collections import OrderedDict
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from baa.singletons import hidden_states


class QuantizedLinearLayerWithActivation(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation_scale=None,
        bias=True,
        dtype=torch.float32,
        bits=8,
    ):
        super().__init__()
        self.bits = bits

        self.type_mapping = {
            8: torch.int8,
            16: torch.int16,
            32: torch.int32,
        }

        # self.qmin = torch.iinfo(self.type_mapping[bits]).min
        # self.qmax = torch.iinfo(self.type_mapping[bits]).max
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

        self.register_buffer(
            "weight",
            torch.randint(
                -(2 ** (bits - 1)),
                2 ** (bits - 1),
                (out_features, in_features),
            ),
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

        self.activation_scale = activation_scale

    def quantize(self, weight):
        weight_f32 = weight.clone().to(torch.float32)

        scale = weight_f32.abs().max(dim=-1).values / ((self.qmax - self.qmin) // 2)
        scale = scale.to(weight.dtype)

        quantized_weight = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)), self.qmin, self.qmax
        )

        self.weight = quantized_weight
        self.scale = scale

    def forward(self, x):
        if self.activation_scale is not None:
            x_int = torch.round(torch.div(x, self.activation_scale)).clamp(
                torch.iinfo(torch.int16).min,
                torch.iinfo(torch.int16).max,
            )
            assert x.shape == x_int.shape

            output_int = F.linear(x_int, self.weight)
            output = output_int * (self.activation_scale * self.scale)

        else:
            output = F.linear(x, self.weight) * self.scale
        if self.bias is not None:
            output += self.bias
        return output


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


def replace_linear_layer_with_activation(
    base_model, quantizer_class, hidden_states=None, exclude_list=[], quantized=True
):
    for name, child in base_model.named_children():
        if name in exclude_list:
            continue
        if isinstance(child, nn.Linear):
            assert hasattr(child, "custom_name")
            old_bias = child.bias
            old_weight = child.weight
            in_features = child.in_features
            out_features = child.out_features

            layer_scale = None
            if hidden_states is not None:
                custom_name = getattr(child, "custom_name")
                if custom_name in hidden_states:
                    layer_activations = hidden_states[custom_name]

                    layer_activations = torch.FloatTensor(layer_activations)
                    max_abs = layer_activations.mean().item()
                    bits = 16
                    Qmax = 2 ** (bits - 1) - 1
                    layer_scale = max_abs / Qmax

            quantizer_layer = quantizer_class(
                in_features,
                out_features,
                activation_scale=layer_scale,
                bias=old_bias is not None,
                dtype=old_weight.dtype,
                bits=5,
            )

            setattr(base_model, name, quantizer_layer)
            if quantized:
                getattr(base_model, name).quantize(old_weight)

            if old_bias is not None:
                getattr(base_model, name).bias = old_bias

        else:
            replace_linear_layer_with_activation(
                child, quantizer_class, hidden_states, exclude_list, quantized=quantized
            )


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


def add_custom_name_to_linear_layers(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            setattr(module, "custom_name", name)


def get_hidden_states_output(module, input, output):
    if isinstance(module, nn.Linear):
        layer_name = module.custom_name
        if layer_name not in hidden_states:
            hidden_states[layer_name] = []
        max_abs_activation = output.abs().max().item()
        hidden_states[layer_name].append(max_abs_activation)
    return hidden_states


def get_hidden_states_input(module, input, output):
    if isinstance(module, nn.Linear):
        try:
            layer_name = module.custom_name
        except AttributeError:
            add_custom_name_to_linear_layers(module)
            layer_name = module.custom_name
        if layer_name not in hidden_states:
            hidden_states[layer_name] = []
        max_abs_activation = input[0].abs().max().item()
        # max_abs_activation = input[0].to("cpu").numpy()
        hidden_states[layer_name].append(max_abs_activation)


def remove_all_hooks(model: nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)


def register_linear_layer_forward_hook(model, hook_fn):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(hook_fn)
