import copy
import gc
from collections import OrderedDict
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from baa.singletons import hidden_states, names


class QuantizedLinearLayerWithActivation(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation_scale=None,
        bias=True,
        dtype=torch.float32,
        weight_bits=8,
        activation_bits=16,
    ):
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

        self.type_mapping = {
            8: torch.int8,
            16: torch.int16,
            32: torch.int32,
        }

        # self.qmin = torch.iinfo(self.type_mapping[bits]).min
        # self.qmax = torch.iinfo(self.type_mapping[bits]).max
        self.weight_qmin = -(2 ** (self.weight_bits - 1))
        self.weight_qmax = 2 ** (self.weight_bits - 1) - 1
        self.activation_qmin = -(2 ** (activation_bits - 1))
        self.activation_qmax = 2 ** (activation_bits - 1) - 1

        self.register_buffer(
            "weight",
            torch.randint(
                self.weight_qmin,
                self.weight_qmax,
                (out_features, in_features),
            ),
        )

        self.register_buffer(
            "scale",
            torch.randn((out_features), dtype=dtype),
        )

        self.register_buffer(
            "zero_point",
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
        weight_f32 = weight.detach().clone().to(torch.float32)

        scale = (weight_f32.max(dim=-1).values - weight_f32.min(dim=-1).values) / (
            self.weight_qmax - self.weight_qmin
        )
        zero_point = self.weight_qmin - weight_f32.min(dim=-1).values / scale

        quantized_weight = torch.clamp(
            torch.round(weight_f32 / scale.unsqueeze(1) + zero_point.unsqueeze(1)),
            self.weight_qmin,
            self.weight_qmax,
        ).to(torch.int8)

        assert quantized_weight.shape == weight.shape

        self.weight = quantized_weight
        self.scale = scale
        self.zero_point = zero_point
        torch.cuda.empty_cache()

    def forward(self, x):
        if self.activation_scale is not None:
            x_int = (
                torch.round(torch.div(x, self.activation_scale))
                .clamp(
                    self.activation_qmin,
                    self.activation_qmax,
                )
                .to
            )
            assert x.shape == x_int.shape
            output_int = F.linear(x_int, self.weight.to(x.dtype))
            output = output_int * (self.activation_scale * self.scale)

        else:
            adjusted_weight = torch.sub(
                self.weight.to(x.dtype), self.zero_point.unsqueeze(1)
            )
            output = F.linear(x, adjusted_weight) * self.scale
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
    base_model,
    quantizer_class,
    weight_bits=8,
    activation_bits=16,
    exclude_list=[],
    quantized=True,
):
    for name, child in base_model.named_children():
        if isinstance(child, nn.Linear):
            assert hasattr(child, "custom_name")
            if child.custom_name in exclude_list:
                continue
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
                    max_abs = layer_activations.max().item()
                    bits = activation_bits
                    Qmax = 2 ** (bits - 1) - 1
                    layer_scale = max_abs / Qmax

            quantizer_layer: QuantizedLinearLayerWithActivation = quantizer_class(
                in_features,
                out_features,
                activation_scale=layer_scale,
                bias=old_bias is not None,
                dtype=old_weight.dtype,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
            )

            setattr(base_model, name, quantizer_layer)
            if quantized:
                getattr(base_model, name).quantize(old_weight)

            if old_bias is not None:
                getattr(base_model, name).bias = old_bias

        else:
            replace_linear_layer_with_activation(
                base_model=child,
                quantizer_class=quantizer_class,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                exclude_list=exclude_list,
                quantized=quantized,
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


def add_custom_name_to_linear_layers(model) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(
            module, QuantizedLinearLayerWithActivation
        ):
            setattr(module, "custom_name", name)
            names.append(name)


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


def get_weights(model):
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(
            module, QuantizedLinearLayerWithActivation
        ):
            try:
                layer_name = module.custom_name
            except AttributeError:
                add_custom_name_to_linear_layers(model)
                layer_name = module.custom_name
            if layer_name not in weights:
                weights[layer_name] = module.weight.data
    return weights


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


class Quantizer:
    def __init__(self, evaluation_fn) -> None:
        self.evaluation_fn = evaluation_fn
        self.old_gc_objects = None

    def quantize_tensor(self, tensor: torch.Tensor, bit_width):
        qmin = -(2 ** (bit_width - 1))
        qmax = 2 ** (bit_width - 1) - 1

        scale = tensor.abs().max() / qmax
        tensor_q = (tensor / scale).round().clamp(qmin, qmax)
        tensor_q = tensor_q * scale
        return tensor_q

    def quantize_linear_layer(self, layer: nn.Module, bit_width):
        # quantized_layer = copy.deepcopy(layer)

        quantized_layer = nn.Linear(
            layer.in_features,
            layer.out_features,
            bias=(layer.bias is not None),
            device=layer.weight.device,
        )
        quantized_weight = self.quantize_tensor(layer.weight.data, bit_width)
        quantized_layer.weight.data = quantized_weight

        if layer.bias is not None:
            quantized_bias = self.quantize_tensor(layer.bias.data, bit_width)
            quantized_layer.bias.data = quantized_bias

        return quantized_layer

    def compute_layer_error(
        self, original_output: torch.Tensor, quantized_output: torch.Tensor
    ):
        # error = nn.functional.mse_loss(
        #     original_output, quantized_output, reduction="mean"
        # )

        # error function with sqnr
        sqnr: torch.Tensor = torch.mean(original_output**2) / torch.mean(
            (original_output.detach().cpu() - quantized_output.detach().cpu()) ** 2
        )
        sqnr_db: torch.Tensor = 10 * torch.log10(sqnr)
        return sqnr_db

    def print_gc_objects(self):
        counter = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    counter += 1
            except:
                pass
        print(counter)

    def replace_layer_in_model(
        self, model: nn.Module, layer_name: nn.Module, new_layer: nn.Module
    ):
        modules = layer_name.split(".")
        parent_module = model
        for name in modules[:-1]:
            parent_module = getattr(parent_module, name)
        getattr(parent_module, modules[-1]).to("cpu")
        setattr(parent_module, modules[-1], new_layer)
        getattr(parent_module, modules[-1]).to(new_layer.weight.device)
        gc.collect()
        torch.cuda.empty_cache()

    def quantize_layer_independently(
        self, model: nn.Module, error_threshold, quantization_levels
    ):
        with torch.no_grad():
            quantized_model = model
            layer_quantization_info = {}

            layer_inputs_outputs = {}

            hooks = []

            def register_hooks():
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):

                        def create_hook(name):
                            def hook_fn(module, input, output):
                                layer_inputs_outputs[name] = (
                                    input[0].detach().cpu(),
                                    output.detach().cpu(),
                                )

                            return hook_fn

                        hook = module.register_forward_hook(create_hook(name))
                        hooks.append(hook)

            register_hooks()

            with torch.no_grad():
                self.evaluation_fn(model)

            for hook in hooks:
                hook.remove()

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"Quantizing layer {name}")
                    original_input, original_output = layer_inputs_outputs[name]

                    min_error = float("inf")
                    best_bit_width = None
                    best_quantized_layer = None

                    for bit_width in quantization_levels:
                        memory_used = torch.cuda.memory_allocated()
                        quantized_layer = self.quantize_linear_layer(module, bit_width)
                        print(
                            "Diff in memory used (GB):",
                            (torch.cuda.memory_allocated() - memory_used) / 1e9,
                        )
                        quantized_output = quantized_layer(
                            original_input.to(quantized_layer.weight.device)
                        )

                        error = self.compute_layer_error(
                            original_output, quantized_output
                        )

                        if error >= error_threshold:
                            min_error = error
                            best_bit_width = bit_width
                            best_quantized_layer = quantized_layer
                        torch.cuda.empty_cache()

                    if best_quantized_layer is not None:
                        self.replace_layer_in_model(
                            quantized_model, name, best_quantized_layer
                        )

                        layer_quantization_info[name] = (best_bit_width, min_error)
                        # print(
                        #     "Max memory allocation (GB):",
                        #     torch.cuda.max_memory_allocated() / 1e9,
                        # )
                        # print(
                        #     "Memory used (GB):",
                        #     torch.cuda.memory_allocated() / 1e9,
                        # )
                        # print(
                        #     "Memory reserved (GB):",
                        #     torch.cuda.memory_reserved() / 1e9,
                        # )

                    else:
                        print(f"Could not quantize layer {name}")

            return layer_quantization_info
