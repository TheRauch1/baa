import copy
import gc
from collections import OrderedDict
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
    def __init__(self, evaluation_fn, min_quantile, max_quantile) -> None:
        self.evaluation_fn = evaluation_fn
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

    def quantize_tensor(self, tensor: torch.Tensor, bit_width):
        qmin = -(2 ** (bit_width - 1))
        qmax = 2 ** (bit_width - 1) - 1

        # tensor_q = tensor.clone()
        # scale = tensor.abs().max() / qmax
        # tensor_q = (tensor_q / scale).round().clamp(qmin, qmax)
        # tensor_q = tensor_q * scale

        # use min and max of tensor to calculate scale
        # scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        # zero_point = qmin - tensor.min() / scale
        # tensor_q = (
        #     torch.clamp(torch.round(tensor / scale - zero_point), qmin, qmax) * scale
        #     + zero_point
        # )

        try:
            scale_min = tensor.quantile(self.min_quantile, dim=0)
            scale_max = tensor.quantile(self.max_quantile, dim=0)
        except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
            gc.collect()
            torch.cuda.empty_cache()
            t = tensor.cpu()
            scale_min = t.quantile(self.min_quantile, dim=0).to(tensor.device)
            scale_max = t.quantile(self.max_quantile, dim=0).to(tensor.device)
            del t
            gc.collect()
            torch.cuda.empty_cache()

        # scale_min = tensor.quantile(0.05, dim=0)
        # scale_min = torch.sort(tensor, dim=0).values[int(tensor.shape[0] * 0.05)]
        # scale_max = tensor.quantile(0.95, dim=0)
        # scale_max = torch.sort(tensor, dim=0).values[int(tensor.shape[0] * 0.95)]

        try:
            tensor_q = tensor.clone()
            scale = (qmax - qmin) / (scale_max - scale_min)
        except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
            old_device = tensor.device
            tensor_q = tensor.cpu().clone()
            gc.collect()
            torch.cuda.empty_cache()
            scale = (qmax - qmin) / (scale_max - scale_min)
            zero_point = (-scale * scale_min).round() - scale_max
            tensor_q.multiply_(scale.cpu()).add_(zero_point.cpu()).round_()
            tensor_q.sub_(zero_point.cpu()).div_(scale.cpu())
            return tensor_q

        zero_point = (-scale * scale_min).round() - scale_max
        # quantize
        tensor_q.multiply_(scale).add_(zero_point).round_()
        # dequantize
        tensor_q.sub_(zero_point).div_(scale)
        return tensor_q

    def quantize_linear_layer(self, layer: nn.Module, bit_width):
        quantized_layer = nn.Linear(
            layer.in_features,
            layer.out_features,
            bias=(layer.bias is not None),
            # device=layer.weight.device,
            dtype=layer.weight.dtype,
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
            # (original_output.detach().cpu() - quantized_output.detach().cpu()) ** 2
            (original_output.to(quantized_output.device) - quantized_output) ** 2
        )
        sqnr_db: torch.Tensor = 10 * torch.log10(sqnr)
        return sqnr_db

    def replace_layer_in_model(
        self, model: nn.Module, layer_name: nn.Module, new_layer: nn.Module
    ):
        modules = layer_name.split(".")
        parent_module = model
        for name in modules[:-1]:
            parent_module = getattr(parent_module, name)
        device = getattr(parent_module, modules[-1]).weight.device
        old_layer = getattr(parent_module, modules[-1]).to("cpu")
        setattr(parent_module, modules[-1], new_layer.to(device))

        # Clear parameters and buffers
        old_layer._parameters = {k: None for k in old_layer._parameters}
        old_layer._buffers = {k: None for k in old_layer._buffers}

        # Clear hooks
        if hasattr(old_layer, "_backward_hooks"):
            old_layer._backward_hooks.clear()
        if hasattr(old_layer, "_forward_hooks"):
            old_layer._forward_hooks.clear()
        if hasattr(old_layer, "_forward_pre_hooks"):
            old_layer._forward_pre_hooks.clear()
        if hasattr(old_layer, "_state_dict_hooks"):
            old_layer._state_dict_hooks.clear()
        if hasattr(old_layer, "_load_state_dict_pre_hooks"):
            old_layer._load_state_dict_pre_hooks.clear()

        del old_layer
        # getattr(parent_module, modules[-1]).to(new_layer.weight.device)
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
                original_model_accuracy = self.evaluation_fn(model)

            for hook in hooks:
                hook.remove()

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"Quantizing layer {name}")
                    original_input, original_output = layer_inputs_outputs[name]

                    min_error = float("inf")
                    best_bit_width = None
                    best_quantized_layer = None

                    for bit_width in tqdm(quantization_levels):
                        quantized_layer = self.quantize_linear_layer(module, bit_width)
                        quantized_output = quantized_layer(
                            original_input.to(quantized_layer.weight.device)
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        # print(torch.cuda.memory_stats())

                        error = self.compute_layer_error(
                            original_output, quantized_output
                        )

                        if error >= error_threshold:
                            min_error = error
                            best_bit_width = bit_width
                            if best_quantized_layer is not None:
                                best_quantized_layer.to("cpu")
                                del best_quantized_layer
                            best_quantized_layer = quantized_layer
                        gc.collect()
                        torch.cuda.empty_cache()

                    if best_quantized_layer is not None:
                        self.replace_layer_in_model(
                            quantized_model, name, best_quantized_layer
                        )

                        layer_quantization_info[name] = (
                            best_bit_width,
                            min_error.item(),
                        )
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

            return (
                layer_quantization_info,
                original_model_accuracy,
            )
