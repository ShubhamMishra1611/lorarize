import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parameterize
import yaml

from LoRA_parameterization import linear_layer_parametrize, LORAparameterization
from original_weight_save import save_weights


with open("config.yaml", "r") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


def get_lora_stats(model_layers: list) -> None:
    total_parameters_original = 0
    for index, layer in enumerate(model_layers):
        total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
        print(
            f"Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}")
    print(f"Total number of parameters: {total_parameters_original:,}")
    total_parameters_lora = 0
    total_parameters_non_lora = 0
    for index, layer in enumerate(model_layers):
        total_parameters_lora += (
            layer.parametrizations["weight"][0].lora_A.nelement()
            + layer.parametrizations["weight"][0].lora_B.nelement()
        )
        total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
        print(
            f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
        )
    # The non-LoRA parameters count must match the original network
    assert total_parameters_non_lora == total_parameters_original
    print(
        f"Total number of parameters (original): {total_parameters_non_lora:,}")
    print(
        f"Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}"
    )
    print(f"Parameters introduced by LoRA: {total_parameters_lora:,}")
    parameters_incremment = (total_parameters_lora /
                             total_parameters_non_lora) * 100
    print(f"Parameters incremment: {parameters_incremment:.3f}%")


def enable_disable_lora(model:nn.Module|nn.ModuleDict|nn.ModuleList, model_layers: list| None = None, enabled: bool = True) -> None:
    if model_layers is None:
        model_layers = [
            layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    for layer in model_layers:
        layer.parametrizations["weight"][0].enabled = enabled


def main(model: nn.Module, model_layer: list | None = None):
    if model_layer is None:
        model_layer = [
            layer for layer in model if isinstance(layer, nn.Linear)]
    # Save original weights
    save_weights(model, config.original_weight_file)
    # Parametrize layers
    for layer in model_layer:
        parameterize.register_parametrization(
            layer,
            "weight",
            linear_layer_parametrize(
                layer, device=config.device, rank=config.rank, lora_alpha=config.alpha
            ),
        )
