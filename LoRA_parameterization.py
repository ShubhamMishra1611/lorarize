import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parameterize


class LORAparameterization(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=1, device="cpu"):
        super().__init__()
        self.lora_A = nn.Parameter(
            torch.zeros((rank, out_features)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((in_features, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)

        self.scale = alpha / rank

        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            return (
                original_weights
                + torch.matmul(self.lora_B,
                               self.lora_A).view(original_weights.shape)
                * self.scale
            )
        else:
            return original_weights


def linear_layer_parametrize(layer, device, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape

    return LORAparameterization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )
