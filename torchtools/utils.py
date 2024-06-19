import math
import torch.nn as nn


def divide_int(a: int, b: int) -> int:
    if not isinstance(a, int):
        raise TypeError
    if not isinstance(b, int):
        raise TypeError
    if b == 0:
        return math.nan
    else:
        return a // b


def get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Activation {activation} not supported")


class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", blocks=1):
        super(CustomModel, self).__init__()
        hidden_blocks = []
        for _ in range(blocks):
            hidden_blocks.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_blocks.append(get_activation(activation))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            *hidden_blocks,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)
