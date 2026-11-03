
import torch
import torch.nn as nn
import tinycudann as tcnn


class NeuralSDF_4D(nn.Module):
    def __init__(self, log2_hashmap_size: int = 24):
        super().__init__()
        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 10,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        }
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=1,
            encoding_config=config["encoding"],
            network_config=config["network"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class DeformerMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)