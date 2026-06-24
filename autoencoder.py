import torch
import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(
        self,
        image_size=8,
        num_params=48,
        hidden_dim=32,
        bottleneck_size=24,
        activation="ReLU",
    ):
        super().__init__()
        self.input_dim = image_size * image_size
        try:
            self.activation_class = getattr(nn, activation)
        except AttributeError as exc:
            raise ValueError(f"Unknown activation '{activation}'") from exc

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            self.activation_class(),
            nn.Linear(hidden_dim, bottleneck_size),
            self.activation_class(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_dim),
            self.activation_class(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.activation_class(),
            nn.Linear(hidden_dim * 2, num_params),
            nn.Tanh(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded * torch.pi