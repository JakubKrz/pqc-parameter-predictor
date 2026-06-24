import math
import torch
import torch.nn as nn


class HybridAutoencoder(nn.Module):
    def __init__(self, classical_encoder, quantum_layer, num_qubits=6):
        super().__init__()
        self.classical_encoder = classical_encoder
        self.quantum_layer = quantum_layer
        self.num_qubits = num_qubits
        self.num_pixels = 2 ** num_qubits

    def forward(self, x):
        params = self.classical_encoder(x)
        quantum_probs = self.quantum_layer(params)
        
        num_pixels = 2 ** self.num_qubits
        output_pixels = quantum_probs[:, num_pixels:]
        output_image_scaled = output_pixels * num_pixels
        size = int(num_pixels ** 0.5)
        return output_image_scaled.view(-1, size, size)