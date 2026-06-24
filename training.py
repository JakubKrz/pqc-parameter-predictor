import sys
import os
import time
import json
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import ProfilerActivity, profile, record_function
import pennylane as qml
from torch.utils.data import DataLoader, Subset

from circuit import create_qnode
from autoencoder import Autoencoder
from hybrid_autoencoder import HybridAutoencoder
from data import get_mnist_dataloaders

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.set_default_dtype(torch.float64)


class PennyLaneQuantumLayer(nn.Module):
    def __init__(self, qnode):
        super().__init__()
        self.qnode = qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.qnode(x)
        return torch.stack([self.qnode(xi) for xi in x])

def main(num_layers_override=None):
    if num_layers_override is not None:
        NUM_LAYERS = num_layers_override
    else:
        NUM_LAYERS = 5
    
    NUM_QUBITS   = 2
    NUM_PARAMETERS = (NUM_QUBITS * 4) * NUM_LAYERS

    IMAGE_SIZE   = int(2 ** (NUM_QUBITS / 2))
    BATCH_SIZE   = 2
    EPOCHS       = 100
    LEARNING_RATE = 0.001
    ACTIVATION = "ReLU"
    HIDDEN = 48
    BOTTLENECK = 23
    NOISE_LEVEL = 0.0

    OPTIMIZER_NAME = "Adam"
    LOSS_FUNCTION_NAME = "L1Loss"
    BACKEND_NAME = "default.mixed"
    GRADIENT_NAME = "Backprop_PL"
    SCHEDULER_NAME = "ReduceLROnPlateau"

    TRAIN_FRACTION = 0.05
    VAL_FRACTION   = 0.05

    SCHEDULER_FACTOR   = 0.5 
    SCHEDULER_PATIENCE = 5
    LOG_FREQ      = 1

    device = torch.device("cpu")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = os.path.join("runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_file_path     = os.path.join(run_dir, "training_log.txt")
    metrics_file_path = os.path.join(run_dir, "metrics.json")

    def log_print(message):
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    hyperparameters = {
        "NUM_QUBITS":    NUM_QUBITS,
        "NUM_LAYERS":    NUM_LAYERS,
        "NUM_PARAMETERS": NUM_PARAMETERS,
        "IMAGE_SIZE":    IMAGE_SIZE,
        "BATCH_SIZE":    BATCH_SIZE,
        "EPOCHS":        EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "TRAIN_FRACTION": TRAIN_FRACTION,
        "OPTIMIZER":  OPTIMIZER_NAME,
        "SCHEDULER": {
                    "TYPE":     SCHEDULER_NAME,
                    "FACTOR":   SCHEDULER_FACTOR,
                    "PATIENCE": SCHEDULER_PATIENCE,
        },
        "LOSS_FUNCTION": LOSS_FUNCTION_NAME,
        "BACKEND":       BACKEND_NAME,
        "GRADIENT":      GRADIENT_NAME,
        "HIDDEN" : HIDDEN,
        "BOTTLENECK": BOTTLENECK,
        "ACTIVATION" : ACTIVATION,
        "NOISE_LEVEL" : NOISE_LEVEL,
    }

    with open(os.path.join(run_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=4)

    log_print(f"Rozpoczęto nową sesję. Pliki w: {run_dir}")

    dev = qml.device(BACKEND_NAME, wires=NUM_QUBITS + 1)
    qnode = create_qnode(NUM_QUBITS, NUM_LAYERS, dev, NOISE_LEVEL)
    quantum_layer = PennyLaneQuantumLayer(qnode)

    classical_ae     = Autoencoder(image_size=IMAGE_SIZE, num_params=NUM_PARAMETERS,
                        hidden_dim=HIDDEN, bottleneck_size=BOTTLENECK, activation=ACTIVATION)
    model_hybrydowy  = HybridAutoencoder(classical_ae, quantum_layer,
                                        num_qubits=NUM_QUBITS)
    model_hybrydowy.to(device)

    with open(os.path.join(run_dir, "architecture.txt"), "w", encoding="utf-8") as f:
        f.write(str(model_hybrydowy))

    full_train_loader, full_val_loader = get_mnist_dataloaders(
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

    train_dataset     = full_train_loader.dataset
    num_train_samples = int(len(train_dataset) * TRAIN_FRACTION)
    indices           = torch.randperm(len(train_dataset))[:num_train_samples]
    subset_train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset     = full_val_loader.dataset
    num_val_samples = int(len(val_dataset) * VAL_FRACTION)
    val_indices     = torch.randperm(len(val_dataset))[:num_val_samples]
    subset_val_dataset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(subset_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    log_print(f"Liczba próbek treningowych: {num_train_samples}")
    log_print(f"Liczba próbek walidacyjnych: {num_val_samples}")

    if OPTIMIZER_NAME.lower() == "adam":
        optimizer = optim.Adam(model_hybrydowy.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_NAME.lower() == "sgd":
        optimizer = optim.SGD(model_hybrydowy.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model_hybrydowy.parameters(), lr=LEARNING_RATE)

    # Create scheduler according to SCHEDULER_NAME
    if SCHEDULER_NAME.lower() == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    elif SCHEDULER_NAME.lower() == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=SCHEDULER_PATIENCE, gamma=SCHEDULER_FACTOR)
    elif SCHEDULER_NAME.lower() in ("none", "no", ""):
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    if LOSS_FUNCTION_NAME.lower() in ("l1loss", "l1"):
        criterion = nn.L1Loss()
    elif LOSS_FUNCTION_NAME.lower() in ("mseloss", "mse"):
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    history       = {"train_loss": [], "val_loss": [], "learning_rate": []}
    total_start_time = time.time()

    PROFILING_ENABLED = True
    profiling_batches = 1
    profiler = None

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model_hybrydowy.train()
        train_epoch_loss = 0.0

        if PROFILING_ENABLED and profiler is None:
            profiler = profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            )
            profiler.__enter__()

        for batch_idx, (batch_images, _) in enumerate(train_loader):
            batch_images  = batch_images.to(device)
            target_images = batch_images.squeeze(1)

            if PROFILING_ENABLED and batch_idx < profiling_batches:
                with record_function(f"train_batch_{batch_idx + 1}"):
                    optimizer.zero_grad()
                    reconstructed_batch = model_hybrydowy(batch_images)
                    loss = criterion(reconstructed_batch, target_images)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                reconstructed_batch = model_hybrydowy(batch_images)
                loss = criterion(reconstructed_batch, target_images)
                loss.backward()
                optimizer.step()

            train_epoch_loss += loss.item()

            if PROFILING_ENABLED and batch_idx == profiling_batches - 1 and profiler is not None:
                profiler.__exit__(None, None, None)
                profiler_summary = profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20)
                log_print("Profiler: podsumowanie pierwszych 3 batchy:\n" + profiler_summary)
                with open(os.path.join(run_dir, "profiler_first_3_batches.txt"), "w", encoding="utf-8") as pf:
                    pf.write(profiler_summary)
                profiler = None

            current_batch = batch_idx + 1
            if current_batch % LOG_FREQ == 0 or current_batch == len(train_loader):
                elapsed_time   = time.time() - epoch_start_time
                avg_batch_time = elapsed_time / current_batch
                eta_seconds   = (len(train_loader) - current_batch) * avg_batch_time
                log_print(f"  [Train] Epoka {epoch+1:02d} | Batch {current_batch:04d}/{len(train_loader)} "
                    f"| Loss (bieżący): {loss.item():.4f} "
                    f"| Śr. czas/batch: {avg_batch_time:.3f}s | ETA epoki: {eta_seconds:.0f}s")
                
        avg_train_loss = train_epoch_loss / len(train_loader)
        
        model_hybrydowy.eval()
        val_epoch_loss = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for batch_images, _ in val_loader:
                batch_images  = batch_images.to(device)
                target_images = batch_images.squeeze(1)
                reconstructed_batch = model_hybrydowy(batch_images)
                val_epoch_loss += criterion(reconstructed_batch, target_images).item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_duration = time.time() - val_start_time

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(current_lr)
        with open(metrics_file_path, "w") as f:
            json.dump(history, f, indent=4)

        epoch_duration = time.time() - epoch_start_time
        log_print(f"=== Epoka {epoch+1}/{EPOCHS} ZAKOŃCZONA ===")
        log_print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        log_print(f"    Czas Treningu: {epoch_duration - val_duration:.1f}s | Czas Walidacji: {val_duration:.1f}s\n")

        torch.save({
            'epoch':                epoch + 1,
            'model_state_dict':     model_hybrydowy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss':           avg_train_loss,
            'val_loss':             avg_val_loss,
        }, os.path.join(run_dir, f"model_epoch_{epoch+1}.pth"))

    total_duration = time.time() - total_start_time
    log_print(f"\nTrening zakończony! Całkowity czas: {total_duration/60:.2f} min.")

    torch.save({
        'epoch':                EPOCHS,
        'model_state_dict':     model_hybrydowy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(run_dir, "model_FINAL.pth"))


if __name__ == "__main__":
    main()