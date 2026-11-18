import sys

import numpy as np
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import wandb


def quick_test():
    train_states, _ = load_dataset()
    train_dataset = GridDataset(train_states)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    batch = next(iter(train_loader))
    print(batch[0].shape, batch[0].dtype)
    print(batch[1].shape)


class GridDataset(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        flatten_dim: int = config.FLATTEN_DIM,
    ):
        super().__init__()

        if states.ndim != 4:
            raise ValueError("States must be a 4D array")

        self.inputs = states.reshape(-1, flatten_dim)  # Shape (N, 192)
        self.targets = states.argmax(axis=1)  # Shape (N, 8, 8)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


class ModelV(nn.Module):
    def __init__(
        self,
        flatten_dim=config.FLATTEN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flatten_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


def get_data_loader(mode):
    if mode == "train":
        with np.load(config.TRAIN_DATASET_PATH, allow_pickle=True) as train_dataset:
            train_states = torch.from_numpy(
                np.concatenate(train_dataset["states"], axis=0)
            )
        return DataLoader(
            GridDataset(train_states),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )

    elif mode == "val":
        with np.load(config.VAL_DATASET_PATH, allow_pickle=True) as val_dataset:
            val_states = torch.from_numpy(np.concatenate(val_dataset["states"], axis=0))
        return DataLoader(
            GridDataset(val_states),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
        )


def weight_distribution(targets):
    counts = torch.bincount(targets.flatten(), minlength=config.NUM_CHANNELS).clamp(
        min=1
    )
    class_weights = (counts.sum() / (config.NUM_CHANNELS * counts)).float()
    return class_weights


def train(vision_model, train_loader, optimizer, device):
    vision_model.train()
    best_loss = float("inf")

    total_loss = 0

    for inputs, targets in train_loader:
        criterion = nn.CrossEntropyLoss(weight=weight_distribution(targets)).to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        logits = vision_model(inputs).view(
            -1, config.NUM_CHANNELS, config.GRID_SIZE, config.GRID_SIZE
        )

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def val(vision_model, val_loader, device):
    vision_model.eval()

    with torch.no_grad():
        total_loss = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_distribution(targets)).to(
                device
            )

            logits = vision_model(inputs).view(
                -1, config.NUM_CHANNELS, config.GRID_SIZE, config.GRID_SIZE
            )

            loss = criterion(logits, targets)
            total_loss += loss.item()

        return total_loss / len(val_loader)


def main():
    wandb.init(
        project="World-model-grid",
        name="run-02",
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "total_epochs": config.TOTAL_EPOCHS,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = ModelV().to(device)
    optimizer = torch.optim.Adam(vision_model.parameters(), lr=config.LEARNING_RATE)


    train_loader = get_data_loader("train")
    val_loader = get_data_loader("val")

    best_val_loss = float("inf")
    for epoch in range(config.TOTAL_EPOCHS):
        avg_train_loss = train(vision_model, train_loader, optimizer, device)
        avg_val_loss = val(vision_model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{config.TOTAL_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        wandb.log(
            {"Train_loss": avg_train_loss, "Val_loss": avg_val_loss, "Epoch": epoch + 1}
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vision_model.state_dict(), config.MODEL_V_PARAMETERS_PATH)


if __name__ == "__main__":
    main()
