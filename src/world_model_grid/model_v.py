import numpy as np
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, flatten_dim = config.FLATTEN_DIM, hidden_dim = config.HIDDEN_DIM, latent_dim = config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flatten_dim)
        )

    def forward(self,x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

def load_dataset():
    with np.load(config.DATASET_PATH, allow_pickle=True) as data:
        dataset = data["states"]
    train_states = torch.from_numpy(np.concatenate(
        dataset[: int(config.TOTAL_EPISODES * 0.9)], axis=0
    ))  # Concatenate iterates list of pointers
    val_states = torch.from_numpy(np.concatenate(dataset[int(config.TOTAL_EPISODES * 0.9) :], axis=0))
    return train_states, val_states


def weight_distribution(targets):
    counts = torch.bincount(targets.flatten(), minlength=config.NUM_CHANNELS).clamp(min=1)
    class_weights = (counts.sum() / (config.NUM_CHANNELS * counts)).float()
    return class_weights                                        


def train():
    train_states, _ = load_dataset()
    train_dataset = GridDataset(train_states)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = ModelV().to(device)
    vision_model.train()
    
    optimizer = torch.optim.Adam(vision_model.parameters(), lr=0.001)

    for epoch in range(config.TOTAL_EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            criterion = nn.CrossEntropyLoss(weight = weight_distribution(targets)).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = vision_model(inputs)
            logits = outputs.view(-1, config.NUM_CHANNELS, config.GRID_SIZE, config.GRID_SIZE)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{config.TOTAL_EPOCHS}, Avg Loss: {avg_loss:.9f}')

if __name__ == "__main__":
    train()





















