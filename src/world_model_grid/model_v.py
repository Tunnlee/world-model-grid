"""State reconstruction model and training utilities for Model V."""

import numpy as np
from . import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import wandb


def quick_test() -> None:
    """Print one training batch's input and target shapes."""
    train_loader = get_data_loader("train")
    batch = next(iter(train_loader))
    print(batch[0].shape, batch[0].dtype)
    print(batch[1].shape)


class GridDataset(Dataset):
    """Flatten observations and create per-cell class labels."""

    def __init__(
        self,
        states: torch.Tensor,
        flatten_dim: int = config.FLATTEN_DIM,
    ):
        """Create flattened inputs and class-index targets.

        Parameters
        ----------
        states : torch.Tensor
            State tensor shaped (n_states, channels, height, width).
        flatten_dim : int, optional
            Flattened input size. Defaults to the configured grid size.

        Raises
        ------
        ValueError
            If states is not four-dimensional.
        """
        super().__init__()

        if states.ndim != 4:
            raise ValueError("States must be a 4D array")

        self.inputs = states.reshape(-1, flatten_dim)  # Shape (N, 192)
        self.targets = states.argmax(axis=1)  # Shape (N, 8, 8)

    def __len__(self) -> int:
        """Return the number of observations.

        Returns
        -------
        int
            Dataset size.
        """
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        """Return one input vector and its cell labels.

        Parameters
        ----------
        idx : int
            Observation index.

        Returns
        -------
        tuple of torch.Tensor
            Flattened input and (height, width) target.
        """
        return self.inputs[idx], self.targets[idx]


class ModelV(nn.Module):
    """MLP autoencoder that reconstructs flattened grid states."""

    def __init__(
        self,
        flatten_dim=config.FLATTEN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
    ):
        """Initialize the encoder and decoder.

        Parameters
        ----------
        flatten_dim : int
            Flattened state size.
        hidden_dim : int
            Hidden layer width.
        latent_dim : int
            Latent vector size.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct a batch of states.

        Parameters
        ----------
        x : torch.Tensor
            Input shaped (batch_size, flatten_dim).

        Returns
        -------
        torch.Tensor
            Reconstruction logits with the input shape.
        """
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


def get_data_loader(mode: str) -> DataLoader:
    """Load a train or validation split.

    Parameters
    ----------
    mode : {"train", "val"}
        Dataset split to load.

    Returns
    -------
    DataLoader
        Batches of flattened states and cell labels.

    Raises
    ------
    ValueError
        If mode is unsupported.
    """
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

    if mode == "val":
        with np.load(config.VAL_DATASET_PATH, allow_pickle=True) as val_dataset:
            val_states = torch.from_numpy(np.concatenate(val_dataset["states"], axis=0))
        return DataLoader(
            GridDataset(val_states),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
        )

    raise ValueError("mode must be either 'train' or 'val'")


def weight_distribution(targets: torch.Tensor) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Parameters
    ----------
    targets : torch.Tensor
        Integer class labels.

    Returns
    -------
    torch.Tensor
        One weight per configured class.
    """
    counts = torch.bincount(targets.flatten(), minlength=config.NUM_CHANNELS).clamp(
        min=1
    )
    class_weights = (counts.sum() / (config.NUM_CHANNELS * counts)).float()
    return class_weights


def train(
    vision_model: ModelV,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Parameters
    ----------
    vision_model : ModelV
        Model to update.
    train_loader : DataLoader
        Training batches.
    optimizer : torch.optim.Optimizer
        Parameter optimizer.
    device : torch.device
        Device for computation.

    Returns
    -------
    float
        Mean batch loss.
    """
    vision_model.train()
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


def val(
    vision_model: ModelV,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model without updating weights.

    Parameters
    ----------
    vision_model : ModelV
        Model to evaluate.
    val_loader : DataLoader
        Validation batches.
    device : torch.device
        Device for computation.

    Returns
    -------
    float
        Mean validation batch loss.
    """
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


def main() -> None:
    """Train Model V, log losses, and save the best checkpoint."""
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
