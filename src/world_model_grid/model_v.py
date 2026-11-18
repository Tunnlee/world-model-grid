"""State reconstruction model and training utilities for Model V."""

import numpy as np
from . import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import wandb


def quick_test() -> None:
    """Print the shapes and dtypes from one training batch.

    Notes
    -----
    This helper loads the configured training dataset through
    :func:`get_data_loader` and is intended for a quick input-pipeline check.
    """
    train_loader = get_data_loader("train")
    batch = next(iter(train_loader))
    print(batch[0].shape, batch[0].dtype)
    print(batch[1].shape)


class GridDataset(Dataset):
    """Flatten grid observations and derive per-cell class targets."""

    def __init__(
        self,
        states: torch.Tensor,
        flatten_dim: int = config.FLATTEN_DIM,
    ):
        """Create input vectors and class-index targets from grid states.

        Parameters
        ----------
        states : torch.Tensor
            Observations with shape ``(n_states, channels, height, width)``.
            The channels are expected to encode the grid cell classes.
        flatten_dim : int, optional
            Number of values in a flattened observation. Defaults to the
            configured grid size times the number of channels.

        Raises
        ------
        ValueError
            If ``states`` is not a four-dimensional tensor.
        """
        super().__init__()

        if states.ndim != 4:
            raise ValueError("States must be a 4D array")

        self.inputs = states.reshape(-1, flatten_dim)  # Shape (N, 192)
        self.targets = states.argmax(axis=1)  # Shape (N, 8, 8)

    def __len__(self) -> int:
        """Return the number of grid states in the dataset.

        Returns
        -------
        int
            Number of observations.
        """
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        """Return one flattened observation and its per-cell labels.

        Parameters
        ----------
        idx : int
            Index of the observation to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            The flattened input vector and a ``(height, width)`` tensor of
            integer class labels.
        """
        return self.inputs[idx], self.targets[idx]


class ModelV(nn.Module):
    """Encode a grid observation into a latent vector and reconstruct it.

    The decoder returns unnormalized per-cell class logits. A spatial view of
    those logits is used with cross-entropy loss during training.
    """

    def __init__(
        self,
        flatten_dim=config.FLATTEN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
    ):
        """Initialize the encoder and decoder multilayer perceptrons.

        Parameters
        ----------
        flatten_dim : int, optional
            Number of values in a flattened grid observation.
        hidden_dim : int, optional
            Width of the hidden layer in both encoder and decoder.
        latent_dim : int, optional
            Width of the compressed latent representation.
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
        """Encode and reconstruct a batch of flattened grid observations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, flatten_dim)``.

        Returns
        -------
        torch.Tensor
            Unnormalized reconstruction logits with shape
            ``(batch_size, flatten_dim)``.
        """
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon


def get_data_loader(mode: str) -> DataLoader:
    """Load a configured dataset split and wrap it in a DataLoader.

    Parameters
    ----------
    mode : {"train", "val"}
        Selects the training or validation NPZ file.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of flattened state inputs and per-cell class targets. Training
        batches are shuffled; validation batches preserve dataset order.

    Raises
    ------
    ValueError
        If ``mode`` is neither ``"train"`` nor ``"val"``.

    Notes
    -----
    The NPZ files contain variable-length episodes stored as object arrays,
    so loading enables pickle and concatenates their state sequences.
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
    """Compute inverse-frequency weights for the grid cell classes.

    Parameters
    ----------
    targets : torch.Tensor
        Integer class labels, typically shaped ``(batch, height, width)``.

    Returns
    -------
    torch.Tensor
        One weight per configured class. Classes absent from the batch receive
        a clamped count to avoid division by zero.
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
    """Run one training epoch over the training data.

    Parameters
    ----------
    vision_model : ModelV
        Model whose parameters are updated.
    train_loader : torch.utils.data.DataLoader
        Batches of flattened inputs and cell-class targets.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device to use for model inputs, targets, and loss computation.

    Returns
    -------
    float
        Arithmetic mean of the batch cross-entropy losses for the epoch.
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
    """Measure reconstruction loss without updating model parameters.

    Parameters
    ----------
    vision_model : ModelV
        Model to evaluate.
    val_loader : torch.utils.data.DataLoader
        Batches of flattened inputs and cell-class targets.
    device : torch.device
        Device to use for model inputs, targets, and loss computation.

    Returns
    -------
    float
        Arithmetic mean of the batch cross-entropy losses for validation.
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
    """Train Model V, log losses, and save the best validation checkpoint.

    The function chooses CUDA when available and CPU otherwise, creates the
    train and validation loaders, and logs each epoch to Weights & Biases.
    """
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
