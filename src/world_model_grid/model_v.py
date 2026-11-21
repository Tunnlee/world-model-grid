"""State reconstruction model and training utilities for Model V."""

import sys
import numpy as np
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import wandb


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
            raise ValueError("States must be a 4D Tensor")

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    """Load a train or validation Dataloader split.

    Parameters
    ----------
    mode : {"train", "val"}
        Dataset split to load.

    Returns
    -------
    DataLoader
        Callable data object for the specified datase.

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
    num_channels: int = config.NUM_CHANNELS,
    grid_size: int = config.GRID_SIZE,
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
            -1, num_channels, grid_size, grid_size
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
    num_channels: int = config.NUM_CHANNELS,
    grid_size: int = config.GRID_SIZE,
    ) -> tuple[float, dict]:
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
                -1, num_channels, grid_size, grid_size
            )

            loss = criterion(logits, targets)
            total_loss += loss.item()

            metrics = get_metrics("val", vision_model, logits, targets)

        return total_loss / len(val_loader), metrics


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


def get_metrics(mode, vision_model: ModelV = None, predictions: torch.Tensor = None, targets: torch.Tensor = None):
    if mode == 'val':
        agent_accuracy, exact_map, pred_obs, true_obs, correct_pred = count_metrics(
            predictions.argmax(dim=1), targets
        )

        total_map = inputs.shape[0] 

        agent_accuracy_rate = agent_accuracy / total_map
        exact_map_rate = exact_map / total_map
        obstacle_precision = correct_pred / pred_obs
        obstacle_recall = correct_pred / true_obs
        obstacles_f1 = 2 * (obstacle_precision * obstacle_recall) / (obstacle_precision + obstacle_recall) 

        metrics = {
                "agent_accuracy": agent_accuracy_rate,
                "exact_map": exact_map_rate,
                "obstacle_precision": obstacle_precision,
                "obstacle_recall": obstacle_recall,
                "obstacle_f1": obstacles_f1,
            }
        return metrics
    
    elif mode == 'final':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_model = ModelV().to(device)
        vision_model.load_state_dict(torch.load(config.MODEL_V_PARAMETERS_PATH))
        vision_model.eval()
        val_loader = get_data_loader("val")

        with torch.no_grad():
            total_agent_accuracy = 0
            total_exact_map = 0
            total_pred_obs = 0
            total_true_obs = 0
            total_correct_pred = 0
            total_map = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = vision_model(inputs).view(-1, 3, 8, 8).argmax(dim=1)
            agent_accuracy, exact_map, pred_obs, true_obs, correct_pred = count_metrics(
                predictions, targets
            )
            total_agent_accuracy += agent_accuracy
            total_exact_map += exact_map
            total_pred_obs += pred_obs
            total_true_obs += true_obs
            total_correct_pred += correct_pred
            total_map += inputs.shape[0] 

        agent_accuracy_rate = total_agent_accuracy / total_map
        exact_map_rate = total_exact_map / total_map
        obstacle_precision = total_correct_pred / total_pred_obs
        obstacle_recall = total_correct_pred / total_true_obs
        obstacles_f1 = 2 * (obstacle_precision * obstacle_recall) / (obstacle_precision + obstacle_recall) 

        metrics = {
                "agent_accuracy": agent_accuracy_rate,
                "exact_map": exact_map_rate,
                "obstacle_precision": obstacle_precision,
                "obstacle_recall": obstacle_recall,
                "obstacle_f1": obstacles_f1,
            }
        return metrics


def count_metrics(preds: torch.Tensor, targs: torch.Tensor) -> tuple[int, int, int, int, int]:
    agent_accuracy = ((preds == 2) == (targs == 2)).all(dim=(1, 2)).sum().item()
    exact_map = (preds == targs).all(dim=(1, 2)).sum().item()

    pred_obs = (preds == 1).sum().item()
    true_obs = (targs == 1).sum().item()

    correct_pred = ((preds == 1) & (targs == 1)).sum().item()

    return agent_accuracy, exact_map, pred_obs, true_obs, correct_pred


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        main()  
    elif sys.argv[1] == 'metrics':
        metrics = get_metrics("final")
        for metric in metrics.items():
            print(f'{metric[0]}, {metric[1]}')

