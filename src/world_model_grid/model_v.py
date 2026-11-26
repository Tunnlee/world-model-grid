"""State reconstruction model and training utilities for Model V."""

import sys
import numpy as np

if __package__:
    from . import config
else:
    import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import wandb

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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
        return (
            self.inputs[idx],  # (B, 192)
            self.targets[idx],
        )  # (B, 8, 8)


class ModelV(nn.Module):
    """MLP autoencoder that reconstructs flattened grid states."""

    def __init__(
        self,
        flatten_dim=config.FLATTEN_DIM,
        hidden_dim=config.V_HIDDEN_DIM,
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


def get_data_loader(mode: str, destination: str = "states") -> DataLoader:
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
        file_path = config.TRAIN_DATASET_PATH
        allow_shuffle = True
    elif mode == "val":
        file_path = config.VAL_DATASET_PATH
        allow_shuffle = False
    else:
        raise ValueError("mode must be either 'train' or 'val'")

    with np.load(file_path, allow_pickle=True) as dataset:
        states = torch.from_numpy(
            np.concatenate(dataset[destination], axis=0)
        )  # (Steps,3,8,8)
    return DataLoader(
        GridDataset(states),
        batch_size=config.BATCH_SIZE,
        shuffle=allow_shuffle,
    )


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


def count_metrics(
    preds: torch.Tensor, targs: torch.Tensor
) -> tuple[int, int, int, int, int]:
    agent_accuracy = ((preds == 2) == (targs == 2)).all(dim=(1, 2)).sum().item()
    exact_map = (preds == targs).all(dim=(1, 2)).sum().item()

    pred_obs = (preds == 1).sum().item()
    true_obs = (targs == 1).sum().item()

    correct_pred = ((preds == 1) & (targs == 1)).sum().item()

    return agent_accuracy, exact_map, pred_obs, true_obs, correct_pred


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

        logits = vision_model(inputs).view(-1, num_channels, grid_size, grid_size)

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
) -> tuple[float, dict[str, float]]:
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

    total_agent_accuracy = 0
    total_exact_map = 0
    total_pred_obs = 0
    total_true_obs = 0
    total_correct_pred = 0
    total_map = 0

    with torch.no_grad():
        total_loss = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_distribution(targets)).to(
                device
            )

            logits = vision_model(inputs).view(-1, num_channels, grid_size, grid_size)

            loss = criterion(logits, targets)
            total_loss += loss.item()

            agent_accuracy, exact_map, pred_obs, true_obs, correct_pred = count_metrics(
                logits.argmax(dim=1), targets
            )

            total_agent_accuracy += agent_accuracy
            total_exact_map += exact_map
            total_pred_obs += pred_obs
            total_true_obs += true_obs
            total_correct_pred += correct_pred
            total_map += inputs.shape[0]

        agent_accuracy_rate = total_agent_accuracy / total_map
        exact_map_rate = total_exact_map / total_map
        obstacle_precision = total_correct_pred / (total_pred_obs + 1e-8)
        obstacle_recall = total_correct_pred / (total_true_obs + 1e-8)
        obstacles_f1 = (
            2
            * (obstacle_precision * obstacle_recall)
            / (obstacle_precision + obstacle_recall + 1e-8)
        )

        metrics = {
            "agent_accuracy": agent_accuracy_rate,
            "exact_map": exact_map_rate,
            "obstacle_precision": obstacle_precision,
            "obstacle_recall": obstacle_recall,
            "obstacle_f1": obstacles_f1,
        }

        return total_loss / len(val_loader), metrics


def main() -> None:
    """Train Model V, log losses, and save the best checkpoint."""
    wandb.init(
        project="World-model-grid",
        name="run-03",
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "total_epochs": config.V_TOTAL_EPOCHS,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = ModelV().to(device)
    optimizer = torch.optim.Adam(vision_model.parameters(), lr=config.LEARNING_RATE)

    train_loader = get_data_loader("train")
    val_loader = get_data_loader("val")

    best_val_loss = float("inf")
    for epoch in range(config.V_TOTAL_EPOCHS):
        avg_train_loss = train(vision_model, train_loader, optimizer, device)
        avg_val_loss, val_metrics = val(vision_model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{config.V_TOTAL_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
            f"Agent Acc: {val_metrics['agent_accuracy']:.4f} | Exact Map: {val_metrics['exact_map']:.4f} | Obstacle precision: {val_metrics['obstacle_precision']} | Obstacle recall: {val_metrics['obstacle_recall']:.4f} | Obstacle F1: {val_metrics['obstacle_f1']}"
        )

        wandb.log(
            {
                "Train_loss": avg_train_loss,
                "Val_loss": avg_val_loss,
                **val_metrics,
                "Epoch": epoch + 1,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vision_model.state_dict(), config.MODEL_V_PARAMETERS_PATH)
    wandb.save(config.MODEL_V_PARAMETERS_PATH)
    wandb.finish()


def eval_checkpoint() -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = ModelV().to(device)
    vision_model.load_state_dict(
        torch.load(
            config.MODEL_V_PARAMETERS_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    val_loader = get_data_loader("val")
    _, metrics = val(vision_model, val_loader, device)
    return metrics


def save_recon_samples(num_samples: int = 6) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model = ModelV().to(device)
    vision_model.load_state_dict(
        torch.load(
            config.MODEL_V_PARAMETERS_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    vision_model.eval()

    val_loader = get_data_loader("val")
    inputs, targets = next(iter(val_loader))
    inputs, targets = inputs[:num_samples].to(device), targets[:num_samples]

    with torch.no_grad():
        preds = (
            vision_model(inputs)
            .view(-1, config.NUM_CHANNELS, config.GRID_SIZE, config.GRID_SIZE)
            .argmax(dim=1)
            .cpu()
        )

    cmap = ListedColormap(["#F1F5F9", "#1E293B", "#F97316"])

    fig, axes = plt.subplots(2, num_samples, figsize=(2.4 * num_samples, 5.2))
    for i in range(num_samples):
        for row, (grid, title) in enumerate(
            [
                (targets[i], f"Ground Truth #{i + 1}"),
                (preds[i], f"Reconstructed #{i + 1}"),
            ]
        ):
            ax = axes[row, i]
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=2)
            ax.set_xticks([x - 0.5 for x in range(1, config.GRID_SIZE)], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, config.GRID_SIZE)], minor=True)
            ax.grid(which="minor", color="#CBD5E1", linestyle="-", linewidth=0.8)
            ax.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_title(title, fontsize=10)

    legend_items = [
        Patch(facecolor="#F1F5F9", edgecolor="#CBD5E1", label="Empty (0)"),
        Patch(facecolor="#1E293B", label="Obstacle (1)"),
        Patch(facecolor="#F97316", label="Agent (2)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    plt.savefig(config.MODEL_V_RECON_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved reconstruction comparison to: {config.MODEL_V_RECON_PATH}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        main()
    elif sys.argv[1] == "metrics":
        metrics = eval_checkpoint()
        for metric in metrics.items():
            print(f"{metric[0]}, {metric[1]}")
    elif sys.argv[1] == "recon":
        save_recon_samples()
