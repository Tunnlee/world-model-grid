import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

if __package__:
    from . import config
else:
    import config

import wandb

class LinearDataset(Dataset):
    def __init__(self, dataset: dict):
        super().__init__()

        self.states = torch.from_numpy(dataset["states"])  # (Step,16)
        self.actions = torch.from_numpy(dataset["actions"])  # (Step,4)
        self.next_states = torch.from_numpy(dataset["next_states"])  # (Step,16)
        self.rewards = torch.from_numpy(dataset["rewards"]).reshape(-1,1)  # (Step,1)
        self.terminations = torch.from_numpy(
            dataset["terminations"].astype(np.float32)
        ).reshape(-1,1)  # (Step,1)

    def __len__(self) -> int:
        return self.rewards.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {
                "states": self.states[idx],  # (B,16)
                "actions": self.actions[idx],  # (B,4)
            },
            {
                "next_states": self.next_states[idx],  # (B,16)
                "rewards": self.rewards[idx],  # (B,1)
                "terminations": self.terminations[idx],  # (B,1)
            },
        )


class RolloutDataset(Dataset):
    def __init__(self, dataset: dict, horizon: int = 10):
        super().__init__()
        states = dataset["states"]
        actions = dataset["actions"]
        next_states = dataset["next_states"]
        rewards = dataset["rewards"]
        terminations = dataset["terminations"]

        self.trajectories = []
        for s, a, n_s, r, t in zip(states, actions, next_states, rewards, terminations):
            if s.shape[0] >= horizon:
                self.trajectories.append(
                    {
                        "initial_state": torch.from_numpy(s[0]),  # (16,)
                        "actions": torch.from_numpy(a[:horizon, :]),  # (H,4)
                        "next_states": torch.from_numpy(n_s[:horizon, :]),  # (H,16)
                        "rewards": torch.from_numpy(r[:horizon, :]).reshape(-1,1),  # (H,1)
                        "terminations": torch.from_numpy(
                            t[:horizon, :].astype(np.float32)
                        ).reshape(-1,1),  # (H,1)
                    }
                )

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        return (
            {
                "initial_state": self.trajectories[idx]["initial_state"],
                "actions": self.trajectories[idx]["actions"],
            },
        )  # (B,16) and (B,H,4)
        {
            "next_states": self.trajectories[idx]["next_states"],
            "rewards": self.trajectories[idx]["rewards"],
            "terminations": self.trajectories[idx]["terminations"],
        }  # (B, H, 16) and (B,H,16) and (B,H,1)


class ModelM(nn.Module):
    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        num_action: int = config.NUM_ACTION,
        hidden_dim=config.M_HIDDEN_DIM,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + num_action, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.delta_z = nn.Linear(hidden_dim, latent_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.termination = nn.Linear(hidden_dim, 1)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(torch.cat([z, action], dim=-1))
        delta_z = self.delta_z(hidden)
        reward = self.reward(hidden)
        termination = self.termination(hidden)
        next_z = z + delta_z
        return next_z, reward, termination


def get_data_loader(mode: str, rollout: bool = False, horizon: int = 10) -> DataLoader:
    if mode == "train":
        dataset_path = config.ENCODED_TRAIN_DATASET_PATH
        allow_shuffle = True
    elif mode == "val":
        dataset_path = config.ENCODED_VAL_DATASET_PATH
        allow_shuffle = False
    else:
        raise ValueError

    with np.load(dataset_path, allow_pickle=True) as dataset:
        states = dataset["states"]
        actions = dataset["actions"]
        next_states = dataset["next_states"]
        rewards = dataset["rewards"]
        terminations = dataset["terminated"]

    action_one_hot = [config.ACTION_LOOKUP[action.reshape(-1)] for action in actions] #(Episodes, steps, 4)

    transformed_dataset = {
        "states": states
        if rollout
        else np.concatenate(states, axis=0),  # (Total_episodes,) or (Steps, 16)
        "actions": np.array(action_one_hot, dtype=object)
        if rollout
        else np.concatenate(action_one_hot, axis=0),  # (Total_episodes,) or (Steps, 4)
        "next_states": next_states
        if rollout
        else np.concatenate(next_states, axis=0),  # (Total_episodes,) or (Steps, 16)
        "rewards": rewards
        if rollout
        else np.concatenate(rewards, axis=0),  # (Total_episodes,) or (Steps,)
        "terminations": terminations
        if rollout
        else np.concatenate(terminations, axis=0),  # (Total_episodes,) or (Steps,)
    }
    return DataLoader(
        RolloutDataset(transformed_dataset, horizon)
        if rollout
        else LinearDataset(transformed_dataset),
        batch_size=config.BATCH_SIZE,
        shuffle=allow_shuffle,
    )


def compute_loss(
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targ: dict[str, torch.Tensor],
    pos_weight: torch.Tensor | None = None,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[torch.Tensor, dict[str, float]]:

    pred_z, pred_reward, pred_term = pred
    w_z, w_r, w_t = weights

    loss_s = F.mse_loss(pred_z, targ["next_states"])
    loss_r = F.mse_loss(pred_reward, targ["rewards"])
    loss_t = F.binary_cross_entropy_with_logits(
        pred_term, targ["terminations"], pos_weight=pos_weight
    )

    total_loss = w_z * loss_s + w_r * loss_r + w_t * loss_t
    loss_info = {
        "loss_s": loss_s.item(),
        "loss_r": loss_r.item(),
        "loss_t": loss_t.item(),
    }
    return total_loss, loss_info


def compute_pos_weight(termination):
    total_steps = termination.shape[0]
    total_term_steps = termination.sum().clamp(min=1.0)
    total_safe_steps = total_steps - total_term_steps.clamp(min=1.0)
    return total_safe_steps / total_term_steps


def train(
    dynamic_model: ModelM,
    train_loader: DataLoader,
    optimizer: torch.optim.Adam,
    device: torch.device,):

    dynamic_model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = (
            {k: v.to(device) for k, v in inputs.items()},
            {k: v.to(device) for k, v in targets.items()},
        )
        optimizer.zero_grad()
        prediction = dynamic_model(*inputs.values())

        pos_weight = compute_pos_weight(targets["terminations"])
        train_loss, loss_info = compute_loss(prediction, targets, pos_weight=pos_weight)

        total_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def val(dynamic_model: ModelM, val_loader: DataLoader, device: torch.device):

    dynamic_model.eval()
    total_loss = 0

    for inputs, targets in val_loader:
        with torch.no_grad():
            inputs, targets = (
                {k: v.to(device) for k, v in inputs.items()},
                {k: v.to(device) for k, v in targets.items()},
            )
            prediction = dynamic_model(*inputs.values())

            pos_weight = compute_pos_weight(targets["terminations"])
            val_loss, loss_info = compute_loss(
                prediction, targets, pos_weight=pos_weight
            )

            total_loss += val_loss.item()

    return total_loss / len(val_loader)


def main():
    wandb.init(
        project="World-model-grid-model-m",
        name="test-run",
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "total_epochs": config.TOTAL_EPOCHS,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dynamic_model = ModelM().to(device)
    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=config.LEARNING_RATE)

    train_loader = get_data_loader("train")
    val_loader = get_data_loader("val")

    best_val_loss = float("inf")

    for epoch in range(config.TOTAL_EPOCHS):
        avg_train_loss = train(dynamic_model, train_loader, optimizer, device)
        avg_val_loss = val(dynamic_model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{config.TOTAL_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        wandb.log(
            {
                "Train_loss": avg_train_loss,
                "Val_loss": avg_val_loss,
                "Epoch": epoch + 1,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dynamic_model.state_dict(), config.MODEL_M_PARAMETERS_PATH)

    wandb.save(config.MODEL_M_PARAMETERS_PATH)
    wandb.finish()

if __name__ == "__main__":
    main()