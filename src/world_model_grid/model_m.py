import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

if __package__:
    from . import config
else:
    import config


class PredDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.states = torch.from_numpy(dataset["states"])
        self.actions = torch.from_numpy(dataset["actions"])
        self.next_states = torch.from_numpy(dataset["next_states"])
        self.rewards = torch.from_numpy(dataset["rewards"])
        self.terminations = torch.from_numpy(dataset["terminations"])

    def __len__(self) -> int:
        return self.rewards.shape[0]

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return ({
                "states": self.states[idx],
                "actions": self.actions[idx],
            },
        
        {
            "next_states": self.next_states[idx],
            "rewards": self.rewards[idx],
            "terminations": self.terminations[idx],
        })


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

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(torch.cat([z, action], dim=-1))
        delta_z = self.delta_z(hidden)
        reward = self.reward(hidden)
        termination = self.termination(hidden)
        next_z = z + delta_z
        return next_z, reward, termination


def get_data_loader(mode: str) -> DataLoader:
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
        terminations = dataset["terminations"]

    action_one_hot = [config.ACTION_LOOKUP[action.reshape(-1)] for action in actions]

    transformed_dataset = {
        "states": np.concatenate(states, axis=0),
        "actions": np.concatenate(action_one_hot, axis=0),
        "next_states": np.concatenate(next_states, axis=0),
        "rewards": np.concatenate(rewards, axis=0),
        "terminations": np.concatenate(terminations, axis=0),
    }

    return DataLoader(
        PredDataset(transformed_dataset),
        batch_size=config.BATCH_SIZE,
        shuffle=allow_shuffle,
    )
