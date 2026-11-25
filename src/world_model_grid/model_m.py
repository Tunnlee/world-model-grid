import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

if __package__:
    from . import config
else:
    import config


class LinearDataset(Dataset):
    def __init__(self, dataset: dict):
        super().__init__()

        self.states = torch.from_numpy(dataset["states"])
        self.actions = torch.from_numpy(dataset["actions"])
        self.next_states = torch.from_numpy(dataset["next_states"])
        self.rewards = torch.from_numpy(dataset["rewards"])
        self.terminations = torch.from_numpy(dataset["terminations"].astype(np.float32))

    def __len__(self) -> int:
        return self.rewards.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {
                "states": self.states[idx],
                "actions": self.actions[idx],
            },
            {
                "next_states": self.next_states[idx],
                "rewards": self.rewards[idx],
                "terminations": self.terminations[idx],
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
                        "initial_state": torch.from_numpy(s[0]),
                        "actions": torch.from_numpy(a[:horizon, :]),
                        "next_states": torch.from_numpy(n_s[:horizon, :]),
                        "rewards": torch.from_numpy(r[:horizon, :]),
                        "terminations": torch.from_numpy(t[:horizon, :].astype(np.float32)),
                    }
                )

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        return {
            "initial_state": self.trajectories[idx]["initial_state"],
            "actions": self.trajectories[idx]["actions"],
        }, {
            "next_states": self.trajectories[idx]["next_states"],
            "rewards": self.trajectories[idx]["rewards"],
            "terminations": self.trajectories[idx]["terminations"],
        }


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
        terminations = dataset["terminations"]

    action_one_hot = [config.ACTION_LOOKUP[action.reshape(-1)] for action in actions]

    transformed_dataset = {
        "states": states if rollout else np.concatenate(states, axis=0),
        "actions": np.array(action_one_hot, dtype=object)
        if rollout
        else np.concatenate(action_one_hot, axis=0),
        "next_states": next_states if rollout else np.concatenate(next_states, axis=0),
        "rewards": rewards if rollout else np.concatenate(rewards, axis=0),
        "terminations": terminations
        if rollout
        else np.concatenate(terminations, axis=0),
    }
    return DataLoader(
        RolloutDataset(transformed_dataset, horizon)
        if rollout
        else LinearDataset(transformed_dataset),
        batch_size=config.BATCH_SIZE,
        shuffle=allow_shuffle,
    )
