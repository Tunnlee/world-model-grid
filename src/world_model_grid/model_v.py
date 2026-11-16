import numpy as np
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GridDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        flatten_dim: int = config.GRID_SIZE**2 * config.NUM_CHANNELS,
    ):
        super().__init__()

        if states.ndim != 4:
            raise ValueError("States must be a 4D array")

        self.inputs = torch.from_numpy(
            states.reshape(-1, flatten_dim)
        )  # Shape (N, 192)
        self.targets = torch.from_numpy(states.argmax(axis=1))  # Shape (N, 8, 8)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def load_dataset():
    with np.load(config.DATASET_PATH, allow_pickle=True) as data:
        dataset = data["states"]
    train_states = np.concatenate(
        dataset[: int(config.TOTAL_EPISODES * 0.9)], axis=0
    )  # Concatenate iterates list of pointers
    val_states = np.concatenate(dataset[int(config.TOTAL_EPISODES * 0.9) :], axis=0)
    return train_states, val_states


def weight_distrubution(dataset):
    counts = torch.bincount(dataset.targets.flatten(), minlength=config.NUM_CHANNELS)
    class_weights = (counts.sum() / (config.NUM_CHANNELS * counts)).float()


def train():
    train_states, val_states = load_dataset()
    train_dataset = GridDataset(train_states)

    class_weights = weight_distribution(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)




def quick_test():
    train_states, _ = load_dataset()
    train_dataset = GridDataset(train_states)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    batch = next(iter(train_loader))
    print(batch[0].shape, batch[0].type)
    print(batch[1].shape)


quick_test()
