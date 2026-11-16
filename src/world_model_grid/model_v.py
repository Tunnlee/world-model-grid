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

        self.inputs = torch.from_numpy(states.reshape(-1, flatten_dim)) Shape (N, 128)
        self.targets = torch.from_numpy(states.argmax(axis=1)) #Shape (N, 8, 8)

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



def quick_test():
    train_dataset = GridDataset(train_states)

    x, y = train_dataset[0,:]

    print("Input :", x.shape, x.dtype)
    print("Target:", y.shape, y.dtype)