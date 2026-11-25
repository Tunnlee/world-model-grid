"""Thu thập dữ liệu episode cho môi trường world_model_grid."""

import sys

import numpy as np


def collect(mode: str) -> None:
    """Collect and save the requested dataset split.

    Parameters
    ----------
    mode : {"train", "val"}
        Split to collect.

    Raises
    ------
    ValueError
        If mode is not "train" or "val".

    Notes
    -----
    Each split uses 70% random and 30% one-step safe episodes. Ragged episode
    arrays are saved to the configured NPZ path; truncated flags are omitted.
    """
    if __package__:
        from . import config
        from .env import GridEnv8x8
    else:
        import config
        from env import GridEnv8x8
    if mode not in {"train", "val"}:
        raise ValueError("mode must be either 'train' or 'val'")

    elif mode == "train":
        episodes = config.TOTAL_EPISODES_TRAIN
        seed_addition = 0
        destination = config.TRAIN_DATASET_PATH
    elif mode == "val":
        episodes = config.TOTAL_EPISODES_VAL
        seed_addition = config.TOTAL_EPISODES_TRAIN
        destination = config.VAL_DATASET_PATH
    dataset = []
    env = GridEnv8x8()
    # 70% Random policy
    for episode in range(int(episodes * 0.7)):
        state, _ = env.reset(seed=episode + seed_addition)

        states = []
        actions = []
        rewards = []
        next_states = []
        terminateds = []

        for step in range(config.MAX_STEPS_PER_EPISODE):
            action = env.rng.integers(0, 4)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)

            state = next_state
            if done:
                break

        dataset.append(
            {
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
                "next_states": np.array(next_states, dtype=np.float32),
                "terminated": np.array(terminateds, dtype=np.bool_),
            }
        )

    # 30% Safe Random Walk
    for episode in range(int(episodes * 0.3)):
        state, _ = env.reset(seed=episode + int(episodes * 0.7) + seed_addition)

        states = []
        actions = []
        rewards = []
        next_states = []
        terminateds = []

        for step in range(config.MAX_STEPS_PER_EPISODE):
            valid_actions = []
            for action in range(4):
                a_rows, a_cols = config.ACTION_MAP[action]
                new_row = env.agent_position[0] + a_rows
                new_col = env.agent_position[1] + a_cols

                if not env.is_out_of_bounds(new_row, new_col) and not env.is_collision(
                    new_row, new_col
                ):
                    valid_actions.append(action)

            if valid_actions:
                action = env.rng.choice(valid_actions)
            else:
                action = env.rng.integers(0, 4)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)

            state = next_state
            if done:
                break

        dataset.append(
            {
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
                "next_states": np.array(next_states, dtype=np.float32),
                "terminated": np.array(terminateds, dtype=np.bool_),
            }
        )

    rng = np.random.default_rng(seed=42)
    rng.shuffle(dataset)

    np.savez(
        destination,
        states=np.array([ep["states"] for ep in dataset], dtype=object), 
        actions=np.array([ep["actions"] for ep in dataset], dtype=object),
        rewards=np.array([ep["rewards"] for ep in dataset], dtype=object),
        next_states=np.array([ep["next_states"] for ep in dataset], dtype=object),
        terminated=np.array([ep["terminated"] for ep in dataset], dtype=object),
    )


def encode(mode):
    import torch
    import numpy as np

    if __package__:
        from . import config
        from world_model_grid.model_v import ModelV
    else:
        import config
        from model_v import ModelV

    if mode == "train":
        file_path = config.TRAIN_DATASET_PATH
        destination = config.ENCODED_TRAIN_DATASET_PATH
    elif mode == "val":
        file_path = config.VAL_DATASET_PATH
        destination = config.ENCODED_VAL_DATASET_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(file_path, allow_pickle=True)
    vision_model = ModelV().to(device)
    vision_model.load_state_dict(
        torch.load(
            config.MODEL_V_PARAMETERS_PATH, map_location=device, weights_only=True
        )
    )
    vision_model.eval()

    with torch.no_grad():
        states = [
            vision_model.encoder(torch.from_numpy(state).to(device).reshape(-1, 192))
            .cpu()
            .numpy()
            for state in data["states"]
        ]
        next_states = [
            vision_model.encoder(
                torch.from_numpy(next_state).to(device).reshape(-1, 192)
            )
            .cpu()
            .numpy()
            for next_state in data["next_states"]
        ]
    np.savez(
        destination,
        states=np.array(states, dtype=object),
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=np.array(next_states, dtype=object),
        terminated=data["terminated"],
    )


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        main(sys.argv[2].strip().lower())
    if sys.argv[1] == "encode":
        encode(sys.argv[2].strip().lower())
