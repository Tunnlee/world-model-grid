"""Thu thập dữ liệu episode cho môi trường world_model_grid."""

import sys
from pathlib import Path

import numpy as np

from . import config
from .env import GridEnv8x8


def main(mode: str) -> None:
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
    if mode not in {"train", "val"}:
        raise ValueError("mode must be either 'train' or 'val'")

    if mode == "train":
        dataset = []
        env = GridEnv8x8()
        # 70% Random policy
        for episode in range(int(config.TOTAL_EPISODES_TRAIN * 0.7)):
            state, _ = env.reset(seed=episode)

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
        for episode in range(int(config.TOTAL_EPISODES_TRAIN * 0.3)):
            state, _ = env.reset(seed=episode + int(config.TOTAL_EPISODES_TRAIN * 0.7))

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

                    if not env.is_out_of_bounds(
                        new_row, new_col
                    ) and not env.is_collision(new_row, new_col):
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

        Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
        np.savez(
            config.TRAIN_DATASET_PATH,
            states=np.array([ep["states"] for ep in dataset], dtype=object),
            actions=np.array([ep["actions"] for ep in dataset], dtype=object),
            rewards=np.array([ep["rewards"] for ep in dataset], dtype=object),
            next_states=np.array([ep["next_states"] for ep in dataset], dtype=object),
            terminated=np.array([ep["terminated"] for ep in dataset], dtype=object),
        )

    elif mode == "val":
        dataset = []
        env = GridEnv8x8()
        # 70% Random policy
        for episode in range(int(config.TOTAL_EPISODES_VAL * 0.7)):
            state, _ = env.reset(seed=episode + config.TOTAL_EPISODES_TRAIN)

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
        for episode in range(int(config.TOTAL_EPISODES_VAL * 0.3)):
            state, _ = env.reset(
                seed=episode
                + int(config.TOTAL_EPISODES_VAL * 0.7)
                + config.TOTAL_EPISODES_TRAIN
            )

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

                    if not env.is_out_of_bounds(
                        new_row, new_col
                    ) and not env.is_collision(new_row, new_col):
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

        Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
        np.savez(
            config.VAL_DATASET_PATH,
            states=np.array([ep["states"] for ep in dataset], dtype=object),
            actions=np.array([ep["actions"] for ep in dataset], dtype=object),
            rewards=np.array([ep["rewards"] for ep in dataset], dtype=object),
            next_states=np.array([ep["next_states"] for ep in dataset], dtype=object),
            terminated=np.array([ep["terminated"] for ep in dataset], dtype=object),
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m world_model_grid.collect_data {train,val}")
    main(sys.argv[1].strip().lower())
