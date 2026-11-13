import numpy as np

from . import config
from .env import GridEnv8x8

dataset = []
env = GridEnv8x8()

def main():
    # 70% Random policy
    for episode in range(int(config.TOTAL_EPISODES * 0.7)):
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
    for episode in range(int(config.TOTAL_EPISODES * 0.3)):
        state, _ = env.reset(seed=episode + int(config.TOTAL_EPISODES * 0.7))

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
        config.DATASET_PATH,
        states=np.array([ep["states"] for ep in dataset], dtype=object),
        actions=np.array([ep["actions"] for ep in dataset], dtype=object),
        rewards=np.array([ep["rewards"] for ep in dataset], dtype=object),
        next_states=np.array([ep["next_states"] for ep in dataset], dtype=object),
        terminated=np.array([ep["terminated"] for ep in dataset], dtype=object),
    )


if __name__ == "__main__":
    main()