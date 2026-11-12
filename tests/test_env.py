import numpy as np
import pytest
from world_model_grid import config
from world_model_grid.env import GridEnv8x8


def test_init_and_reset():
    env = GridEnv8x8()
    assert env.grid_size == config.GRID_SIZE
    assert env.min_obstacles == config.MIN_OBSTACLES
    assert env.max_obstacles == config.MAX_OBSTACLES
    assert env.agent_position is None
    assert env.obstacles == set()
    assert env.current_step == 0

    state, info = env.reset(seed=42)
    assert env.min_obstacles <= len(env.obstacles) <= env.max_obstacles
    assert env.agent_position not in env.obstacles
    assert state.shape == config.STATE_SHAPE
    assert np.sum(state[1]) == len(env.obstacles)  # Check obstacle channel
    assert np.sum(state[2]) == 1
    assert (
        np.sum(state[0]) == config.GRID_SIZE**2 - len(env.obstacles) - 1
    )  # Check blank channel

    env2 = GridEnv8x8()
    state2, _ = env2.reset(seed=42)
    assert np.array_equal(state, state2)


def test_step_direction_and_step_count():
    env = GridEnv8x8()
    env.reset(seed=42)
    env.agent_position = (4, 4)
    env.obstacles = set()
    assert env.current_step == 0

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_UP)
    assert env.agent_position == (3, 4)
    assert reward == config.REWARD_SAFE_STEP
    assert not terminated and not truncated
    assert next_state[2, 3, 4] == 1.0
    assert env.current_step == 1

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_DOWN)
    assert env.agent_position == (4, 4)
    assert reward == config.REWARD_SAFE_STEP
    assert not terminated and not truncated
    assert next_state[2, 4, 4] == 1.0
    assert env.current_step == 2

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_LEFT)
    assert env.agent_position == (4, 3)
    assert reward == config.REWARD_SAFE_STEP
    assert not terminated and not truncated
    assert next_state[2, 4, 3] == 1.0
    assert env.current_step == 3

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_RIGHT)
    assert env.agent_position == (4, 4)
    assert reward == config.REWARD_SAFE_STEP
    assert not terminated and not truncated
    assert next_state[2, 4, 4] == 1.0
    assert env.current_step == 4


def test_collision():
    env = GridEnv8x8()
    env.reset(seed=42)
    env.agent_position = (4, 4)
    env.obstacles = {(3, 4)}  # Place an obstacle above the agent
    next_state, reward, terminated, truncated, info = env.step(config.ACTION_UP)
    assert terminated
    assert reward == config.REWARD_COLLISION
    assert env.agent_position == (4, 4)  # Agent should not move into the obstacle


def test_out_of_bounds():
    env = GridEnv8x8()
    env.reset(seed=42)
    env.obstacles = set()  # No obstacles
    env.agent_position = (0, 0)  # Top-left corner

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_UP)
    assert terminated
    assert reward == config.REWARD_COLLISION
    assert env.agent_position == (0, 0)  # Agent should not move out of bounds

    env.agent_position = (7, 7)  # Bottom-right corner
    next_state, reward, terminated, truncated, info = env.step(config.ACTION_DOWN)
    assert terminated
    assert reward == config.REWARD_COLLISION
    assert env.agent_position == (7, 7)  # Agent should not move out


def test_truncation():
    env = GridEnv8x8()
    env.reset(seed=42)
    env.obstacles = set()  # No obstacles
    env.agent_position = (4, 4)  # Start in the middle

    for _ in range(config.MAX_STEPS_PER_EPISODE - 1):
        next_state, reward, terminated, truncated, info = (
            env.step(config.ACTION_UP) if _ % 2 == 0 else env.step(config.ACTION_DOWN)
        )
        assert not terminated and not truncated

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_UP)
    assert not terminated
    assert truncated


def test_collision_on_last_step():
    env = GridEnv8x8()
    env.reset(seed=42)
    env.obstacles = {(3, 5)}  # Place an obstacle above the agent
    env.agent_position = (4, 4)  # Start in the middle

    for _ in range(config.MAX_STEPS_PER_EPISODE - 1):
        next_state, reward, terminated, truncated, info = (
            env.step(config.ACTION_RIGHT)
            if _ % 2 == 0
            else env.step(config.ACTION_LEFT)
        )
        assert not terminated and not truncated

    next_state, reward, terminated, truncated, info = env.step(config.ACTION_UP)
    assert terminated
    assert reward == config.REWARD_COLLISION
    assert not truncated
