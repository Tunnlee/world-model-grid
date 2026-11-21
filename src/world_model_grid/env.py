import numpy as np

if __package__:
    from . import config
else:
    import config


class GridEnv8x8:
    """8×8 grid environment with static obstacles and one agent."""

    def __init__(self):
        """Initialize configuration, empty state, and random generator."""
        # Config values
        self.grid_size = config.GRID_SIZE
        self.min_obstacles = config.MIN_OBSTACLES
        self.max_obstacles = config.MAX_OBSTACLES

        # Dynamic states
        self.agent_position = None  # Agent yet exists
        self.obstacles = set()  # Set of tuples
        self.current_step = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Start a new episode and return its initial observation.

        Parameters
        ----------
        seed : int or None
            Optional seed for reproducible layouts.

        Returns
        -------
        tuple
            Initial state and an empty info dictionary.
        """
        self.current_step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obstacles_num = self.rng.integers(
            low=self.min_obstacles, high=self.max_obstacles + 1
        )
        choice = self.rng.choice(
            self.grid_size**2, size=(obstacles_num + 1,), replace=False
        )

        rows = choice // self.grid_size  # 1D Array
        cols = choice % self.grid_size
        self.obstacles = set(zip(rows[:-1], cols[:-1]))
        self.agent_position = (rows[-1], cols[-1])
        return self.get_state(), {}

    def get_state(self) -> np.ndarray:
        """Return the current grid as a three-channel float32 array.

        Returns
        -------
        numpy.ndarray
            State shaped (3, GRID_SIZE, GRID_SIZE). Call reset() first.
        """
        states = np.zeros(
            (config.NUM_CHANNELS, self.grid_size, self.grid_size), dtype=np.float32
        )
        states[0, :, :] = 1.0  # Blanks

        states[1, *zip(*self.obstacles)] = 1.0  # Obstacles
        states[0, *zip(*self.obstacles)] = 0.0  # Remove blanks

        states[2, *self.agent_position] = 1.0  # Agent
        states[0, *self.agent_position] = 0.0  # Remove blanks

        return states

    def step(self, action: int) -> tuple[np.ndarray, float | int, bool, bool, dict]:
        """Apply an action and return the resulting transition.

        Parameters
        ----------
        action : int
            Action ID from ACTION_MAP.

        Returns
        -------
        tuple
            State, reward, terminated, truncated, and info.

        Raises
        ------
        KeyError
            If action is not in ACTION_MAP.
        """
        # Set up flags
        terminated = False
        truncated = False

        # Update agent position and step count
        self.current_step += 1
        a_rows, a_cols = config.ACTION_MAP[action]
        new_row = self.agent_position[0] + a_rows
        new_col = self.agent_position[1] + a_cols

        # Check for termination
        if self.is_collision(new_row, new_col) or self.is_out_of_bounds(
            new_row, new_col
        ):
            reward = config.REWARD_COLLISION
            terminated = True
        else:
            reward = config.REWARD_SAFE_STEP
            self.agent_position = (new_row, new_col)

        # Check for truncation
        if self.current_step == config.MAX_STEPS_PER_EPISODE:
            if terminated:
                truncated = False
            else:
                truncated = True

        # Return
        return self.get_state(), reward, terminated, truncated, {}

    def is_out_of_bounds(self, row: int, col: int) -> bool:
        """Check whether a grid coordinate is outside the board.

        Parameters
        ----------
        row, col : int
            Grid coordinates.

        Returns
        -------
        bool
            True if the coordinate is outside the board.
        """
        return not (0 <= row < self.grid_size and 0 <= col < self.grid_size)

    def is_collision(self, row: int, col: int) -> bool:
        """Check whether a coordinate contains an obstacle.

        Parameters
        ----------
        row, col : int
            Grid coordinates.

        Returns
        -------
        bool
            True if the coordinate is blocked.
        """
        return (row, col) in self.obstacles
