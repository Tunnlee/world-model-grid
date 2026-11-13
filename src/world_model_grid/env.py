import numpy as np
try:
    from . import config
except ImportError:
    import config


class GridEnv8x8:
    """Môi trường lưới 8×8 với vật cản tĩnh và một agent.

    Observation dùng ba channel: ô trống, vật cản và vị trí agent.
    Episode kết thúc tự nhiên khi agent va chạm; episode đang sống bị
    truncated khi đạt giới hạn số bước.
    """

    def __init__(self):
        """Khởi tạo môi trường trước khi bắt đầu một episode.

        Các giá trị cấu hình được sao chép vào instance. Vị trí agent và
        vật cản chỉ được tạo khi gọi reset().
        """
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
        """Tạo episode mới và trả về observation ban đầu.

        Parameters
        ----------
        seed : int or None, optional
            Seed dùng để khởi tạo lại bộ sinh số ngẫu nhiên. Cùng một seed
            sẽ tạo cùng bố trí vật cản và vị trí agent.

        Returns
        -------
        observation : numpy.ndarray
            State ban đầu có shape (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
            và dtype float32.
        info : dict
            Thông tin phụ; hiện là dictionary rỗng.
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
        """Mã hóa bố trí hiện tại thành observation dạng one-hot theo ô.

        Returns
        -------
        numpy.ndarray
            Mảng float32 có shape (3, GRID_SIZE, GRID_SIZE). Channel 0 đánh
            dấu ô trống, channel 1 đánh dấu vật cản, channel 2 đánh dấu agent.

        Notes
        -----
        Cần gọi reset() trước để đặt vị trí agent.
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
        """Thực hiện một hành động và chuyển môi trường sang bước kế tiếp.

        Parameters
        ----------
        action : int
            ID hành động, phải là một trong các giá trị trong ACTION_MAP.

        Returns
        -------
        observation : numpy.ndarray
            State sau hành động. Nếu va chạm, agent giữ nguyên vị trí.
        reward : float or int
            Reward của bước đi: REWARD_SAFE_STEP nếu an toàn, hoặc
            REWARD_COLLISION nếu va chạm/ra ngoài biên.
        terminated : bool
            True nếu agent kết thúc episode do va chạm hoặc ra ngoài biên.
        truncated : bool
            True nếu episode còn sống nhưng đạt MAX_STEPS_PER_EPISODE.
        info : dict
            Thông tin phụ; hiện là dictionary rỗng.

        Raises
        ------
        KeyError
            Nếu action không có trong ACTION_MAP.

        Notes
        -----
        Phải gọi reset() trước step(). Người gọi nên dừng episode khi
        terminated hoặc truncated trở thành True.
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
        if self.is_collision(new_row, new_col) or self.is_out_of_bounds(new_row, new_col):
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
        """Kiểm tra tọa độ có nằm ngoài lưới hay không.

        Parameters
        ----------
        row : int
            Chỉ số hàng cần kiểm tra.
        col : int
            Chỉ số cột cần kiểm tra.

        Returns
        -------
        bool
            True nếu hàng hoặc cột nằm ngoài khoảng hợp lệ.
        """
        return not (0 <= row < self.grid_size and 0 <= col < self.grid_size) 

    def is_collision(self, row: int, col: int) -> bool:
        """Kiểm tra tọa độ có trùng với một vật cản hay không.

        Parameters
        ----------
        row : int
            Chỉ số hàng cần kiểm tra.
        col : int
            Chỉ số cột cần kiểm tra.

        Returns
        -------
        bool
            True nếu tọa độ thuộc tập vật cản.
        """
        return (row, col) in self.obstacles