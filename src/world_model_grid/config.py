import os

# 1. 8x8 Grid
GRID_SIZE = 8
NUM_CHANNELS = 3  # 3 lớp thông tin: [0: Ô trống, 1: Vật cản, 2: Agent]
STATE_SHAPE = (NUM_CHANNELS, GRID_SIZE, GRID_SIZE) # (C, H, W)

MIN_OBSTACLES = 5  # Số vật cản tối thiểu
MAX_OBSTACLES = 10  # Số vật cản tối đa
MAX_STEPS_PER_EPISODE = 100  # Cho nhân vật đi tối đa 100 bước mỗi lượt chơi

# 2. Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_MAP = {
    ACTION_UP: (-1, 0),  # Lên: Giảm 1 hàng
    ACTION_DOWN: (1, 0),  # Xuống: Tăng 1 hàng
    ACTION_LEFT: (0, -1),  # Trái: Giảm 1 cột
    ACTION_RIGHT: (0, 1),  # Phải: Tăng 1 cột
}

# 3. Rewards
REWARD_SAFE_STEP = 1.0  # Đi 1 bước an toàn: cộng 1 điểm
REWARD_COLLISION = -10.0  # Tông đá hoặc va biên: trừ 10 điểm

# 4. File paths
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.npz")
