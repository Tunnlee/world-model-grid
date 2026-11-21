import os

# 1. 8x8 Grid
GRID_SIZE = 8
NUM_CHANNELS = 3  
STATE_SHAPE = (NUM_CHANNELS, GRID_SIZE, GRID_SIZE) 

MIN_OBSTACLES = 5  
MAX_OBSTACLES = 10 
MAX_STEPS_PER_EPISODE = 100 

# 2. Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_MAP = {
    ACTION_UP: (-1, 0),  
    ACTION_DOWN: (1, 0), 
    ACTION_LEFT: (0, -1),  
    ACTION_RIGHT: (0, 1), 
}

# 3. Rewards
REWARD_SAFE_STEP = 1.0  
REWARD_COLLISION = -10.0 

# 4. File paths
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATASET_PATH = os.path.join(DATA_DIR, "train_dataset.npz")
VAL_DATASET_PATH = os.path.join(DATA_DIR, "val_dataset.npz")
WEIGHT_DIR = os.path.join(PROJECT_ROOT, "weights")
MODEL_V_PARAMETERS_PATH = os.path.join(WEIGHT_DIR, "model_v_parameters.pth")
MODEL_V_RECON_PATH = os.path.join(PROJECT_ROOT, "results", "model_v_recon.png")


# 5 Collect data
TOTAL_EPISODES_TRAIN = 100000
TOTAL_EPISODES_VAL = 10000

# 6 Model V
TOTAL_EPOCHS = 50
BATCH_SIZE = 256
FLATTEN_DIM = GRID_SIZE * GRID_SIZE * NUM_CHANNELS
HIDDEN_DIM = 64
LATENT_DIM = 16
LEARNING_RATE = 0.0015
NUM_SAMPLES = 6
