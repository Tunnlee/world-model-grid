import numpy as np
try:
    from . import config
except ImportError:
    import config

class GridEnv8x8:
    def __init__(self):
        #Config values
        self.grid_size = config.GRID_SIZE
        self.min_obstacles = config.MIN_OBSTACLES
        self.max_obstacles = config.MAX_OBSTACLES

        #Dynamic states
        self.agent_position = None #Agent yet exists
        self.obstacles = set() #Set of tuples
        self.current_step = 0
        self.rng = np.random.default_rng() 

    def reset(self, seed = None):
        self.current_step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        obstacles_num = self.rng.integers(low = self.min_obstacles, high = self.max_obstacles + 1)
        choice  = self.rng.choice(self.grid_size ** 2, size = (obstacles_num + 1,), replace = False)

        rows = choice // self.grid_size
        cols = choice % self.grid_size
        self.obstacles = set(zip(rows[:-1], cols[:-1]))
        self.agent_position = (rows[-1], cols[-1])
        return self.get_state()


    def get_state(self):
        states = np.zeros((config.NUM_CHANNELS, self.grid_size, self.grid_size), dtype=np.float32)
        states[0, *self.agent_position] = 1.0
        states[1, *zip(*self.obstacles)] = 1.0
        return states
        