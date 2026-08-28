import numpy as np

def a2idx(a: np.ndarray) -> tuple:
    """Convert a numpy array to a tuple"""
    return (int(a[0]), int(a[1]))

class GridWorldConfig(object): 

    def __init__(
        self, 
        size: int, 
        p_walls: float, 
        agent_start: np.ndarray, 
        step_penalty: float, 
        small_treasure_rew: float, 
        treasure_rew: float, 
        sd_treasure: float, 
        sd_small_treasure: float, 
        temperature: float = 0.1, 
        gamma: float = 0.99, 
        random_state: int = 0
    ):
        self.size = size 
        self.p_walls = p_walls
        self.agent_start = agent_start
        self.step_penalty = step_penalty
        self.small_treasure_rew = small_treasure_rew
        self.treasure_rew = treasure_rew
        self.sd_treasure = sd_treasure
        self.sd_small_treasure = sd_small_treasure
        self.temperature = temperature
        self.gamma = gamma
        self.random_state = random_state

    def __repr__(self):
        params = "\n  ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"GridWorldConfig(\n  {params}\n)"

def compute_returns_np(r, gamma):
    r = np.array(r)
    returns = np.zeros_like(r, dtype=float)
    g = 0
    for t in reversed(range(len(r))):
        g = r[t] + gamma * g
        returns[t] = g
    return returns
