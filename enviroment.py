import numpy as np

class GridWorld(object):
    
    WALL = 1
    AGENT = 2
    TREASURE = 3


    def __init__(self, size=20, p_walls=0.5, agent_start=(0,0)):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.size = size
        self.p_walls = p_walls

        self.agent_position = agent_start
        self.treasure_position = (size-1, size-1)
    
        self._init_grid()

    def _init_grid(self):
        for r in range(self.size):
            for c in range(self.size):
                if not (r % 2 == 0):
                   if np.random.rand() < self.p_walls:
                        if (r,c) != self.agent_position and (r,c) != self.treasure_position:
                            self.grid[r,c] = self.WALL

        self.grid[self.agent_position] = self.AGENT
        self.grid[self.treasure_position] = self.TREASURE


    def get_actions(self):
        pass

    def do_action(self):
        pass
