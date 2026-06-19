import numpy as np

class GridWorld(object):
    
    WALL = 1
    AGENT = 2
    TREASURE = 3
    SMALL_TREASURE = 4


    def __init__(self, size=20, p_walls=0.5, agent_start=(0,0)):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.size = size
        self.p_walls = p_walls

        self.agent_pos = agent_start
        self.treasure_pos = (size-1, size-1)
        self.streasure_pos = (0, size-1)
    
        self._init_grid()

    def _init_grid(self):
        for r in range(self.size):
            for c in range(self.size):
                if not (r % 2 == 0):
                   if np.random.rand() < self.p_walls:
                        if (r,c) != self.agent_pos and (r,c) != self.treasure_pos:
                            self.grid[r,c] = self.WALL

        self.grid[self.agent_pos] = self.AGENT
        self.grid[self.treasure_pos] = self.TREASURE
        self.grid[self.streasure_pos] = self.SMALL_TREASURE


    def get_actions(self):
        

    def do_action(self):
        pass
