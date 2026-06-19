import numpy as np

class GridWorld(object):
    
    EMPTY = 0
    WALL = 1
    AGENT = 2
    TREASURE = 3
    SMALL_TREASURE = 4


    def __init__(
            self, 
            size=20, 
            p_walls=0.5, 
            agent_start=(0,0),

            step_penalty = -1e-3,
            s_treasure_rew = 1,
            treasure_rew = 1_000,
        ):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.size = size
        self.p_walls = p_walls

        self.agent_pos = agent_start
        self.treasure_pos = (size-1, size-1)
        self.s_treasure_pos = (0, size-1)
    
        self.step = 0
        
        self.step_penalty = step_penalty
        self.s_treasure_rew = s_treasure_rew
        self.treasure_rew = treasure_rew

        self._init_grid()


    def __str__(self):

        tiles = {
            self.EMPTY: " . ",
            self.WALL: "███",
            self.AGENT: " a ",
            self.TREASURE: " T ",
            self.SMALL_TREASURE: " t ",
        }

        s = f"STEP {self.step}\n"
        s += tiles[self.WALL] * (self.size + 2) + "\n"

        for r in range(self.size):
            str_row = tiles[self.WALL]
            for c in range(self.size):
                str_row += tiles[self.grid[r,c]]
            s += str_row + tiles[self.WALL] + "\n"
        
        s += tiles[self.WALL] * (self.size + 2)

        return s

    def _init_grid(self):
        for r in range(self.size):
            for c in range(self.size):
                if not (r % 2 == 0):
                   if np.random.rand() < self.p_walls:
                        if (r,c) != self.agent_pos and (r,c) != self.treasure_pos:
                            self.grid[r,c] = self.WALL

        self.grid[self.agent_pos] = self.AGENT
        self.grid[self.treasure_pos] = self.TREASURE
        self.grid[self.s_treasure_pos] = self.SMALL_TREASURE


    def get_actions(self):
        pass
        

    def do_action(self):
        pass
