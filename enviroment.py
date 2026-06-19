import numpy as np

class GridWorld(object):
    
    EMPTY = 0
    WALL = 1
    AGENT = 2
    TREASURE = 3
    SMALL_TREASURE = 4

    UP = [0, 1]
    DOWN = [0, -1]
    LEFT = [-1, 0]
    RIGHT = [1, 0]


    def __init__(
            self, 
            size=20, 
            p_walls=0.5, 
            agent_start=(0,0),

            step_penalty = -(2**(-10)),
            s_treasure_rew = 1,
            treasure_rew = 1_000,

            temperature = 0.1
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

        self.temperature = temperature

        self.is_terminated = False
        self.total_reward = 0

        self._init_grid()


    def __str__(self):

        tiles = {
            self.EMPTY: " ⋅ ",
            self.WALL: "███",
            self.AGENT: "🤖 ",
            self.TREASURE: "💰 ",
            self.SMALL_TREASURE: "🪙 ",
        }

        s = f"\033[H\nSTEP {self.step}\nTotal reward: {self.total_reward}\n"
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
        if self.is_terminated:
            return []
        np_pos = np.int8(self.agent_pos)
        STATE_UP, STATE_DOWN, STATE_LEFT, STATE_RIGHT = np_pos + self.UP, np_pos + self.DOWN, np_pos + self.LEFT, np_pos + self.RIGHT
        actions = {1: self.UP, 2: self.DOWN, 3: self.LEFT, 4: self.RIGHT}
        if STATE_UP[1] == self.size or self.grid[STATE_UP[0], STATE_UP[1]] == self.WALL:
            actions.pop(1)
        if STATE_DOWN[1] == -1 or self.grid[STATE_DOWN[0], STATE_DOWN[1]] == self.WALL:
            actions.pop(2)
        if STATE_LEFT[0] == -1 or self.grid[STATE_LEFT[0], STATE_LEFT[1]] == self.WALL:
            actions.pop(3)
        if STATE_RIGHT[0] == self.size or self.grid[STATE_RIGHT[0], STATE_RIGHT[1]] == self.WALL:
            actions.pop(4)

        return [move for _, move in actions.items()]

    def do_action(self, action):
        agent_pos = np.int8(self.agent_pos)
        self.grid[self.agent_pos] = self.EMPTY
        agent_pos += np.int8(action)
        self.agent_pos = tuple(agent_pos)
        if self.grid[self.agent_pos] == self.TREASURE:
            self.is_terminated = True
            self.total_reward += self.treasure_rew
            self.grid[self.agent_pos] = self.AGENT
            return self.treasure_rew
        if self.grid[self.agent_pos] == self.SMALL_TREASURE:
            self.is_terminated = True
            self.total_reward += self.s_treasure_rew
            self.grid[self.agent_pos] = self.AGENT
            return self.s_treasure_rew
        self.grid[self.agent_pos] = self.AGENT
        self.step += 1
        self.total_reward += self.step_penalty
        return self.step_penalty
