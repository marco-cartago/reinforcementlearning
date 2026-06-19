import numpy as np
import random

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
            agent_start=(np.int8(0),np.int8(0)),

            step_penalty = -(2**(-10)),
            small_treasure_rew = 1,
            treasure_rew = 1_000,
            sd_treasure = 10,
            sd_small_treasure = 1,

            temperature = 0.1
        ):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.size = size
        self.p_walls = p_walls

        self.agent_pos: tuple = agent_start
        self.treasure_pos: tuple = (size-1, size-1)
        self.small_treasure_pos: tuple = (0, size-1)
    
        self.step = 0
        
        self.step_penalty = step_penalty
        self.small_treasure_rew = small_treasure_rew
        self.treasure_rew = treasure_rew
        self.sd_treasure = sd_treasure
        self.sd_small_treasure = sd_small_treasure

        self.temperature = temperature

        self.is_terminated = False
        self.total_reward = 0

        self.current_episode = []

        self.__init_gridworld()


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

    def __init_gridworld(self):
        for r in range(self.size):
            for c in range(self.size):
                if not (r % 2 == 0):
                   if np.random.rand() < self.p_walls:
                        if (r,c) != self.agent_pos and (r,c) != self.treasure_pos:
                            self.grid[r,c] = self.WALL

        self.grid[self.agent_pos] = self.AGENT
        self.grid[self.treasure_pos] = self.TREASURE
        self.grid[self.small_treasure_pos] = self.SMALL_TREASURE


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


    def get_episode(self):
        return self.current_episode


    def do_action(self, a, move):

        if np.random.rand() < self.temperature and len(a) != 1:
            #Choose a random move with probability temperature
            a.pop(move)
            #Choose with equal probability the new move
            move = random.randint(0, len(a)-1)

        reward = self.step_penalty
        curr_agent_pos = self.agent_pos

        curr_agent_pos_np = np.int8(curr_agent_pos)
        self.grid[curr_agent_pos] = self.EMPTY

        next_agent_pos_np = curr_agent_pos_np + np.int8(a[move])
        next_agent_pos = tuple(next_agent_pos_np)
        self.agent_pos = next_agent_pos

        if self.grid[self.agent_pos] == self.TREASURE:
            self.is_terminated = True
            reward = self.treasure_rew + np.random.normal() * self.sd_treasure
            self.total_reward += reward
        
        
        if self.grid[self.agent_pos] == self.SMALL_TREASURE:
            self.is_terminated = True
            reward = self.s_treasure_rew + np.random.normal() * self.sd_small_treasure
            self.total_reward += reward

        self.grid[self.agent_pos] = self.AGENT
        self.step += 1
        self.total_reward += self.step_penalty
        
        return reward
