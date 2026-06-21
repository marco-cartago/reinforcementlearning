import numpy as np
import random

def a2idx(a: np.ndarray) -> tuple:
    """Convert a numpy array to a tuple"""
    return (int(a[0]), int(a[1]))

class GridWorld(object):
    
    EMPTY = 0
    WALL = 1
    AGENT = 2
    TREASURE = 3
    SMALL_TREASURE = 4

    UP = np.array([0, 1])
    DOWN = np.array([0, -1])
    LEFT = np.array([-1, 0])
    RIGHT = np.array([1, 0])


    def __init__(
            self, 
            size=20, 
            p_walls=0.5, 
            agent_start=np.array([0,0]),

            step_penalty = -(2**(-10)),
            small_treasure_rew = 1,
            treasure_rew = 1_000,
            sd_treasure = 10,
            sd_small_treasure = 1,

            temperature = 0.1,
            gamma = 0.99,
            random_state = np.random.RandomState(0)
        ):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.size = size
        self.p_walls = p_walls

        self.agent_pos: np.ndarray = agent_start
        self.treasure_pos: np.ndarray = np.array([size-1, size-1])
        self.small_treasure_pos: np.ndarray = np.array([0, size-1])
        self.agent_start = agent_start.copy()
    
        self.step = 0
        self.gamma = gamma
        
        self.step_penalty = step_penalty
        self.small_treasure_rew = small_treasure_rew
        self.treasure_rew = treasure_rew
        self.sd_treasure = sd_treasure
        self.sd_small_treasure = sd_small_treasure

        self.temperature = temperature

        self.is_terminated = False
        self.total_reward = 0

        self.current_episode = []
        self.random_state = random_state

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
        s +="\n"

        return s

    def __init_gridworld(self):
        """Randomly initialize the gridworld given the random state"""

        for r in range(self.size):
            for c in range(self.size):
                if not (r % 2 == 0):
                   if self.random_state.rand() < self.p_walls:
                        cp = np.array((r,c))
                        if not (np.array_equal(cp, self.agent_pos) or np.array_equal(cp, self.treasure_pos)):
                            self.grid[r,c] = self.WALL

        self.grid[a2idx(self.agent_pos)] = self.AGENT
        self.grid[a2idx(self.treasure_pos)] = self.TREASURE
        self.grid[a2idx(self.small_treasure_pos)] = self.SMALL_TREASURE


    def get_actions(self):
        """Given the current agent positon return the legal actions"""

        if self.is_terminated:
            return []
        
        c_pos = self.agent_pos

        s_up, s_down, s_left, s_right = (
            c_pos + self.UP, 
            c_pos + self.DOWN, 
            c_pos + self.LEFT, 
            c_pos + self.RIGHT
        )
        actions = []
       
        if s_up[1] != self.size and self.grid[s_up[0], s_up[1]] != self.WALL:
            actions.append(self.UP)
       
        if s_down[1] != -1 and self.grid[s_down[0], s_down[1]] != self.WALL:
            actions.append(self.DOWN)
       
        if s_left[0] != -1 and self.grid[s_left[0], s_left[1]] != self.WALL:
            actions.append(self.LEFT)
       
        if s_right[0] != self.size and self.grid[s_right[0], s_right[1]] != self.WALL:
            actions.append(self.RIGHT)

        return actions


    def get_legal_actions(self, c_pos: np.ndarray):
        """Given the current agent positon return the legal actions"""

        if self.is_terminated:
            return []

        s_up, s_down, s_left, s_right = (
            c_pos + self.UP, 
            c_pos + self.DOWN, 
            c_pos + self.LEFT, 
            c_pos + self.RIGHT
        )
        actions = []
       
        if s_up[1] != self.size and self.grid[s_up[0], s_up[1]] != self.WALL:
            actions.append(self.UP)
       
        if s_down[1] != -1 and self.grid[s_down[0], s_down[1]] != self.WALL:
            actions.append(self.DOWN)
       
        if s_left[0] != -1 and self.grid[s_left[0], s_left[1]] != self.WALL:
            actions.append(self.LEFT)
       
        if s_right[0] != self.size and self.grid[s_right[0], s_right[1]] != self.WALL:
            actions.append(self.RIGHT)

        return actions

    def get_episode(self) -> list:
        return self.current_episode


    def do_action(self, move: np.ndarray) -> float:
        """Moves the agent in the gridworld and returns the reward for that action, does not check for the legality of the action"""

        # Perform a random action with probability temperature
        if np.random.rand() < self.temperature:
            available_actions = self.get_legal_actions(self.agent_pos)
            if len(available_actions) > 1:
                filtered_actions = [
                    a for a in available_actions if not np.array_equal(a, move)
                ]
                idx = random.randint(0, len(filtered_actions))
                move = available_actions[idx]

        reward = self.step_penalty
        start_agent_pos = self.agent_pos.copy()
        end_agent_pos = self.agent_pos + move

        # Move the agent to the new position
        self.grid[start_agent_pos] = self.EMPTY

        self.agent_pos = end_agent_pos

        if self.grid[a2idx(self.agent_pos)].item() == self.TREASURE:
            self.is_terminated = True
            reward = self.treasure_rew + np.random.normal() * self.sd_treasure
            self.total_reward += reward
        
        
        if self.grid[a2idx(self.agent_pos)].item() == self.SMALL_TREASURE:
            self.is_terminated = True
            reward = self.small_treasure_rew + np.random.normal() * self.sd_small_treasure
            self.total_reward += reward

        sasr = (start_agent_pos, move, end_agent_pos, reward)
        self.current_episode.append(sasr)

        self.grid[self.agent_pos] = self.AGENT
        self.step += 1
        self.total_reward += self.step_penalty * self.gamma**self.step

        return reward
