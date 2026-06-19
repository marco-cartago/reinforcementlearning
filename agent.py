
import numpy as np
from enviroment import GridWorld

class QLearning(object):

    def __init__(self, gridworld: GridWorld, terminal_states, alpha=0.01):
        
        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.n_action = 4
        self.alpha = alpha

        self.terminal_states = terminal_states
        
        self.table = {}
        self.__init_table()

    def __init_table(self):
        table = {}
        for i in range(self.gridworld_size):
            for j in range(self.gridworld_size):
                if self.gridworld.grid[(i,j)] != self.gridworld.WALL:
                    for a in self.gridworld.get_legal_actions((i,j)):
                        table[((i,j),a)] = np.random.randn()

        for a in range(self.n_action):
            table[(self.gridworld.treasure_pos, a)] = 0
            table[(self.gridworld.small_treasure_pos, a)] = 0

        self.table = table

    def learn_from_episode(self):
        pass

    def Q(self, s, a) -> float:
        return self.table[(s,a)]

    def best_value(self, s) -> float:
        maximum = float("-inf")
        for a in self.gridworld.get_legal_actions(s):
            maximum = max(self.Q(s,a), maximum)
        return maximum

        