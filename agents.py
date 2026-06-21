import numpy as np
from enviroment import GridWorld, a2idx

class QLearning(object):

    def __init__(self, gridworld: GridWorld, terminal_states, alpha=0.01):
        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.n_action = 4
        self.alpha = alpha
        self.gamma = gridworld.gamma  # Use the same gamma as the environment

        self.terminal_states = terminal_states

        self.table = {}
        self.__init_table()

    def __init_table(self):
        table = {}
        for i in range(self.gridworld_size):
            for j in range(self.gridworld_size):
                if self.gridworld.grid[(i,j)] != self.gridworld.WALL:
                    for a in self.gridworld.get_legal_actions((i,j)):
                        table[((i,j), tuple(a))] = 0

        # Terminal states have Q-value 0 for all actions
        for a in [self.gridworld.UP, self.gridworld.DOWN,
                  self.gridworld.LEFT, self.gridworld.RIGHT]:
            table[(a2idx(self.gridworld.treasure_pos), a2idx(a))] = 0
            table[(a2idx(self.gridworld.small_treasure_pos), a2idx(a))] = 0

        self.table = table

    def learn_from_episode(self):
        """Update Q-table using the current episode from the gridworld"""
        episode = self.gridworld.get_episode()
        for t in range(len(episode)):
            s, a, s_next, r = episode[t]
            best_next_value = self.best_value(s_next)
            self.table[(s, a)] = (1 - self.alpha) * self.Q(s, tuple(a)) + \
                                          self.alpha * (r + self.gamma * best_next_value)
            

    def Q(self, s, a) -> float:
        s = (int(s[0]), int(s[1]))
        return self.table.get((s, a), 0.0)

    def best_value(self, s) -> float:
        maximum = float("-inf")
        for a in self.gridworld.get_legal_actions(s):
            maximum = max(self.Q(s, tuple(a)), maximum)
        return maximum

    def best_action(self, s):
        """Return the best action for a given state"""
        best_val = float("-inf")
        best_act = None
        for a in self.gridworld.get_legal_actions(s):
            val = self.Q(s, tuple(a))
            if val > best_val:
                best_val = val
                best_act = a
        return best_act