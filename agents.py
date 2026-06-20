import numpy as np
from enviroment import GridWorld

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
            table[(self.gridworld.treasure_pos, tuple(a))] = 0
            table[(self.gridworld.small_treasure_pos, tuple(a))] = 0

        self.table = table

    def learn_from_episode(self):
        """Update Q-table using the current episode from the gridworld"""
        episode = self.gridworld.get_episode()
        for t in range(len(episode)):
            s, a_idx, s_next, r = episode[t]
            a = self.gridworld.get_actions()[a_idx]  # Get the action vector

            if (s, tuple(a)) in self.table:
                if s_next in self.terminal_states:
                    best_next_value = 0
                else:
                    best_next_value = self.best_value(s_next)

                # Update Q-value
                self.table[(s, tuple(a))] = (1 - self.alpha) * self.Q(s, tuple(a)) + \
                                          self.alpha * (r + self.gamma * best_next_value)

    def Q(self, s, a) -> float:
        a_tuple = tuple(a) if not isinstance(a, tuple) else a
        return self.table.get((s, a_tuple), 0.0)

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