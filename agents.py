import numpy as np
from enviroment import GridWorld, a2idx
from scipy.optimize import minimize
import math

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
                    for a in self.gridworld.get_legal_actions(np.array([i,j])):
                        table[((i,j), a2idx(a))] = 0

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
            qs = (a2idx(s), a2idx(a))
            qv = self.Q(s, a)
            self.table[qs] = (
                (1 - self.alpha) * qv + self.alpha * (r + self.gamma * best_next_value)
            )
            

    def Q(self, s: np.ndarray, a: np.ndarray) -> float:
        return self.table.get((a2idx(s), a2idx(a)), 0.0)

    def best_value(self, s) -> float:
        """Return the best Q-value for a given state"""
        legal_actions = self.gridworld.get_legal_actions(s)
        maximum = self.Q(s, legal_actions[0])
        for a in legal_actions[1:]:
            maximum = max(self.Q(s, a), maximum)
        return maximum

    def best_action(self, s):
        """Return the best action for a given state"""
        legal_actions = self.gridworld.get_legal_actions(s)
        a = legal_actions[0]
        best_val = self.Q(s, a)
        best_act = a
        for a in legal_actions[1:]:
            val = self.Q(s, a)
            if val > best_val:
                best_val = val
                best_act = a
        return best_act
    
    def best_action_epsilon_greedy(self, s, epsilon: float=0.1):
        if np.random.rand() < epsilon:
            actions = self.gridworld.get_legal_actions(s)
            a_idx = np.random.randint(0, len(actions))
            a = actions[a_idx]
        else:
            a = self.best_action(s)
        
        return a

class VAPOR(object):
    def __init__(self, gridworld: GridWorld, terminal_states, alpha=0.01, nu = 100):
        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.n_action = 4
        self.alpha = alpha
        self.gamma = gridworld.gamma  # Use the same gamma as the environment

        self.terminal_states = terminal_states

        self.table_lambda = {}
        self.table_rewards = {}
        self.table_episodes = {}
        self.__init_table()

        self.np_reward = np.array(list(self.table_rewards.values()))
        self.np_episodes = np.array(list(self.table_episodes.values()))

        self.nu = nu

    def __init_table(self):
        table = {}
        for i in range(self.gridworld_size):
            for j in range(self.gridworld_size):
                if self.gridworld.grid[(i,j)] != self.gridworld.WALL:
                    for a in self.gridworld.get_legal_actions(np.array([i,j])):
                        table[((i,j), a2idx(a))] = 1

        # Terminal states have Q-value 0 for all actions
        for a in [self.gridworld.UP, self.gridworld.DOWN,
                  self.gridworld.LEFT, self.gridworld.RIGHT]:
            table[(a2idx(self.gridworld.treasure_pos), a2idx(a))] = 1
            table[(a2idx(self.gridworld.small_treasure_pos), a2idx(a))] = 1

        self.table_lambda = table
        self.table_episodes = table
        self.table_episodes = table

    def learn_from_episode(self):
        """Update Q-table using the current episode from the gridworld"""
        episode = self.gridworld.get_episode()
        for t in range(len(episode)):
            s, a, s_next, r = episode[t]
            #best_next_value = self.best_value(s_next)
            qs = (a2idx(s), a2idx(a))
            self.table_rewards[qs] += r
            self.table_episodes[qs] += 1


    def best_action_lambda(self, s):
        """Return the best action for a given state"""
        legal_actions = self.gridworld.get_legal_actions(s)
        moves = np.zeros(len(legal_actions))
        for i, a in enumerate(legal_actions):
            qs = (a2idx(s), a2idx(a))
            moves[i] = self.table_lambda[qs]
        return np.random.choice(moves, p = moves/sum(moves))
    
    def find_lambdas(self):
        chiavi = list(self.table_lambda.keys())
        np_lambdas = np.array(list(self.table_lambda.values()))
        self.np_reward = np.array(list(self.table_rewards.values()))
        self.np_episodes = np.array(list(self.table_episodes.values()))
        result = minimize(self.minus_VAPOR_function, x0 = np_lambdas, method = 'BFGS')
        self.table_lambda = dict(zip(chiavi, result.x))

    def VAPOR_function(self, lambdas):
        return np.dot(lambdas, (self.np_reward + (self.nu / self.np_episodes)*np.sqrt(-2*np.log(lambdas))))
        pass

    def minus_VAPOR_function(self, lambdas):
        return -self.VAPOR_function(lambdas)