import numpy as np
from enviroment import GridWorld, a2idx
from scipy.optimize import minimize
import math
from copy import deepcopy

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
    def __init__(self, gridworld: GridWorld, terminal_states, alpha=0.01, nu = 50):
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

        self.keys = list(self.table_lambda.keys())
        self.q2idx = {self.keys[i]: i for i in range(len(self.keys))}

        self.nu = nu

    def __init_table(self):
        table = {}
        table_ones = {}
        for i in range(self.gridworld_size):
            for j in range(self.gridworld_size):
                if self.gridworld.grid[(i,j)] != self.gridworld.WALL:
                    for a in self.gridworld.get_legal_actions(np.array([i,j])):
                        table[((i,j), a2idx(a))] = 0
                        table_ones[((i,j), a2idx(a))] = 1e-10

        # Terminal states have Q-value 0 for all actions
        for a in [self.gridworld.UP, self.gridworld.DOWN,
                  self.gridworld.LEFT, self.gridworld.RIGHT]:
            table[(a2idx(self.gridworld.treasure_pos), a2idx(a))] = 0
            table[(a2idx(self.gridworld.small_treasure_pos), a2idx(a))] = 0
            table_ones[(a2idx(self.gridworld.treasure_pos), a2idx(a))] = 1e-10
            table_ones[(a2idx(self.gridworld.small_treasure_pos), a2idx(a))] = 1e-10

        self.table_lambda = table
        self.table_episodes = table_ones
        self.table_rewards = deepcopy(table)

    def learn_from_episode(self):
        """Update Q-table using the current episode from the gridworld"""
        episode = self.gridworld.get_episode()
        for t in range(len(episode)):
            s, a, s_next, r = episode[t]
            #best_next_value = self.best_value(s_next)
            qs = (a2idx(s), a2idx(a))
            self.table_rewards[qs] += r
            self.table_episodes[qs] += 1
        self.find_lambdas()


    def best_action_lambda(self, s):
        """Return the best action for a given state"""
        legal_actions = self.gridworld.get_legal_actions(s)
        moves = np.zeros(len(legal_actions))
        
        for i, a in enumerate(legal_actions):
            qs = (a2idx(s), a2idx(a))
            moves[i] = self.table_lambda[qs] + 1e-8

        idx = np.random.choice(len(moves), p = moves/(sum(moves)))
        return legal_actions[int(idx)]
    
    def is_occupancy_measure(self, lam):
        """Constrained optimisation so that the policy is on the simplex"""

        constraint = 0

        # The occupancy of the initial state is 1
        state = self.gridworld.agent_start
        for a in self.gridworld.get_legal_actions(state):
            idx = self.q2idx[(state, a2idx(a))]
            constraint += lam[idx] 

    def find_lambdas(self):
        np_lambdas = np.array(list(self.table_lambda.values()))
        self.np_reward = np.array(list(self.table_rewards.values()))
        self.np_episodes = np.array(list(self.table_episodes.values()))
                
        result = minimize(
            self.minus_VAPOR_function, 
            x0 = np_lambdas, 
            method = 'L-BFGS-B',
            constraints = {self.is_occupancy_measure},

        )
        
        self.table_lambda = dict(zip(self.keys, result.x))

    def VAPOR_function(self, lambdas):
        
        expected_rewards = self.np_reward / self.np_episodes
        entropy = (self.nu / np.sqrt(self.np_episodes))*np.sqrt(-2*np.log(lambdas))

        return np.dot(lambdas, (expected_rewards + entropy))

    def minus_VAPOR_function(self, lambdas):
        return -self.VAPOR_function(lambdas)