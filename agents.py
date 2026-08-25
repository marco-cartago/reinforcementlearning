import numpy as np
from enviroment import GridWorld, a2idx
from scipy.optimize import minimize
import math
from copy import deepcopy
from time import sleep
import cvxpy as cv


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
                if self.gridworld.grid[(i, j)] != self.gridworld.WALL:
                    for a in self.gridworld.get_legal_actions(np.array([i, j])):
                        table[((i, j), a2idx(a))] = 0

        # Terminal states have Q-value 0 for all actions
        for a in [
            self.gridworld.UP,
            self.gridworld.DOWN,
            self.gridworld.LEFT,
            self.gridworld.RIGHT,
        ]:
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
            self.table[qs] = (1 - self.alpha) * qv + self.alpha * (
                r + self.gamma * best_next_value
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

    def best_action_epsilon_greedy(self, s, epsilon: float = 0.1):
        if np.random.rand() < epsilon:
            actions = self.gridworld.get_legal_actions(s)
            a_idx = np.random.randint(0, len(actions))
            a = actions[a_idx]
        else:
            a = self.best_action(s)

        return a


class VAPOR(object):

    def __init__(
        self, gridworld: GridWorld, terminal_states, horizon: int, sigma_prior: float = 100.0, sigma_noise: float = 10.0,
    ):
        # Gridworld structure
        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.gamma = gridworld.gamma
        self.initial_state = a2idx(gridworld.agent_start)
        self.terminal_states = terminal_states
        self.horizon = horizon

        # Prior paremeters
        self.sigma_noise = sigma_noise
        self.sigma_prior = sigma_prior

        # Definition of lambda, r, sigma
        self.qtable_lambda = {}
        self.qtable_r = {}
        self.qtable_sigma = {}

        self.qstate_to_idx = {}

        self.legal_qstates = []  # Each element is of the form (l, (i,j), a) (timestep, position, action)
        self.legal_states = []
        self.n_actions = []

        self.init_table()        # Initializes the tables

    def init_table(self):
        gw = self.gridworld

        for l in range(self.horizon):
            for i in range(self.gridworld_size):
                for j in range(self.gridworld_size):

                    if gw.grid[(i, j)] != gw.WALL:
                        legal_actions = gw.get_legal_actions(np.array([i, j]))
                        self.legal_states.append((i,j))

                        for a in legal_actions:
                            sa = (l, (i, j), a2idx(a))
                            self.legal_qstates.append(sa)

        # Establish a mapping between state and vector position
        self.qstate_to_idx = {key: idx for key, idx in enumerate(self.qtable_lambda.keys())}

        # Normalize the lambda vector so that

    def lambda_stat_constraint(self, x: cv.Variable):
        """
        Imposess the stationarity constraint on the varaible passed as input.
        Assumes that the structure of `x` is the same as the one of
         - `self.qtable_lambda`
         - `self.qtable_r`

        In terms of sequential order of the unrolled states in list form.
        """
        constraints = []
        indexof = lambda s: self.qstate_to_idx[s]
        is_initial = lambda s: 1 if s == self.initial_state else 0

        # Non negativity constraint (a positive constraint ;))
        constraints.append(x >= 0)

        # \ro(s) = \sum _a \lambda_1(s,a)
        # As the agent deterministically starts always in the same position.
        for s in self.legal_states:
            idxs = [indexof((0, s, a2idx(a))) for a in self.gridworld.get_legal_actions(np.array(s))]
            constraints.append(cv.sum(x[idxs]) == is_initial(s))

        # \forall s' \forall l \in 1 .. L-1 -> \sum _{a'} \lambda_{l+1}(s', a') = \sum_{s,a} Pr(s' | s, a) \lambda_l(s,a)
        for l in range(self.horizon-1):
            for sp in self.legal_states:

                s_a_idxs = []
                probs = []
                # We obtain the index of each (l, s, a) pair
                # We save the corresponding probability
                for act in self.gridworld.get_legal_actions(sp):
                    s = np.array(sp) - act
                    for a in self.gridworld.get_legal_actions(s):
                        s_a_idxs.append( indexof( (l, a2idx(s), a2idx(a)) ) )
                        probs.append(self.gridworld.get_transition_prob(s, a, sp))

                probs = np.array(probs)

                sp_idxs = [indexof((l+1, sp, a2idx(ap))) for ap in self.gridworld.get_legal_actions(sp)]
                constraints.append(
                    cv.sum(x[sp_idxs]) == probs.T @ x[s_a_idxs]
                )

        return constraints
            


