import numpy as np
from enviroment import GridWorld, a2idx
from scipy.optimize import minimize
import math
from copy import deepcopy
from time import sleep, time_ns
import random
import cvxpy as cp

def compute_returns_np(r, gamma):
    r = np.array(r)
    returns = np.zeros_like(r, dtype=float)
    g = 0
    for t in reversed(range(len(r))):
        g = r[t] + gamma * g
        returns[t] = g
    return returns

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


class RepBuffer(object):

    def __init__(self, size: int = 8, seed: int = 0):
        self.size = size
        self.filled = 0
        self.array = np.zeros(size)
        self.rng = np.random.RandomState(seed)

    def add(self, x):
        if self.filled < self.size:
            self.array[self.filled] = x
            self.filled += 1
        else:
            idx = self.rng.randint(self.size)
            self.array[idx] = x

    def mean(self):
        return np.mean(self.array)

    def var(self):
        return np.var(self.array)


class Vapor(object):

    """
    
    """

    def __init__(
        self,
        gridworld: GridWorld,
        terminal_states,
        horizon: int = -1,
        sigma_prior: float = 1.0,
    ):
        # Gridworld structure
        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.gamma = gridworld.gamma
        self.initial_state = a2idx(gridworld.agent_start)
        self.terminal_states = terminal_states
        if horizon == -1:
            self.horizon = self.gridworld_size * 2 - 1
        else:
            self.horizon = horizon

        # Prior paremeters
        self.sigma_prior = sigma_prior

        self.qstate_to_idx = (
            {}
        )  # Mapping from qstate (l, (i,j), a) to position inside the array
        self.legal_qstates = (
            []
        )  # Each element is of the form (l, (i,j), a) (timestep, position, action)


        # Used to keep tab of the last k rewards coming from a particular
        # q-state, so to have, for the bayesian update both a mean and a 
        # variance. This introduces some bias, but the buffer is kept small.
        # It is implemented as a dict from a (l, s, a) tuple to a RepBuffer object.
        self.reward_buff = {}


        # Mapping from qstate (l, (i,j), a) to position inside the array
        self.qstate_to_idx = {}  
        # Each element is of the form (l, (i,j), a) (timestep, position, action)
        self.legal_qstates = []
        self.legal_states = []
        

        self._init_table()  # Initializes the tables
        #print(f"[DEBUG] Found {len(self.legal_qstates)} q-states")

        # Initialize enviroment priors
        self.curr_lambda = np.zeros(len(self.legal_qstates))
        self.curr_reward_mean = np.zeros(len(self.legal_qstates))
        self.curr_reward_variance = np.zeros(len(self.legal_qstates)) + self.sigma_prior


    def _init_table(self, repbuffer_size:int=5):
        """
        Initializes:
            - List of available legal position in the enviroment
            - List of all available q-states for a maximum of L steps

        """
        gw = self.gridworld
        for l in range(self.horizon):
            for i in range(self.gridworld_size):
                for j in range(self.gridworld_size):
                    if gw.grid[(i, j)] != gw.WALL:
                        legal_actions = gw.get_legal_actions(np.array([i, j]))
                        self.legal_states.append((i, j))
                        for a in legal_actions:
                            sa = (l, (i, j), a2idx(a))
                            self.legal_qstates.append(sa)

        # Establish a mapping between state and vector position
        self.qstate_to_idx = {key: idx for idx, key in enumerate(self.legal_qstates)}

        # Initializes buffers for the state rewards
        self.reward_buff = {qs: RepBuffer(size=repbuffer_size, seed=0) for qs in self.legal_qstates}


    def update_env_model(self, lsa_s: list, r_s: list[float], noise: float = 1.0):
        """
        Performs the bayesian update only on those states wich have been visited.
        Internally updates the means and the variances, it:
         1. Estimates mean and reward variance using the `RepBuffer` for that particular q-state
         2. Locally modifies via byesian update `self.curr_reward_variance` and `self.curr_reward_mean` using the estimates
        """

        mu = self.curr_reward_mean
        s = self.curr_reward_variance

        for qs in lsa_s:
            # Obtain corrispondence
            idx = self.qstate_to_idx[qs]

            # Calculate estimates
            mu_p = self.reward_buff[qs].mean()
            s_p = self.reward_buff[qs].var() 
            s_p += noise

            # Update
            self.curr_reward_variance[idx] = (s[idx] * s_p) / (s[idx] + s_p)
            self.curr_reward_mean[idx] = self.curr_reward_variance[idx] * (mu[idx] / s[idx] + mu_p / s_p)


    def lambda_stat_constraint(self, x: cp.Variable):
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

        # \ro(s) = \sum _a \lambda_1(s,a)
        # As the agent deterministically starts always in the same position.
        for s in self.legal_states:
            idxs = [
                indexof((0, s, a2idx(a)))
                for a in self.gridworld.get_legal_actions(np.array(s))
            ]
            constraints.append(cp.sum(x[idxs]) == is_initial(s))

        # \forall s' \forall l \in 1 .. L-1 -> \sum _{a'} \lambda_{l+1}(s', a') = \sum_{s,a} Pr(s' | s, a) \lambda_l(s,a)
        for l in range(self.horizon - 1):
            for sp in self.legal_states:
                s_a_idxs = []
                probs = []
                # We obtain the index of each (l, s, a) pair
                # We save the corresponding probability
                for act in self.gridworld.get_legal_actions(sp):
                    s = np.array(sp) - act

                    # Skip state action combinations that are illegal
                    if ((a2idx(s), a2idx(act)) not in self.legal_qstates):
                        continue

                    for a in self.gridworld.get_legal_actions(s):
                        s_a_idxs.append(indexof((l, a2idx(s), a2idx(a))))
                        probs.append(self.gridworld.get_transition_prob(s, a, sp))

                probs = np.array(probs)
                sp_idxs = [
                    indexof((l + 1, sp, a2idx(ap)))
                    for ap in self.gridworld.get_legal_actions(sp)
                ]
                constraints.append(cp.sum(x[sp_idxs]) == probs.T @ x[s_a_idxs])

        return constraints


    def update_lambda(self, solver: str = "CLARABEL") -> None:
        nv = len(self.legal_qstates) # Number of varaibles
        x = cp.Variable(nv)
        y = cp.Variable(nv) # Auxiluiary variables
        r = self.curr_reward_mean
        s = self.curr_reward_variance

        # Modified objective
        objective = cp.Maximize(cp.sum(cp.multiply(x, r) + y))
        
        # Additionlal axuiliary variable constrints
        y_constr = [
            cp.quad_over_lin(y[i], x[i]) <= 2 * (s[i]**2) * cp.entr(x[i]) 
            for i in range(nv)
        ]
        pos_constraints = [x >= 0, y >= 0]
        constraints = y_constr + pos_constraints + self.lambda_stat_constraint(x)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        self.curr_lambda = x.value


    def learn_from_episode(self, episode, cum_sum:bool=False):
        lsa_s = [
            (l, a2idx(s), a2idx(a)) 
            for l, (s, a, _, _) in zip(range(len(episode)), episode)
        ]
        r_s = [r for (_, _, _, r) in episode]
        returns = compute_returns_np(r_s, self.gamma)

        # Update the buffer of collected rewards
        for qs, r in zip(lsa_s, r_s):
            self.reward_buff[qs].add(r)

        self.update_env_model(lsa_s, r_s) # Change the prior on the enviroment
        self.update_lambda() # Update the env lambdas


    def lamb(self, l: int, s: np.ndarray, a: np.ndarray) -> float:
        idx = self.qstate_to_idx[(l, a2idx(s), a2idx(a))]
        return self.curr_lambda[idx]


    def best_value(self, l, s) -> float:
        """Return the best Q-value for a given state"""
        legal_actions = self.gridworld.get_legal_actions(s)
        maximum = self.lamb(l, s, legal_actions[0])
        for a in legal_actions[1:]:
            maximum = max(self.lamb(l, s, a), maximum)
        return maximum


    def best_action(self, l, s):
        """Return the best action for a given state"""
        legal_actions = self.gridworld.get_legal_actions(s)
        a = legal_actions[0]
        best_val = self.lamb(l, s, a)
        best_act = a
        for a in legal_actions[1:]:
            val = self.lamb(l, s, a)
            if val > best_val:
                best_val = val
                best_act = a
        return best_act


    def sample_action(self, l, s):
        """Sample an action for a given state according to lambda weights"""
        legal_actions = self.gridworld.get_legal_actions(s)
        if len(legal_actions) == 1:
            return legal_actions[0]

        weights = [float(self.lamb(l, s, a)) + 1e-8 for a in legal_actions]
        
        tot_weights = sum(weights)
        weights = [w / tot_weights for w in weights]
        return random.choices(legal_actions, weights=weights, k=1)[0]


    def best_action_epsilon_greedy(self, l, s, epsilon: float = 0.1):
        if np.random.rand() < epsilon:
            actions = self.gridworld.get_legal_actions(s)
            a_idx = np.random.randint(0, len(actions))
            a = actions[a_idx]
        else:
            a = self.best_action(l, s)
        return a
