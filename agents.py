import numpy as np
from enviroment import GridWorld, a2idx
from scipy.optimize import minimize
import math
from copy import deepcopy
from time import sleep

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
    def __init__(self, gridworld: GridWorld, terminal_states, sigma_prior=100.0, sigma_noise=10.0):
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise

        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.n_action = 4
        self.gamma = gridworld.gamma

        self.terminal_states = terminal_states

        self.table_lambda = {}
        self.table_rewards = {}
        self.table_episodes = {}
        self.legal_states = []
        self.__init_table()

        # FIX 2: costruisci lambda_keys e q2idx PRIMA di creare gli array numpy
        self.lambda_keys = list(self.table_lambda.keys())
        self.q2idx = {k: i for i, k in enumerate(self.lambda_keys)}

        # FIX 1+2: gli array numpy seguono l'ordine di lambda_keys
        self.np_reward = np.array([self.table_rewards[k] for k in self.lambda_keys])
        self.np_episodes = np.array([self.table_episodes[k] for k in self.lambda_keys])

    def __init_table(self):
        for i in range(self.gridworld_size):
            for j in range(self.gridworld_size):
                if (self.gridworld.grid[(i,j)] != self.gridworld.WALL):
                    self.legal_states.append((i,j))
                    legal_actions = self.gridworld.get_legal_actions(np.array([i,j]))
                    n_actions = len(legal_actions)
                    for a in legal_actions:
                        key = ((i,j), a2idx(a))
                        self.table_lambda[key] = (1 - self.gamma) / n_actions
                        self.table_rewards[key] = 0.0
                        self.table_episodes[key] = 1.0

    def learn_from_episode(self):
        episode = self.gridworld.get_episode()
        # Il reward totale dell'episodio
        total_reward = sum(r for _, _, _, r in episode)
        
        # Ogni (s,a) visitata riceve il reward totale dell'episodio (REINFORCE-style)
        for t in range(len(episode)):
            s, a, s_next, r = episode[t]
            qs = (a2idx(s), a2idx(a))
            self.table_rewards[qs] += total_reward
            self.table_episodes[qs] += 1

        self.np_reward = np.array([self.table_rewards[k] for k in self.lambda_keys])
        self.np_episodes = np.array([self.table_episodes[k] for k in self.lambda_keys])
        self.find_lambdas()

    def best_action_lambda(self, s):
        legal_actions = self.gridworld.get_legal_actions(s)
        moves = np.zeros(len(legal_actions))
        for i, a in enumerate(legal_actions):
            qs = (a2idx(s), a2idx(a))
            moves[i] = self.table_lambda.get(qs, 1e-8)
        moves = np.maximum(moves, 1e-8)
        idx = np.random.choice(len(moves), p=moves / moves.sum())
        return legal_actions[int(idx)]

    def occupancy_constraints(self, lam):
        cons = []
        s0 = self.gridworld.agent_start
        start_idx = [self.q2idx[(a2idx(s0), a2idx(a))]
                    for a in self.gridworld.get_legal_actions(s0)]
        cons.append(np.sum(lam[start_idx]) - (1 - self.gamma))  # FIX 2: era - 1.0

        for sp in self.legal_states:
            if (tuple(sp) == tuple(self.gridworld.treasure_pos)
                    or tuple(sp) == tuple(self.gridworld.small_treasure_pos)):
                continue

            lhs = sum(lam[self.q2idx[(a2idx(sp), a2idx(a))]]
                    for a in self.gridworld.get_legal_actions(np.array(sp)))

            rhs = 0.0
            for s in self.legal_states:
                for a in self.gridworld.get_legal_actions(np.array(s)):
                    prob = self.gridworld.get_transition_prob(
                        np.array(s), np.array(a), np.array(sp)
                    )
                    if prob > 0:
                        rhs += prob * lam[self.q2idx[(a2idx(s), a2idx(a))]]

            cons.append(lhs - self.gamma * rhs)  # FIX 2: era lhs - rhs

        return np.array(cons)

    def find_lambdas(self):
        from scipy.optimize import linprog

        x0 = np.array([self.table_lambda[k] for k in self.lambda_keys])
        n = len(x0)

        non_terminal_states = [
            s for s in self.legal_states
            if tuple(s) != tuple(self.gridworld.treasure_pos)
            and tuple(s) != tuple(self.gridworld.small_treasure_pos)
        ]

        all_states = self.legal_states + [
            tuple(self.gridworld.treasure_pos),
            tuple(self.gridworld.small_treasure_pos)
        ]

        # Solo flow constraints, niente normalizzazione separata
        A_eq = np.zeros((len(non_terminal_states), n))
        b_eq = np.zeros(len(non_terminal_states))

        row = 0
        s0 = self.gridworld.agent_start
        for s in non_terminal_states:
            for a in self.gridworld.get_legal_actions(np.array(s)):
                idx = self.q2idx[(a2idx(s), a2idx(a))]
                A_eq[row, idx] += 1.0

            for sp in all_states:
                for a in self.gridworld.get_legal_actions(np.array(sp)):
                    prob = self.gridworld.get_transition_prob(
                        np.array(sp), np.array(a), np.array(s)
                    )
                    if prob > 0:
                        key = (a2idx(sp), a2idx(a))
                        idx = self.q2idx.get(key)
                        if idx is not None:
                            A_eq[row, idx] -= self.gamma * prob

            b_eq[row] = (1 - self.gamma) if np.array_equal(s, s0) else 0.0
            row += 1

        # Feasibility check
        res_feas = linprog(
            c=np.zeros(n),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(1e-12, None)] * n,
            method='highs'
        )
        #print(f"Feasibility: {res_feas.message}")
        if res_feas.status != 0:
            return

        result = minimize(
            self.minus_VAPOR_function,
            x0=x0,
            method="SLSQP",
            bounds=[(1e-12, None)] * n,
            constraints=[{
                "type": "eq",
                "fun": lambda x: A_eq @ x - b_eq,
                "jac": lambda x: A_eq
            }],
            options={"maxiter": 500, "ftol": 1e-9}
        )

        #print(f"Success: {result.success}")
        if result.success:
            self.table_lambda = dict(zip(self.lambda_keys, result.x))

    def VAPOR_function(self, lambdas):
        np_expected_rewards = self.np_reward / self.np_episodes
        posterior_std = self.sigma_prior * self.sigma_noise / np.sqrt(
            self.sigma_noise**2 * self.np_episodes + self.sigma_prior**2
        )
        log_term = np.clip(-2 * np.log(lambdas + 1e-12), 0, None)
        entropy_term = posterior_std * np.sqrt(log_term)
        return np.dot(lambdas, np_expected_rewards + entropy_term)

    def minus_VAPOR_function(self, lambdas):
        return -self.VAPOR_function(lambdas)
    

class VAPOR_variant(object):
    def __init__(self, gridworld: GridWorld, terminal_states, sigma_prior=1.0, sigma_noise=1.0):
        self.sigma_prior = sigma_prior
        self.sigma_noise = sigma_noise

        self.gridworld = gridworld
        self.gridworld_size = self.gridworld.size
        self.n_action = 4
        # self.alpha = alpha
        self.gamma = gridworld.gamma  # Use the same gamma as the environment

        self.terminal_states = terminal_states

        self.table_lambda = {}
        self.table_rewards = {}
        self.table_episodes = {}
        self.legal_states = []
        self.__init_table()

        self.np_reward = np.array(list(self.table_rewards.values()))
        self.np_episodes = np.array(list(self.table_episodes.values()))

        self.lambda_keys = list(self.table_lambda.keys())
        self.q2idx = {self.lambda_keys[i]: i for i in range(len(self.lambda_keys))}
        self.sigma_prior = sigma_prior  # Prior std for rewards
        self.sigma_noise = sigma_noise  # Noise std for rewards
        self.table_expected_rewards = {}  # Precomputed E[r]
        self.__init_table()

    def __init_table(self):
        table = {}
        for i in range(self.gridworld_size):
            for j in range(self.gridworld_size):
                if self.gridworld.grid[(i,j)] not in [self.gridworld.WALL, self.gridworld.SMALL_TREASURE, self.gridworld.TREASURE]:
                    self.legal_states.append((i,j))
                    for a in self.gridworld.get_legal_actions(np.array([i,j])):
                        key = (a2idx((i,j)), a2idx(a))
                        table[key] = 0.0
                        self.table_expected_rewards[key] = 0.0
                        self.table_episodes[key] = 1.0
                        self.table_rewards[key] = 0.0

        self.lambda_keys = list(table.keys())
        self.q2idx = {k: i for i, k in enumerate(self.lambda_keys)}
        self.np_expected_rewards = np.zeros(len(self.lambda_keys))
        self.np_episodes = np.ones(len(self.lambda_keys))  # Start at 1.0
        self.np_lambda = np.ones(len(self.lambda_keys)) * 1e-5  # Small but stable

        # Precompute flow constraints (see above)
        self.__precompute_constraints()

    def learn_from_episode(self):
        episode = self.gridworld.get_episode()
        for t in range(len(episode)):
            s, a, s_next, r = episode[t]
            qs = (a2idx(s), a2idx(a))
            self.table_rewards[qs] += r
            self.table_episodes[qs] += 1
            self.table_expected_rewards[qs] = self.table_rewards[qs] / self.table_episodes[qs]

        # Update numpy arrays
        self.np_expected_rewards = np.array(list(self.table_expected_rewards.values()))
        self.np_episodes = np.array(list(self.table_episodes.values()))
        self.find_lambdas()

    def VAPOR_function(self, lambdas):
        # Posterior std: σ = σ_prior * σ_noise / sqrt(σ_noise² * N + σ_prior²)
        posterior_std = self.sigma_prior * self.sigma_noise / np.sqrt(
            self.sigma_noise**2 * self.np_episodes + self.sigma_prior**2
        )
        entropy_term = posterior_std * np.sqrt(-2 * np.log(lambdas + 1e-12))
        return np.dot(lambdas, self.np_expected_rewards + entropy_term)

    def __precompute_constraints(self):
        # Precompute start-state indices
        s0 = self.gridworld.agent_start
        self.start_indices = [self.q2idx[(a2idx(s0), a2idx(a))]
                             for a in self.gridworld.get_legal_actions(s0)]

        # Precompute flow constraints (see above)
        self.flow_constraints = []
        for sp in self.legal_states:
            if tuple(sp) in [tuple(self.gridworld.treasure_pos), tuple(self.gridworld.small_treasure_pos)]:
                continue
            lhs_indices = [self.q2idx[(a2idx(sp), a2idx(a))] for a in self.gridworld.get_legal_actions(np.array(sp))]
            rhs_indices, rhs_coeffs = [], []
            for s in self.legal_states:
                for a in self.gridworld.get_legal_actions(np.array(s)):
                    prob = self.gridworld.get_transition_prob(np.array(s), np.array(a), np.array(sp))
                    if prob > 0:
                        rhs_indices.append(self.q2idx[(a2idx(s), a2idx(a))])
                        rhs_coeffs.append(prob)
            self.flow_constraints.append((lhs_indices, rhs_indices, rhs_coeffs))