import numpy as np
import cvxpy as cp
import random


def a2idx(a):
    return int(a[0]) if isinstance(a, (np.ndarray, list)) else int(a)

class Agent(object):

    def __init__(self, *args, **kwargs):
        """Initialization of the agent"""
        raise NotImplementedError

    def init_table(self, *args, **kwargs):
        """
        Initalizes from scratch all of the state 
        and learning parameters of the model in question.
        """
        raise NotImplementedError

    def learn_from_episode(self, *args, **kwargs):
        """For learning from an episode"""
        raise NotImplementedError

    def get_action(self, *args, **kwargs):
        """
        Preferred way for the model to generate an 
        action in the enviroment during training
        """
        raise NotImplementedError

    def best_action(self, *args, **kwargs):
        """Used to generate actions during the evauation of the model"""
        raise NotImplementedError


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
        if self.filled == 0:
            return 0.0
        return np.mean(self.array[:self.filled])

    def var(self):
        if self.filled == 0:
            return 0.0
        return np.var(self.array[:self.filled])


class QLearningBandit(Agent):
    def __init__(self, bandit, alpha=0.01, epsilon=0.2):
        self.bandit = bandit
        self.size = bandit.size
        self.alpha = alpha
        self.epsilon = epsilon
        
        # In MAB, there is only one state: 0
        self.table = {}
        self.init_table()

    def init_table(self):
        # Initialize Q-values for all arms in the single starting state
        self.table = {(0, a): 0.0 for a in range(self.size)}

    def learn_from_episode(self):
        """Update Q-table using the current episode.
        Since the episode terminates immediately, target = reward.
        """
        episode = self.bandit.get_episode()
        for (s, a, s_next, r) in episode:
            # In a Bandit, there is no future value (s_next is terminal)
            # Target is just the immediate reward r
            a_idx = a2idx(a)
            qv = self.Q(0, a_idx)
            self.table[(0, a_idx)] = (1 - self.alpha) * qv + self.alpha * r

    def Q(self, s: int, a: int) -> float:
        return self.table.get((s, a), 0.0)

    def best_value(self, s: int) -> float:
        """Return the best Q-value for the current state"""
        return max([self.Q(s, a) for a in range(self.size)])

    def best_action(self, s: int):
        """Return the best action for the current state"""
        qs = [self.Q(s, a) for a in range(self.size)]
        return np.array([np.argmax(qs)])

    def best_action_epsilon_greedy(self, s: int = 0, epsilon: float | None = None):
        # Use provided epsilon or fallback to the default
        eps = epsilon if epsilon is not None else self.epsilon
        
        if np.random.rand() < eps:
            # Explore: pick a random arm
            a_idx = np.random.randint(0, self.size)
            return np.array([a_idx])
        else:
            # Exploit: pick the best arm
            return self.best_action(s)

    def get_action(self, s=0, **kwargs):
        return self.best_action_epsilon_greedy(s, **kwargs)


class VaporBandit(Agent):
    def __init__(
        self,
        bandit, 
        horizon: int = 1,
        sigma_prior: float = 1.0,
        repbuffer_size: int = 4
    ):
        self.bandit = bandit
        self.size = bandit.size
        self.gamma = bandit.gamma
        self.horizon = horizon
        self.repbuffer_size = repbuffer_size
        self.sigma_prior = sigma_prior

        # In MAB, the only state is the start state [0]
        self.initial_state = 0 
        self.legal_states = [0]
        
        # Q-states are (timestep, state, action) -> (0, 0, action_idx)
        self.legal_qstates = [(0, 0, a) for a in range(self.size)]
        self.qstate_to_idx = {qs: idx for idx, qs in enumerate(self.legal_qstates)}

        # Buffers and parameters
        self.reward_buff = {qs: RepBuffer(size=self.repbuffer_size, seed=i) 
                            for i, qs in enumerate(self.legal_qstates)}
        
        self.curr_lambda = np.zeros(len(self.legal_qstates))
        self.curr_reward_mean = np.zeros(len(self.legal_qstates))
        self.curr_reward_variance = np.zeros(len(self.legal_qstates)) + self.sigma_prior

        self._problem_is_initialized = False

    def update_env_model(self, lsa_s: list, r_s: list[float], noise: float = 1e-7):
        for qs, r in zip(lsa_s, r_s):
            self.reward_buff[qs].add(r)

        mu = self.curr_reward_mean
        s = self.curr_reward_variance

        for qs in lsa_s:
            idx = self.qstate_to_idx[qs]
            mu_p = self.reward_buff[qs].mean()
            s_p = self.reward_buff[qs].var() + noise

            # Bayesian update for mean and variance
            self.curr_reward_variance[idx] = (s[idx] * s_p) / (s[idx] + s_p)
            self.curr_reward_mean[idx] = self.curr_reward_variance[idx] * (mu[idx] / s[idx] + mu_p / s_p)

    def lambda_stat_constraint(self, x: cp.Variable) -> list[cp.Constraint]:
        # In MAB, the only constraint is that the probabilities of all arms sum to 1
        # \sum_a \lambda(0, 0, a) = 1
        idxs = [self.qstate_to_idx[(0, 0, a)] for a in range(self.size)]
        return [cp.sum(x[idxs]) == 1]

    def update_lambda(self, solver: str = "CLARABEL", verbose: bool = False) -> None:
        if not self._problem_is_initialized:
            nv = len(self.legal_qstates)
            self._x = cp.Variable(nv, bounds=[0, 1])
            self._y = cp.Variable(nv, bounds=[0, 1])
            self._r = cp.Parameter(nv)
            self._s = cp.Parameter(nv, nonneg=True)
            
            self._objective = cp.Maximize(cp.sum(cp.multiply(self._x, self._r) + self._y))
            self._t = cp.Variable(nv, nonneg=True)

            entropy_constr = [self._t <= 2 * cp.multiply(self._s, cp.entr(self._x))]
            X = cp.vstack([2 * self._y, self._x - self._t])
            soc_constr = [cp.SOC(self._x + self._t, X, axis=0)]      

            self._constraints = soc_constr + entropy_constr + self.lambda_stat_constraint(self._x)
            self._problem = cp.Problem(self._objective, self._constraints)
            self._problem_is_initialized = True

        self._r.value = self.curr_reward_mean
        self._s.value = self.curr_reward_variance
        
        try:
            self._problem.solve(solver=solver, tol_gap_abs=1e-5, verbose=verbose)
        except Exception as e:
            print(f"Optimization error: {e}")

        if self._x.value is not None:
            self.curr_lambda = self._x.value

    def learn_from_episode(self):
        episode = self.bandit.get_episode()
        # Episode format: (agent_pos, move, end_pos, reward)
        # Map to: (timestep, state, action_idx)
        lsa_s = [(0, 0, a2idx(step[1])) for step in episode]
        r_s = [step[3] for step in episode]

        self.update_env_model(lsa_s, r_s)
        self.update_lambda()

    def lamb(self, l: int, s: int, a: int) -> float:
        return self.curr_lambda[self.qstate_to_idx[(l, s, a)]]

    def sample_action(self, l, s, eps=1e-8):
        # Weights for each arm
        weights = [float(self.lamb(l, s, a)) + eps for a in range(self.size)]
        tot = sum(weights)
        weights = [w / tot for w in weights]
        # Return as np.array to satisfy bandit.do_action([0])
        return np.array([random.choices(range(self.size), weights=weights, k=1)[0]])

    def get_action(self, l=0, s=0, **kwargs):
        return self.sample_action(l, s, **kwargs)


class SoftQLearningBandit(Agent):
    def __init__(self, bandit, alpha=0.1, temperature=1.0):
        self.bandit = bandit
        self.alpha = alpha
        self.temperature = temperature
        self.size = bandit.size
        # Table maps (state, action_idx) -> value. State is always 0.
        self.table = {(0, a): 0.0 for a in range(self.size)}

    def Q(self, s: int, a: int) -> float:
        return self.table.get((s, a), 0.0)

    def soft_value(self, s: int) -> float:
        qs = np.array([self.Q(s, a) for a in range(self.size)])
        m = np.max(qs / self.temperature)
        return self.temperature * (m + np.log(np.sum(np.exp(qs / self.temperature - m))))

    def sample_action(self, s: int):
        qs = np.array([self.Q(s, a) for a in range(self.size)])
        v = self.soft_value(s)
        probs = np.exp((qs - v) / self.temperature)
        probs /= probs.sum()
        idx = np.random.choice(self.size, p=probs)
        return np.array([idx])

    def get_action(self, s=0, **kwargs):
        return self.sample_action(s)

    def learn_from_episode(self):
        episode = self.bandit.get_episode()
        for (s, a, s_next, r) in episode:
            # In MAB, s_next is always terminal, so target is just the reward
            target = r 
            a_idx = a2idx(a)
            # Update Q(0, a)
            self.table[(0, a_idx)] = (1 - self.alpha) * self.Q(0, a_idx) + self.alpha * target

    def best_action(self, s=0):
        qs = [self.Q(s, a) for a in range(self.size)]
        return np.array([np.argmax(qs)])
