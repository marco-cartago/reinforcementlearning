import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Callable, Type

from agents import Agent, QLearning, SoftQLearning, Vapor
from enviroment import GridWorld
from utils import GridWorldConfig

from copy import deepcopy

# Simulation functions


def run_single_simulation(
    agent_class: Type[Agent],
    agent_config: dict,
    grid_config: GridWorldConfig,
    max_episodes: int,
    max_steps: int,
    decay_alpha: bool = False
) -> np.ndarray:
    """
    Runs a single simulation for an agent and returns the reward history.
    """
    agent = agent_class(**agent_config)
    agent.init_table()
    
    rewards = np.zeros(max_episodes)

    temp_start = agent.temperature
    temp_end = 1e-4
    
    for episode in range(max_episodes):
        gridworld = GridWorld(grid_config)
        gridworld.reset()
        agent.gridworld = gridworld
        
        # Handle learning rate decay internally if requested
        if decay_alpha and hasattr(agent, 'alpha'):
            agent.alpha = agent_config.get('alpha', 0.1) * (1 - episode / max_episodes)

        agent.temperature = temp_start * (temp_end / temp_start) ** (episode/max_episodes)

        steps = 0
        total_reward = 0
        while (not gridworld.is_terminated) and (steps < max_steps):
            s = gridworld.agent_pos
            a = agent.get_action(s)
            total_reward += gridworld.do_action(a)
            steps += 1
        
        agent.learn_from_episode()
        rewards[episode] = total_reward
        
    return rewards


def run_experiment(
    agent_class: Type[Agent],
    agent_config: dict,
    grid_config: GridWorldConfig,
    max_episodes: int,
    max_steps: int,
    n_simulations: int = 5,
    decay_alpha: bool = False
) -> np.ndarray:
    """
    Runs multiple simulations and returns a matrix of shape (n_simulations, max_episodes).
    """
    results = np.zeros((n_simulations, max_episodes))
    for sim in range(n_simulations):
        results[sim, :] = run_single_simulation(
            agent_class, agent_config, grid_config, max_episodes, max_steps, decay_alpha
        )
    return results

# Plotting Functions


def plot_learning_curve(data: np.ndarray, filename: str, title: str):
    """
    Reward per episode as the agent learns (Fixed Dim, Fixed Episodes).
    data shape: (n_simulations, episodes)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = np.arange(len(mean))

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, color='blue', label='Mean Reward')
    plt.fill_between(x, mean - std, mean + std, color='blue', alpha=0.2, label='Std Dev')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()


def plot_reward_vs_dimension(dims: List[int], results: List[np.ndarray], filename: str, algo: str, n_epsiodes: int = 100):
    """
    Reward against maze dimension (Fixed Episodes).
    results: List of reward arrays (one per dimension), 
    where each array contains final rewards of n_sims.
    """
    means = [np.mean(res) for res in results]
    stds = [np.std(res) for res in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(dims, means, yerr=stds, fmt='-o', color='green', capsize=5)
    plt.xlabel("Maze Dimension")
    plt.ylabel("Final Episode Reward")
    plt.title(f"Performance vs Maze Scale - {algo} (fixed n. episodes {n_epsiodes})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()


def plot_reward_vs_episodes(episode_counts: List[int], results: List[np.ndarray], filename: str):
    """
    Reward against total training episodes (Fixed Dim).
    results: List of reward arrays (one per episode count), 
    containing final rewards of n_sims.
    """
    means = [np.mean(res) for res in results]
    stds = [np.std(res) for res in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(episode_counts, means, yerr=stds, fmt='-o', color='red', capsize=5)
    plt.xlabel("Training Episodes")
    plt.ylabel("Final Episode Reward")
    plt.title("Learning Convergence")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # Base Configurations
    BASE_SIZE = 8
    MAX_EPISODES = 10_000
    DEFAULT_CONFIG = GridWorldConfig(
        size=BASE_SIZE, 
        p_walls=0.70, 
        agent_start=np.array((0, 0)),
        step_penalty=-(2**-10), 
        small_treasure_rew=1e-3, 
        treasure_rew=1,
        sd_small_treasure=1e-3, 
        sd_treasure=1e-3, 
        temperature=0.0,
        gamma=0.995, 
        random_state=4
    )

    # Fresh config
    temp_grid = GridWorld(DEFAULT_CONFIG)

    q_config = {
        "gridworld": temp_grid, 
        "terminal_states": [temp_grid.treasure_pos, temp_grid.small_treasure_pos], 
        "alpha": 1e-1, 
        "epsilon": 1e-1   
    }

    soft_q_config = {
        "gridworld": temp_grid, 
        "terminal_states": [temp_grid.treasure_pos, temp_grid.small_treasure_pos], 
        "alpha": 1e-1, 
        "temperature": 5e-3
    }

    vapor_config = {
        "gridworld": temp_grid, 
        "terminal_states": [temp_grid.treasure_pos, temp_grid.small_treasure_pos], 
        "horizon": MAX_EPISODES,
        "sigma_prior": 1.0,
        "repbuffer_size": 4
    }

    # {model_name: (model's class, model's class config)}
    models_dict = {
        "Q-learning": (QLearning, q_config),
        "Soft-Q-Learning": (SoftQLearning, soft_q_config),
        "VAPOR": (Vapor, vapor_config)
    }

    MODEL = "Soft-Q-Learning"

    # Reward vs Dimension 
    dimensions = [4, 8, 12, 16]
    dim_results = []
    print("Running Dimension Experiment...")
    for d in tqdm(dimensions):
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg.size = d # Update dimension
        # Run experiment and take the reward of the last episode across n_sims
        res = run_experiment(
            models_dict[MODEL][0], 
            models_dict[MODEL][1], 
            cfg, 
            max_episodes=MAX_EPISODES, 
            max_steps=d*3, 
            n_simulations=5
        )
        dim_results.append(res[:, -1]) 
    plot_reward_vs_dimension(
        dimensions, 
        dim_results, 
        f"./figures/dim_study_{time.time_ns()}.png", MODEL, n_epsiodes=MAX_EPISODES)


    # Reward vs Episode Number 
    ep_counts = [10, 50, 100, 500, 1000, 2000, 2500, 5000]
    ep_results = []
    print("Running Episode Count Experiment...")
    for e in tqdm(ep_counts):
        res = run_experiment(
            models_dict[MODEL][0], 
            models_dict[MODEL][1], 
            DEFAULT_CONFIG, 
            max_episodes=e, 
            max_steps=BASE_SIZE*3, 
            n_simulations=5
        )
        ep_results.append(res[:, -1])
    plot_reward_vs_episodes(ep_counts, ep_results, f"./figures/ep_study_{time.time_ns()}.png")


    # Learning Curve (Single Agent Journey) 
    print("Running Learning Curve Simulation...")
    learning_data = run_experiment(
        models_dict[MODEL][0], 
        models_dict[MODEL][1], 
        DEFAULT_CONFIG, 
        max_episodes=MAX_EPISODES, 
        max_steps=BASE_SIZE*3, 
        n_simulations=5, 
        decay_alpha=True
    )
    plot_learning_curve(learning_data, f"./figures/learning_curve_{time.time_ns()}.png", "SoftQLearning Convergence over 1000 Episodes")
