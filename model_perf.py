import time

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import Agent, QLearning, SoftQLearning, Vapor
from enviroment import GridWorld
from utils import GridWorldConfig


def num_episode_statistics(
        config: GridWorldConfig, 
        max_steps_per_episode: int,
        episode_step: int,
        max_episodes: int, 
        n_simulation: int
    ):

    # Initialize the environment
    gridworld = GridWorld(config)
    terminal_states = [gridworld.treasure_pos, gridworld.small_treasure_pos]

    # Initialize the agent
    q_agent = QLearning(gridworld, terminal_states, alpha=1)
    show_final_path = True

    episode_rewards = np.zeros(shape=(max_episodes // episode_step, n_simulation)) 

    for n_episodes in range(0, max_episodes, episode_step):
        for sim in range(n_simulation):

            q_agent.init_table() # Resets what the agent has learned
            final_reward = 0

            for episode in tqdm(range(n_episodes)):
                # Reset environment
                gridworld = GridWorld(config)
                gridworld.reset()
                steps = 0
                total_reward = 0
                q_agent.gridworld = gridworld
                q_agent.alpha = 0.1 * (1 - episode / n_episodes)  # Decaying learning rate

                # Run episode
                while (not gridworld.is_terminated) and (steps < max_steps_per_episode):
                    s = gridworld.agent_pos
                    a = q_agent.best_action_epsilon_greedy(s, epsilon=0.5)
                    reward = gridworld.do_action(a)
                    total_reward += reward
                    steps += 1

                # Learn
                q_agent.learn_from_episode()

                # Store episode reward
                final_reward = total_reward

            episode_rewards[n_episodes//episode_step, sim] = final_reward

    return episode_rewards

def save_average_plot(data, filename="plot.png", xlabel="Time", ylabel="Value", title="Average with Variance Bound"):
    """
    Plots the average of multiple series with a shaded area representing the variance.
    """
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    std_dev = np.sqrt(variance)
    x = np.arange(len(mean))
    
    plt.figure(figsize=(10, 6))
    
    # Plot the mean
    plt.plot(x, mean, color='blue', label='Mean', linewidth=2)
    
    # Fill the area
    plt.fill_between(x, mean - std_dev, mean + std_dev, color='blue', alpha=0.2, label='Standard deviation')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    SIZE = 10
    CONFIG: GridWorldConfig = GridWorldConfig(
        size=SIZE,
        p_walls=0.70,
        agent_start=np.array((0, 0)),
        step_penalty=-(2 ** (-10)),
        small_treasure_rew=1e-3,
        treasure_rew=1,
        sd_small_treasure=1e-3,
        sd_treasure=1e-3,
        temperature=0.0,
        gamma=0.995,
        random_state=4
    )
    MAX_STEPS_PER_EPISODE = int(3 * SIZE)
    N_EPISODES = 100

    er = num_episode_statistics(
        CONFIG, 
        MAX_STEPS_PER_EPISODE, 
        episode_step=100, 
        max_episodes=10_000, 
        n_simulation=20
    )
    save_average_plot(er, filename=f"./figures/qlearning_plot_{time.time_ns()}")